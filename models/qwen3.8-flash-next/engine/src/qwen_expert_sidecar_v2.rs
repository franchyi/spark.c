//! Layer-local SoA-v2 expert sidecar for FlashInfer fused MoE.
//!
//! Unlike the legacy expert-major AoS file, each record is one complete layer
//! containing eight contiguous planes.  W13 weights and scales are emitted as
//! `[up; gate]`, which is the order consumed by SGLang/FlashInfer's
//! `CompressedTensorsW4A4Nvfp4MoE` path.  The builder streams one source tensor
//! at a time, so producing the 63.282-GiB artifact never requires a layer-sized
//! staging allocation.
//!
//! File layout: `[4096-byte header][48 CRC64s][padding][48 layer records]`.
//! Every layer record is 1,415,585,792 bytes and contains:
//!
//! 1. W13 U8 weights `[512,1280,1280]`, expert-major `[up;gate]`;
//! 2. W2 U8 weights `[512,2560,320]`;
//! 3. W13 CUTLASS-swizzled scales `[512,1280,160]`, `[up;gate]`;
//! 4. W2 CUTLASS-swizzled scales `[512,2560,40]`;
//! 5. four F32 `[512]` scalar planes in input-quant, alpha order for W13/W2.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::checkpoint::{FlashNextCheckpoint, SafetensorLocation};
use crate::qwen_expert_sidecar::{
    ExpertSidecarBuildReport, ExpertSidecarError, ExpertSidecarProgress,
    expert_sidecar_source_fingerprint,
};

pub const EXPERT_SIDECAR_V2_VERSION: u32 = 2;
pub const EXPERT_SIDECAR_V2_HEADER_BYTES: u64 = 4096;
pub const EXPERT_SIDECAR_V2_LAYERS: u32 = 48;
pub const EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER: u32 = 512;

pub const EXPERT_SIDECAR_V2_W13_WEIGHT_BYTES_PER_EXPERT: u64 = 1_638_400;
pub const EXPERT_SIDECAR_V2_W2_WEIGHT_BYTES_PER_EXPERT: u64 = 819_200;
pub const EXPERT_SIDECAR_V2_W13_SCALE_BYTES_PER_EXPERT: u64 = 204_800;
pub const EXPERT_SIDECAR_V2_W2_SCALE_BYTES_PER_EXPERT: u64 = 102_400;

pub const EXPERT_SIDECAR_V2_W13_WEIGHT_OFFSET: u64 = 0;
pub const EXPERT_SIDECAR_V2_W13_WEIGHT_BYTES: u64 =
    EXPERT_SIDECAR_V2_W13_WEIGHT_BYTES_PER_EXPERT
        * EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64;
pub const EXPERT_SIDECAR_V2_W2_WEIGHT_OFFSET: u64 =
    EXPERT_SIDECAR_V2_W13_WEIGHT_OFFSET + EXPERT_SIDECAR_V2_W13_WEIGHT_BYTES;
pub const EXPERT_SIDECAR_V2_W2_WEIGHT_BYTES: u64 =
    EXPERT_SIDECAR_V2_W2_WEIGHT_BYTES_PER_EXPERT
        * EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64;
pub const EXPERT_SIDECAR_V2_W13_SCALE_OFFSET: u64 =
    EXPERT_SIDECAR_V2_W2_WEIGHT_OFFSET + EXPERT_SIDECAR_V2_W2_WEIGHT_BYTES;
pub const EXPERT_SIDECAR_V2_W13_SCALE_BYTES: u64 =
    EXPERT_SIDECAR_V2_W13_SCALE_BYTES_PER_EXPERT
        * EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64;
pub const EXPERT_SIDECAR_V2_W2_SCALE_OFFSET: u64 =
    EXPERT_SIDECAR_V2_W13_SCALE_OFFSET + EXPERT_SIDECAR_V2_W13_SCALE_BYTES;
pub const EXPERT_SIDECAR_V2_W2_SCALE_BYTES: u64 =
    EXPERT_SIDECAR_V2_W2_SCALE_BYTES_PER_EXPERT
        * EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64;
pub const EXPERT_SIDECAR_V2_W13_INPUT_SCALE_QUANT_OFFSET: u64 =
    EXPERT_SIDECAR_V2_W2_SCALE_OFFSET + EXPERT_SIDECAR_V2_W2_SCALE_BYTES;
pub const EXPERT_SIDECAR_V2_W13_ALPHA_OFFSET: u64 = EXPERT_SIDECAR_V2_W13_INPUT_SCALE_QUANT_OFFSET
    + EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64 * 4;
pub const EXPERT_SIDECAR_V2_W2_INPUT_SCALE_QUANT_OFFSET: u64 =
    EXPERT_SIDECAR_V2_W13_ALPHA_OFFSET + EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64 * 4;
pub const EXPERT_SIDECAR_V2_W2_ALPHA_OFFSET: u64 =
    EXPERT_SIDECAR_V2_W2_INPUT_SCALE_QUANT_OFFSET
        + EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64 * 4;
pub const EXPERT_SIDECAR_V2_LAYER_BYTES: u64 = EXPERT_SIDECAR_V2_W2_ALPHA_OFFSET
    + EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER as u64 * 4;

const _: () = assert!(EXPERT_SIDECAR_V2_W13_WEIGHT_BYTES == 838_860_800);
const _: () = assert!(EXPERT_SIDECAR_V2_W2_WEIGHT_BYTES == 419_430_400);
const _: () = assert!(EXPERT_SIDECAR_V2_W13_SCALE_BYTES == 104_857_600);
const _: () = assert!(EXPERT_SIDECAR_V2_W2_SCALE_BYTES == 52_428_800);
const _: () = assert!(EXPERT_SIDECAR_V2_LAYER_BYTES == 1_415_585_792);
const _: () = assert!(EXPERT_SIDECAR_V2_LAYER_BYTES % 16 == 0);

const MAGIC: [u8; 16] = *b"SPARKQWSOAV2\0\0\0\0";
const LAYOUT_NAME: &[u8] = b"qwen3.8-flash-next-nvfp4-layer-soa-v2";
const STATE_BUILDING: u32 = 0;
const STATE_COMPLETE: u32 = 1;
const CHECKSUM_BYTES: u64 = 8;

const VERSION_OFFSET: usize = 16;
const HEADER_BYTES_OFFSET: usize = 20;
const STATE_OFFSET: usize = 24;
const LAYERS_OFFSET: usize = 28;
const EXPERTS_OFFSET: usize = 32;
const RECORD_BYTES_OFFSET: usize = 40;
const RECORD_COUNT_OFFSET: usize = 48;
const CHECKSUMS_OFFSET_OFFSET: usize = 56;
const RECORDS_OFFSET_OFFSET: usize = 64;
const TOTAL_BYTES_OFFSET: usize = 72;
const COMMITTED_RECORDS_OFFSET: usize = 80;
const SOURCE_FINGERPRINT_OFFSET: usize = 88;
const HEADER_CHECKSUM_OFFSET: usize = 120;
const LAYOUT_NAME_OFFSET: usize = 128;
const CRC64_TABLE: [u64; 256] = crc64_table();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertSidecarV2Header {
    pub state_complete: bool,
    pub layer_bytes: u64,
    pub layer_count: u64,
    pub checksums_offset: u64,
    pub layers_offset: u64,
    pub total_bytes: u64,
    pub committed_layers: u64,
    pub source_fingerprint: [u8; 32],
}

impl ExpertSidecarV2Header {
    pub fn layer_file_offset(&self, layer: u32) -> Result<u64, ExpertSidecarError> {
        if layer >= EXPERT_SIDECAR_V2_LAYERS {
            return Err(ExpertSidecarError::Invalid(format!(
                "Qwen SoA-v2 layer {layer} is out of range"
            )));
        }
        self.layers_offset
            .checked_add(
                u64::from(layer)
                    .checked_mul(self.layer_bytes)
                    .ok_or(ExpertSidecarError::IntegerOverflow)?,
            )
            .ok_or(ExpertSidecarError::IntegerOverflow)
    }

    pub fn layers_bytes(&self) -> Result<u64, ExpertSidecarError> {
        self.layer_count
            .checked_mul(self.layer_bytes)
            .ok_or(ExpertSidecarError::IntegerOverflow)
    }
}

pub fn build_expert_sidecar_v2(
    checkpoint: &FlashNextCheckpoint,
    output: &Path,
    mut progress: impl FnMut(ExpertSidecarProgress),
) -> Result<ExpertSidecarBuildReport, ExpertSidecarError> {
    validate_checkpoint_geometry(checkpoint)?;
    let fingerprint = expert_sidecar_source_fingerprint(checkpoint)?;
    let expected = expected_header(fingerprint, false, 0)?;

    if output.exists() {
        let header = read_header(output)?;
        validate_header(&header, &expected, true)?;
        validate_file_size(output, &header)?;
        return Ok(report(&header, header.layer_count, true));
    }

    let partial = partial_path(output)?;
    let mut file = if partial.exists() {
        OpenOptions::new().read(true).write(true).open(&partial)?
    } else {
        let mut created = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&partial)?;
        created.set_len(expected.total_bytes)?;
        write_header(&mut created, &expected)?;
        created.sync_data()?;
        created
    };
    if file.metadata()?.len() != expected.total_bytes {
        return Err(ExpertSidecarError::Invalid(format!(
            "partial SoA-v2 sidecar {} has {} bytes, expected {}",
            partial.display(),
            file.metadata()?.len(),
            expected.total_bytes
        )));
    }
    let mut header = read_header_from(&file)?;
    validate_header(&header, &expected, false)?;
    recover_last_committed_layer(&mut file, &mut header)?;
    let resumed_layers = header.committed_layers;
    progress(ExpertSidecarProgress {
        completed_records: resumed_layers,
        total_records: header.layer_count,
        resumed_records: resumed_layers,
    });

    let mut source = ExpertSource::new(checkpoint);
    while header.committed_layers < header.layer_count {
        let layer = u32::try_from(header.committed_layers)
            .map_err(|_| ExpertSidecarError::IntegerOverflow)?;
        let layer_offset = header.layer_file_offset(layer)?;
        let checksum = source.pack_layer(&file, layer_offset, layer)?;
        write_checksum(&file, &header, u64::from(layer), checksum)?;
        file.sync_data()?;
        header.committed_layers += 1;
        write_header(&mut file, &header)?;
        file.sync_data()?;
        progress(ExpertSidecarProgress {
            completed_records: header.committed_layers,
            total_records: header.layer_count,
            resumed_records: resumed_layers,
        });
    }

    header.state_complete = true;
    write_header(&mut file, &header)?;
    file.sync_all()?;
    drop(file);
    if output.exists() {
        return Err(ExpertSidecarError::Invalid(format!(
            "refusing to replace SoA-v2 sidecar created concurrently: {}",
            output.display()
        )));
    }
    std::fs::rename(&partial, output)?;
    sync_parent(output)?;
    Ok(report(&header, resumed_layers, false))
}

/// Verify checkpoint identity, header, size and all 48 layer CRC64 values.
pub fn verify_expert_sidecar_v2(
    checkpoint: &FlashNextCheckpoint,
    path: &Path,
    mut progress: impl FnMut(ExpertSidecarProgress),
) -> Result<ExpertSidecarBuildReport, ExpertSidecarError> {
    validate_checkpoint_geometry(checkpoint)?;
    let fingerprint = expert_sidecar_source_fingerprint(checkpoint)?;
    let expected = expected_header(
        fingerprint,
        true,
        u64::from(EXPERT_SIDECAR_V2_LAYERS),
    )?;
    let header = read_header(path)?;
    validate_header(&header, &expected, true)?;
    validate_file_size(path, &header)?;
    let file = File::open(path)?;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    for layer in 0..header.layer_count {
        let actual = checksum_range(
            &file,
            header
                .layers_offset
                .checked_add(
                    layer
                        .checked_mul(header.layer_bytes)
                        .ok_or(ExpertSidecarError::IntegerOverflow)?,
                )
                .ok_or(ExpertSidecarError::IntegerOverflow)?,
            header.layer_bytes,
            &mut buffer,
        )?;
        let expected = read_checksum(&file, &header, layer)?;
        if actual != expected {
            return Err(ExpertSidecarError::Checksum {
                record: layer,
                expected,
                actual,
            });
        }
        progress(ExpertSidecarProgress {
            completed_records: layer + 1,
            total_records: header.layer_count,
            resumed_records: header.layer_count,
        });
    }
    Ok(report(&header, header.layer_count, true))
}

pub fn read_expert_sidecar_v2_header(
    path: &Path,
) -> Result<ExpertSidecarV2Header, ExpertSidecarError> {
    read_header(path)
}

pub fn validate_expert_sidecar_v2_for_checkpoint(
    checkpoint: &FlashNextCheckpoint,
    path: &Path,
) -> Result<ExpertSidecarV2Header, ExpertSidecarError> {
    validate_checkpoint_geometry(checkpoint)?;
    let expected = expected_header(
        expert_sidecar_source_fingerprint(checkpoint)?,
        true,
        u64::from(EXPERT_SIDECAR_V2_LAYERS),
    )?;
    let header = read_header(path)?;
    validate_header(&header, &expected, true)?;
    validate_file_size(path, &header)?;
    Ok(header)
}

fn report(
    header: &ExpertSidecarV2Header,
    resumed_records: u64,
    already_complete: bool,
) -> ExpertSidecarBuildReport {
    ExpertSidecarBuildReport {
        records: header.layer_count,
        bytes: header.total_bytes,
        resumed_records,
        already_complete,
        source_fingerprint: header.source_fingerprint,
    }
}

fn validate_checkpoint_geometry(
    checkpoint: &FlashNextCheckpoint,
) -> Result<(), ExpertSidecarError> {
    if checkpoint.plan.config.layers != u64::from(EXPERT_SIDECAR_V2_LAYERS)
        || checkpoint.plan.config.experts != u64::from(EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER)
    {
        return Err(ExpertSidecarError::Invalid(format!(
            "SoA-v2 sidecar requires {} layers x {} experts, checkpoint has {} x {}",
            EXPERT_SIDECAR_V2_LAYERS,
            EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER,
            checkpoint.plan.config.layers,
            checkpoint.plan.config.experts
        )));
    }
    Ok(())
}

struct ExpertSource<'checkpoint> {
    checkpoint: &'checkpoint FlashNextCheckpoint,
    files: BTreeMap<PathBuf, File>,
    scale_scratch: Vec<u8>,
    tensor_scratch: Vec<u8>,
}

impl<'checkpoint> ExpertSource<'checkpoint> {
    fn new(checkpoint: &'checkpoint FlashNextCheckpoint) -> Self {
        Self {
            checkpoint,
            files: BTreeMap::new(),
            scale_scratch: vec![0_u8; 102_400],
            tensor_scratch: vec![0_u8; 819_200],
        }
    }

    fn pack_layer(
        &mut self,
        output: &File,
        layer_offset: u64,
        layer: u32,
    ) -> Result<u64, ExpertSidecarError> {
        let mut crc = Crc64::new();
        let mut cursor = layer_offset;

        // FlashInfer's fused activation expects [up; gate], while v1 and the
        // source checkpoint enumerate gate first. Emit the physical contract
        // directly rather than relying on a serving-time transpose.
        for expert in 0..EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER {
            self.write_tensor(
                output,
                &mut cursor,
                &mut crc,
                &tensor_name(layer, expert, "up_proj.weight"),
                "U8",
                &[640, 1280],
                819_200,
            )?;
            self.write_tensor(
                output,
                &mut cursor,
                &mut crc,
                &tensor_name(layer, expert, "gate_proj.weight"),
                "U8",
                &[640, 1280],
                819_200,
            )?;
        }
        debug_assert_eq!(cursor - layer_offset, EXPERT_SIDECAR_V2_W2_WEIGHT_OFFSET);

        for expert in 0..EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER {
            self.write_tensor(
                output,
                &mut cursor,
                &mut crc,
                &tensor_name(layer, expert, "down_proj.weight"),
                "U8",
                &[2560, 320],
                819_200,
            )?;
        }
        debug_assert_eq!(cursor - layer_offset, EXPERT_SIDECAR_V2_W13_SCALE_OFFSET);

        for expert in 0..EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER {
            self.write_scale(
                output,
                &mut cursor,
                &mut crc,
                &tensor_name(layer, expert, "up_proj.weight_scale"),
                640,
                160,
            )?;
            self.write_scale(
                output,
                &mut cursor,
                &mut crc,
                &tensor_name(layer, expert, "gate_proj.weight_scale"),
                640,
                160,
            )?;
        }
        debug_assert_eq!(cursor - layer_offset, EXPERT_SIDECAR_V2_W2_SCALE_OFFSET);

        for expert in 0..EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER {
            self.write_scale(
                output,
                &mut cursor,
                &mut crc,
                &tensor_name(layer, expert, "down_proj.weight_scale"),
                2560,
                40,
            )?;
        }
        debug_assert_eq!(
            cursor - layer_offset,
            EXPERT_SIDECAR_V2_W13_INPUT_SCALE_QUANT_OFFSET
        );

        self.write_scalar_plane(output, &mut cursor, &mut crc, layer, "gate_proj", true)?;
        debug_assert_eq!(cursor - layer_offset, EXPERT_SIDECAR_V2_W13_ALPHA_OFFSET);
        self.write_scalar_plane(output, &mut cursor, &mut crc, layer, "gate_proj", false)?;
        debug_assert_eq!(
            cursor - layer_offset,
            EXPERT_SIDECAR_V2_W2_INPUT_SCALE_QUANT_OFFSET
        );
        self.write_scalar_plane(output, &mut cursor, &mut crc, layer, "down_proj", true)?;
        debug_assert_eq!(cursor - layer_offset, EXPERT_SIDECAR_V2_W2_ALPHA_OFFSET);
        self.write_scalar_plane(output, &mut cursor, &mut crc, layer, "down_proj", false)?;
        if cursor - layer_offset != EXPERT_SIDECAR_V2_LAYER_BYTES {
            return Err(ExpertSidecarError::IntegerOverflow);
        }
        Ok(crc.finish())
    }

    fn write_tensor(
        &mut self,
        output: &File,
        cursor: &mut u64,
        crc: &mut Crc64,
        name: &str,
        dtype: &str,
        shape: &[u64],
        bytes: usize,
    ) -> Result<(), ExpertSidecarError> {
        let tensor = self.checkpoint.tensor(name)?.clone();
        validate_location(name, &tensor, dtype, shape, bytes as u64)?;
        self.ensure_file(&tensor.relative_file)?;
        let source = self
            .files
            .get(&tensor.relative_file)
            .expect("expert shard inserted");
        let buffer = &mut self.tensor_scratch[..bytes];
        source.read_exact_at(buffer, tensor.absolute_offset)?;
        output.write_all_at(buffer, *cursor)?;
        crc.update(buffer);
        *cursor = cursor
            .checked_add(bytes as u64)
            .ok_or(ExpertSidecarError::IntegerOverflow)?;
        Ok(())
    }

    fn write_scale(
        &mut self,
        output: &File,
        cursor: &mut u64,
        crc: &mut Crc64,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<(), ExpertSidecarError> {
        let bytes = rows
            .checked_mul(columns)
            .ok_or(ExpertSidecarError::IntegerOverflow)?;
        let tensor = self.checkpoint.tensor(name)?.clone();
        validate_location(
            name,
            &tensor,
            "F8_E4M3",
            &[rows as u64, columns as u64],
            bytes as u64,
        )?;
        self.ensure_file(&tensor.relative_file)?;
        self.files
            .get(&tensor.relative_file)
            .expect("expert shard inserted")
            .read_exact_at(&mut self.scale_scratch[..bytes], tensor.absolute_offset)?;
        let packed = &mut self.tensor_scratch[..bytes];
        interleave_128x4(&self.scale_scratch[..bytes], packed, rows, columns);
        output.write_all_at(packed, *cursor)?;
        crc.update(packed);
        *cursor = cursor
            .checked_add(bytes as u64)
            .ok_or(ExpertSidecarError::IntegerOverflow)?;
        Ok(())
    }

    fn write_scalar_plane(
        &mut self,
        output: &File,
        cursor: &mut u64,
        crc: &mut Crc64,
        layer: u32,
        projection: &str,
        reciprocal_input: bool,
    ) -> Result<(), ExpertSidecarError> {
        for expert in 0..EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER {
            let prefix = format!(
                "model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
            );
            let input = self.read_f32(&format!("{prefix}.input_scale"))?;
            let value = if reciprocal_input {
                1.0 / input
            } else {
                input * self.read_f32(&format!("{prefix}.weight_scale_2"))?
            };
            let bytes = value.to_le_bytes();
            output.write_all_at(&bytes, *cursor)?;
            crc.update(&bytes);
            *cursor = cursor
                .checked_add(4)
                .ok_or(ExpertSidecarError::IntegerOverflow)?;
        }
        Ok(())
    }

    fn read_f32(&mut self, name: &str) -> Result<f32, ExpertSidecarError> {
        let tensor = self.checkpoint.tensor(name)?.clone();
        validate_location(name, &tensor, "F32", &[], 4)?;
        self.ensure_file(&tensor.relative_file)?;
        let mut bytes = [0_u8; 4];
        self.files
            .get(&tensor.relative_file)
            .expect("expert shard inserted")
            .read_exact_at(&mut bytes, tensor.absolute_offset)?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn ensure_file(&mut self, relative: &Path) -> Result<(), ExpertSidecarError> {
        if !self.files.contains_key(relative) {
            self.files.insert(
                relative.to_path_buf(),
                File::open(self.checkpoint.plan.root.join(relative))?,
            );
        }
        Ok(())
    }
}

fn tensor_name(layer: u32, expert: u32, suffix: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{suffix}")
}

fn validate_location(
    name: &str,
    tensor: &SafetensorLocation,
    dtype: &str,
    shape: &[u64],
    bytes: u64,
) -> Result<(), ExpertSidecarError> {
    if tensor.dtype != dtype || tensor.shape != shape || tensor.data_bytes != bytes {
        return Err(ExpertSidecarError::Invalid(format!(
            "tensor {name} expected {dtype} {shape:?} {bytes} bytes, got {} {:?} {} bytes",
            tensor.dtype, tensor.shape, tensor.data_bytes
        )));
    }
    Ok(())
}

fn interleave_128x4(input: &[u8], output: &mut [u8], rows: usize, columns: usize) {
    debug_assert_eq!(input.len(), rows * columns);
    debug_assert_eq!(output.len(), input.len());
    let column_blocks = columns / 4;
    for row_block in 0..rows / 128 {
        for column_block in 0..column_blocks {
            for row_lane in 0..32 {
                for row_four in 0..4 {
                    let row = row_block * 128 + row_four * 32 + row_lane;
                    let source = row * columns + column_block * 4;
                    let destination =
                        (((row_block * column_blocks + column_block) * 32 + row_lane) * 4
                            + row_four)
                            * 4;
                    output[destination..destination + 4]
                        .copy_from_slice(&input[source..source + 4]);
                }
            }
        }
    }
}

fn expected_header(
    source_fingerprint: [u8; 32],
    complete: bool,
    committed_layers: u64,
) -> Result<ExpertSidecarV2Header, ExpertSidecarError> {
    let checksums = u64::from(EXPERT_SIDECAR_V2_LAYERS)
        .checked_mul(CHECKSUM_BYTES)
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    let layers_offset = align_up(
        EXPERT_SIDECAR_V2_HEADER_BYTES
            .checked_add(checksums)
            .ok_or(ExpertSidecarError::IntegerOverflow)?,
        4096,
    )?;
    let total_bytes = layers_offset
        .checked_add(
            u64::from(EXPERT_SIDECAR_V2_LAYERS)
                .checked_mul(EXPERT_SIDECAR_V2_LAYER_BYTES)
                .ok_or(ExpertSidecarError::IntegerOverflow)?,
        )
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    Ok(ExpertSidecarV2Header {
        state_complete: complete,
        layer_bytes: EXPERT_SIDECAR_V2_LAYER_BYTES,
        layer_count: u64::from(EXPERT_SIDECAR_V2_LAYERS),
        checksums_offset: EXPERT_SIDECAR_V2_HEADER_BYTES,
        layers_offset,
        total_bytes,
        committed_layers,
        source_fingerprint,
    })
}

fn serialize_header(
    header: &ExpertSidecarV2Header,
) -> Result<[u8; 4096], ExpertSidecarError> {
    let mut bytes = [0_u8; 4096];
    bytes[..MAGIC.len()].copy_from_slice(&MAGIC);
    put_u32(&mut bytes, VERSION_OFFSET, EXPERT_SIDECAR_V2_VERSION)?;
    put_u32(
        &mut bytes,
        HEADER_BYTES_OFFSET,
        EXPERT_SIDECAR_V2_HEADER_BYTES as u32,
    )?;
    put_u32(
        &mut bytes,
        STATE_OFFSET,
        if header.state_complete {
            STATE_COMPLETE
        } else {
            STATE_BUILDING
        },
    )?;
    put_u32(&mut bytes, LAYERS_OFFSET, EXPERT_SIDECAR_V2_LAYERS)?;
    put_u32(
        &mut bytes,
        EXPERTS_OFFSET,
        EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER,
    )?;
    put_u64(&mut bytes, RECORD_BYTES_OFFSET, header.layer_bytes)?;
    put_u64(&mut bytes, RECORD_COUNT_OFFSET, header.layer_count)?;
    put_u64(&mut bytes, CHECKSUMS_OFFSET_OFFSET, header.checksums_offset)?;
    put_u64(&mut bytes, RECORDS_OFFSET_OFFSET, header.layers_offset)?;
    put_u64(&mut bytes, TOTAL_BYTES_OFFSET, header.total_bytes)?;
    put_u64(
        &mut bytes,
        COMMITTED_RECORDS_OFFSET,
        header.committed_layers,
    )?;
    bytes[SOURCE_FINGERPRINT_OFFSET..SOURCE_FINGERPRINT_OFFSET + 32]
        .copy_from_slice(&header.source_fingerprint);
    bytes[LAYOUT_NAME_OFFSET..LAYOUT_NAME_OFFSET + LAYOUT_NAME.len()].copy_from_slice(LAYOUT_NAME);
    let checksum = crc64_ecma(&bytes);
    put_u64(&mut bytes, HEADER_CHECKSUM_OFFSET, checksum)?;
    Ok(bytes)
}

fn parse_header(bytes: &[u8]) -> Result<ExpertSidecarV2Header, ExpertSidecarError> {
    if bytes.len() != EXPERT_SIDECAR_V2_HEADER_BYTES as usize || bytes[..16] != MAGIC {
        return Err(ExpertSidecarError::Invalid(
            "invalid Qwen SoA-v2 sidecar magic or header size".into(),
        ));
    }
    let stored_checksum = get_u64(bytes, HEADER_CHECKSUM_OFFSET)?;
    let mut checksum_input = [0_u8; 4096];
    checksum_input.copy_from_slice(bytes);
    checksum_input[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 8].fill(0);
    if crc64_ecma(&checksum_input) != stored_checksum {
        return Err(ExpertSidecarError::Invalid(
            "Qwen SoA-v2 sidecar header checksum mismatch".into(),
        ));
    }
    if get_u32(bytes, VERSION_OFFSET)? != EXPERT_SIDECAR_V2_VERSION
        || get_u32(bytes, HEADER_BYTES_OFFSET)? != EXPERT_SIDECAR_V2_HEADER_BYTES as u32
        || get_u32(bytes, LAYERS_OFFSET)? != EXPERT_SIDECAR_V2_LAYERS
        || get_u32(bytes, EXPERTS_OFFSET)? != EXPERT_SIDECAR_V2_EXPERTS_PER_LAYER
        || &bytes[LAYOUT_NAME_OFFSET..LAYOUT_NAME_OFFSET + LAYOUT_NAME.len()] != LAYOUT_NAME
    {
        return Err(ExpertSidecarError::Invalid(
            "unsupported Qwen SoA-v2 sidecar schema".into(),
        ));
    }
    let state = get_u32(bytes, STATE_OFFSET)?;
    if state != STATE_BUILDING && state != STATE_COMPLETE {
        return Err(ExpertSidecarError::Invalid(
            "invalid Qwen SoA-v2 sidecar state".into(),
        ));
    }
    let mut fingerprint = [0_u8; 32];
    fingerprint.copy_from_slice(&bytes[SOURCE_FINGERPRINT_OFFSET..SOURCE_FINGERPRINT_OFFSET + 32]);
    Ok(ExpertSidecarV2Header {
        state_complete: state == STATE_COMPLETE,
        layer_bytes: get_u64(bytes, RECORD_BYTES_OFFSET)?,
        layer_count: get_u64(bytes, RECORD_COUNT_OFFSET)?,
        checksums_offset: get_u64(bytes, CHECKSUMS_OFFSET_OFFSET)?,
        layers_offset: get_u64(bytes, RECORDS_OFFSET_OFFSET)?,
        total_bytes: get_u64(bytes, TOTAL_BYTES_OFFSET)?,
        committed_layers: get_u64(bytes, COMMITTED_RECORDS_OFFSET)?,
        source_fingerprint: fingerprint,
    })
}

fn validate_header(
    actual: &ExpertSidecarV2Header,
    expected: &ExpertSidecarV2Header,
    require_complete: bool,
) -> Result<(), ExpertSidecarError> {
    if actual.layer_bytes != expected.layer_bytes
        || actual.layer_count != expected.layer_count
        || actual.checksums_offset != expected.checksums_offset
        || actual.layers_offset != expected.layers_offset
        || actual.total_bytes != expected.total_bytes
        || actual.source_fingerprint != expected.source_fingerprint
        || actual.committed_layers > actual.layer_count
        || (require_complete
            && (!actual.state_complete || actual.committed_layers != actual.layer_count))
        || (!require_complete && actual.state_complete)
    {
        return Err(ExpertSidecarError::Invalid(
            "Qwen SoA-v2 sidecar metadata does not match this checkpoint or build".into(),
        ));
    }
    Ok(())
}

fn recover_last_committed_layer(
    file: &mut File,
    header: &mut ExpertSidecarV2Header,
) -> Result<(), ExpertSidecarError> {
    if header.committed_layers == 0 {
        return Ok(());
    }
    let layer = header.committed_layers - 1;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let actual = checksum_range(
        file,
        header
            .layers_offset
            .checked_add(
                layer
                    .checked_mul(header.layer_bytes)
                    .ok_or(ExpertSidecarError::IntegerOverflow)?,
            )
            .ok_or(ExpertSidecarError::IntegerOverflow)?,
        header.layer_bytes,
        &mut buffer,
    )?;
    if actual != read_checksum(file, header, layer)? {
        header.committed_layers = layer;
        write_header(file, header)?;
        file.sync_data()?;
    }
    Ok(())
}

fn checksum_range(
    file: &File,
    mut offset: u64,
    mut bytes: u64,
    buffer: &mut [u8],
) -> Result<u64, ExpertSidecarError> {
    let mut crc = Crc64::new();
    while bytes != 0 {
        let take = usize::try_from(bytes.min(buffer.len() as u64))
            .map_err(|_| ExpertSidecarError::IntegerOverflow)?;
        file.read_exact_at(&mut buffer[..take], offset)?;
        crc.update(&buffer[..take]);
        offset = offset
            .checked_add(take as u64)
            .ok_or(ExpertSidecarError::IntegerOverflow)?;
        bytes -= take as u64;
    }
    Ok(crc.finish())
}

fn write_checksum(
    file: &File,
    header: &ExpertSidecarV2Header,
    layer: u64,
    checksum: u64,
) -> Result<(), ExpertSidecarError> {
    let offset = header
        .checksums_offset
        .checked_add(
            layer
                .checked_mul(CHECKSUM_BYTES)
                .ok_or(ExpertSidecarError::IntegerOverflow)?,
        )
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    file.write_all_at(&checksum.to_le_bytes(), offset)?;
    Ok(())
}

fn read_checksum(
    file: &File,
    header: &ExpertSidecarV2Header,
    layer: u64,
) -> Result<u64, ExpertSidecarError> {
    let offset = header
        .checksums_offset
        .checked_add(
            layer
                .checked_mul(CHECKSUM_BYTES)
                .ok_or(ExpertSidecarError::IntegerOverflow)?,
        )
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    let mut bytes = [0_u8; 8];
    file.read_exact_at(&mut bytes, offset)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_header(
    file: &mut File,
    header: &ExpertSidecarV2Header,
) -> Result<(), ExpertSidecarError> {
    file.write_all_at(&serialize_header(header)?, 0)?;
    Ok(())
}

fn read_header(path: &Path) -> Result<ExpertSidecarV2Header, ExpertSidecarError> {
    read_header_from(&File::open(path)?)
}

fn read_header_from(file: &File) -> Result<ExpertSidecarV2Header, ExpertSidecarError> {
    let mut bytes = [0_u8; 4096];
    file.read_exact_at(&mut bytes, 0)?;
    parse_header(&bytes)
}

fn validate_file_size(
    path: &Path,
    header: &ExpertSidecarV2Header,
) -> Result<(), ExpertSidecarError> {
    let actual = path.metadata()?.len();
    if actual != header.total_bytes {
        return Err(ExpertSidecarError::Invalid(format!(
            "SoA-v2 sidecar {} has {actual} bytes, expected {}",
            path.display(),
            header.total_bytes
        )));
    }
    Ok(())
}

fn partial_path(output: &Path) -> Result<PathBuf, ExpertSidecarError> {
    let file_name = output.file_name().ok_or_else(|| {
        ExpertSidecarError::Invalid("SoA-v2 sidecar output needs a file name".into())
    })?;
    let mut partial_name = file_name.to_os_string();
    partial_name.push(".partial");
    Ok(output.with_file_name(partial_name))
}

fn sync_parent(path: &Path) -> Result<(), ExpertSidecarError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ExpertSidecarError> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or(ExpertSidecarError::IntegerOverflow)
}

fn put_u32(output: &mut [u8], offset: usize, value: u32) -> Result<(), ExpertSidecarError> {
    output
        .get_mut(offset..offset + 4)
        .ok_or(ExpertSidecarError::IntegerOverflow)?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u64(output: &mut [u8], offset: usize, value: u64) -> Result<(), ExpertSidecarError> {
    output
        .get_mut(offset..offset + 8)
        .ok_or(ExpertSidecarError::IntegerOverflow)?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn get_u32(input: &[u8], offset: usize) -> Result<u32, ExpertSidecarError> {
    Ok(u32::from_le_bytes(
        input
            .get(offset..offset + 4)
            .ok_or(ExpertSidecarError::IntegerOverflow)?
            .try_into()
            .expect("four-byte slice"),
    ))
}

fn get_u64(input: &[u8], offset: usize) -> Result<u64, ExpertSidecarError> {
    Ok(u64::from_le_bytes(
        input
            .get(offset..offset + 8)
            .ok_or(ExpertSidecarError::IntegerOverflow)?
            .try_into()
            .expect("eight-byte slice"),
    ))
}

#[derive(Clone, Copy)]
struct Crc64(u64);

impl Crc64 {
    fn new() -> Self {
        Self(0)
    }

    fn update(&mut self, bytes: &[u8]) {
        let mut crc = self.0;
        for byte in bytes {
            let index = ((crc >> 56) as u8 ^ *byte) as usize;
            crc = CRC64_TABLE[index] ^ (crc << 8);
        }
        self.0 = crc;
    }

    fn finish(self) -> u64 {
        self.0
    }
}

fn crc64_ecma(bytes: &[u8]) -> u64 {
    let mut crc = Crc64::new();
    crc.update(bytes);
    crc.finish()
}

const fn crc64_table() -> [u64; 256] {
    let mut table = [0_u64; 256];
    let mut index = 0_usize;
    while index < table.len() {
        let mut value = (index as u64) << 56;
        let mut bit = 0;
        while bit < 8 {
            value = if value & (1_u64 << 63) != 0 {
                (value << 1) ^ 0x42f0_e1eb_a9ea_3693
            } else {
                value << 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

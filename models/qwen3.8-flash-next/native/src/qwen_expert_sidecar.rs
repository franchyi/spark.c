//! Deterministic, resumable packer for the Qwen3.8 Flash-Next expert sidecar.
//!
//! The checkpoint stores every expert as ten safetensors entries. Serving wants
//! four kernel-ready byte planes plus four scalar planes. This module performs
//! that immutable conversion once, without CUDA or a Python dependency. The
//! final file is atomically published only after all records and checksums are
//! durable.
//!
//! File layout: `[4096-byte header][24576 CRC64s][padding][24576 records]`.
//! Each 2,764,816-byte record is 16-byte aligned and contains, in order,
//! `w13 weight`, `w2 weight`, interleaved `w13 scale`, interleaved `w2 scale`,
//! then the four little-endian F32 values consumed by grouped GEMM. Records are
//! layer-major, then expert-major. The records region starts on a 4-KiB boundary
//! so serving can mmap and CUDA-register it without repacking the file.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::checkpoint::{CheckpointError, FlashNextCheckpoint, SafetensorLocation};
use crate::ffi::{
    QWEN_W2_SCALE_BYTES, QWEN_W2_WEIGHT_BYTES, QWEN_W13_SCALE_BYTES, QWEN_W13_WEIGHT_BYTES,
};

pub const EXPERT_SIDECAR_VERSION: u32 = 1;
pub const EXPERT_SIDECAR_HEADER_BYTES: u64 = 4096;
pub const EXPERT_SIDECAR_LAYERS: u32 = 48;
pub const EXPERT_SIDECAR_EXPERTS_PER_LAYER: u32 = 512;
pub const EXPERT_SIDECAR_RECORDS: u64 =
    EXPERT_SIDECAR_LAYERS as u64 * EXPERT_SIDECAR_EXPERTS_PER_LAYER as u64;

pub const EXPERT_W13_WEIGHT_OFFSET: u64 = 0;
pub const EXPERT_W13_WEIGHT_BYTES: u64 = QWEN_W13_WEIGHT_BYTES;
pub const EXPERT_W2_WEIGHT_OFFSET: u64 = EXPERT_W13_WEIGHT_OFFSET + EXPERT_W13_WEIGHT_BYTES;
pub const EXPERT_W2_WEIGHT_BYTES: u64 = QWEN_W2_WEIGHT_BYTES;
pub const EXPERT_W13_SCALE_OFFSET: u64 = EXPERT_W2_WEIGHT_OFFSET + EXPERT_W2_WEIGHT_BYTES;
pub const EXPERT_W13_SCALE_BYTES: u64 = QWEN_W13_SCALE_BYTES;
pub const EXPERT_W2_SCALE_OFFSET: u64 = EXPERT_W13_SCALE_OFFSET + EXPERT_W13_SCALE_BYTES;
pub const EXPERT_W2_SCALE_BYTES: u64 = QWEN_W2_SCALE_BYTES;
pub const EXPERT_W13_INPUT_GLOBAL_SCALE_OFFSET: u64 =
    EXPERT_W2_SCALE_OFFSET + EXPERT_W2_SCALE_BYTES;
pub const EXPERT_W13_ALPHA_OFFSET: u64 = EXPERT_W13_INPUT_GLOBAL_SCALE_OFFSET + 4;
pub const EXPERT_W2_INPUT_GLOBAL_SCALE_OFFSET: u64 = EXPERT_W13_ALPHA_OFFSET + 4;
pub const EXPERT_W2_ALPHA_OFFSET: u64 = EXPERT_W2_INPUT_GLOBAL_SCALE_OFFSET + 4;
pub const EXPERT_SIDECAR_RECORD_BYTES: u64 = EXPERT_W2_ALPHA_OFFSET + 4;
pub const EXPERT_SIDECAR_RECORD_ALIGNMENT: u64 = 16;
const _: () = assert!(EXPERT_SIDECAR_RECORD_BYTES == 2_764_816);
const _: () = assert!(EXPERT_SIDECAR_RECORD_BYTES % EXPERT_SIDECAR_RECORD_ALIGNMENT == 0);

const MAGIC: [u8; 16] = *b"SPARKQWEXPERT\0\0\0";
const STATE_BUILDING: u32 = 0;
const STATE_COMPLETE: u32 = 1;
const CHECKSUM_BYTES: u64 = 8;
const DEFAULT_SYNC_RECORDS: u64 = 64;
const LAYOUT_NAME: &[u8] = b"qwen3.8-flash-next-nvfp4-aos-v1";

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
pub struct ExpertSidecarHeader {
    pub state_complete: bool,
    pub record_bytes: u64,
    pub record_count: u64,
    pub checksums_offset: u64,
    pub records_offset: u64,
    pub total_bytes: u64,
    pub committed_records: u64,
    pub source_fingerprint: [u8; 32],
}

impl ExpertSidecarHeader {
    /// Byte offset of one layer-major expert record in the sidecar file. The
    /// records region itself is 4-KiB aligned for a direct coherent mmap.
    pub fn record_file_offset(&self, layer: u32, expert: u32) -> Result<u64, ExpertSidecarError> {
        if layer >= EXPERT_SIDECAR_LAYERS || expert >= EXPERT_SIDECAR_EXPERTS_PER_LAYER {
            return Err(ExpertSidecarError::Invalid(format!(
                "Qwen sidecar expert {layer}:{expert} is out of range"
            )));
        }
        let index = u64::from(layer)
            .checked_mul(u64::from(EXPERT_SIDECAR_EXPERTS_PER_LAYER))
            .and_then(|base| base.checked_add(u64::from(expert)))
            .ok_or(ExpertSidecarError::IntegerOverflow)?;
        self.records_offset
            .checked_add(
                index
                    .checked_mul(self.record_bytes)
                    .ok_or(ExpertSidecarError::IntegerOverflow)?,
            )
            .ok_or(ExpertSidecarError::IntegerOverflow)
    }

    pub fn records_bytes(&self) -> Result<u64, ExpertSidecarError> {
        self.record_count
            .checked_mul(self.record_bytes)
            .ok_or(ExpertSidecarError::IntegerOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertSidecarProgress {
    pub completed_records: u64,
    pub total_records: u64,
    pub resumed_records: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertSidecarBuildReport {
    pub records: u64,
    pub bytes: u64,
    pub resumed_records: u64,
    pub already_complete: bool,
    pub source_fingerprint: [u8; 32],
}

pub fn build_expert_sidecar(
    checkpoint: &FlashNextCheckpoint,
    output: &Path,
    mut progress: impl FnMut(ExpertSidecarProgress),
) -> Result<ExpertSidecarBuildReport, ExpertSidecarError> {
    validate_checkpoint_geometry(checkpoint)?;
    let fingerprint = source_fingerprint(checkpoint)?;
    let expected = expected_header(fingerprint, false, 0)?;

    if output.exists() {
        let header = read_header(output)?;
        validate_header(&header, &expected, true)?;
        let actual_bytes = output.metadata()?.len();
        if actual_bytes != header.total_bytes {
            return Err(ExpertSidecarError::Invalid(format!(
                "sidecar {} has {actual_bytes} bytes, expected {}",
                output.display(),
                header.total_bytes
            )));
        }
        return Ok(ExpertSidecarBuildReport {
            records: header.record_count,
            bytes: header.total_bytes,
            resumed_records: header.record_count,
            already_complete: true,
            source_fingerprint: fingerprint,
        });
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
            "partial sidecar {} has {} bytes, expected {}",
            partial.display(),
            file.metadata()?.len(),
            expected.total_bytes
        )));
    }
    let mut header = read_header_from(&file)?;
    validate_header(&header, &expected, false)?;
    recover_last_committed_record(&mut file, &mut header)?;
    let resumed_records = header.committed_records;
    progress(ExpertSidecarProgress {
        completed_records: resumed_records,
        total_records: header.record_count,
        resumed_records,
    });

    let mut source = ExpertSource::new(checkpoint);
    let mut record = vec![0_u8; usize_from_u64(header.record_bytes)?];
    let mut committed = header.committed_records;
    while committed < header.record_count {
        let batch_end = header
            .record_count
            .min(committed.saturating_add(DEFAULT_SYNC_RECORDS));
        for record_index in committed..batch_end {
            let layer = u32::try_from(record_index / u64::from(EXPERT_SIDECAR_EXPERTS_PER_LAYER))
                .map_err(|_| ExpertSidecarError::IntegerOverflow)?;
            let expert = u32::try_from(record_index % u64::from(EXPERT_SIDECAR_EXPERTS_PER_LAYER))
                .map_err(|_| ExpertSidecarError::IntegerOverflow)?;
            source.pack_record(layer, expert, &mut record)?;
            let record_offset = header
                .records_offset
                .checked_add(
                    record_index
                        .checked_mul(header.record_bytes)
                        .ok_or(ExpertSidecarError::IntegerOverflow)?,
                )
                .ok_or(ExpertSidecarError::IntegerOverflow)?;
            file.write_all_at(&record, record_offset)?;
            let checksum = crc64_ecma(&record).to_le_bytes();
            let checksum_offset = header
                .checksums_offset
                .checked_add(
                    record_index
                        .checked_mul(CHECKSUM_BYTES)
                        .ok_or(ExpertSidecarError::IntegerOverflow)?,
                )
                .ok_or(ExpertSidecarError::IntegerOverflow)?;
            file.write_all_at(&checksum, checksum_offset)?;
        }

        // Publish only a prefix whose payload and checksums are already
        // durable. A crash before the second sync merely repeats a safe batch.
        file.sync_data()?;
        header.committed_records = batch_end;
        write_header(&mut file, &header)?;
        file.sync_data()?;
        committed = batch_end;
        progress(ExpertSidecarProgress {
            completed_records: committed,
            total_records: header.record_count,
            resumed_records,
        });
    }

    header.state_complete = true;
    write_header(&mut file, &header)?;
    file.sync_all()?;
    drop(file);
    if output.exists() {
        return Err(ExpertSidecarError::Invalid(format!(
            "refusing to replace sidecar created concurrently: {}",
            output.display()
        )));
    }
    std::fs::rename(&partial, output)?;
    sync_parent(output)?;
    Ok(ExpertSidecarBuildReport {
        records: header.record_count,
        bytes: header.total_bytes,
        resumed_records,
        already_complete: false,
        source_fingerprint: fingerprint,
    })
}

/// Verify the source identity, header, file size, and every record checksum.
/// This intentionally scans the complete 63.3-GiB payload and is not called on
/// the normal serving or already-built fast path.
pub fn verify_expert_sidecar(
    checkpoint: &FlashNextCheckpoint,
    path: &Path,
    mut progress: impl FnMut(ExpertSidecarProgress),
) -> Result<ExpertSidecarBuildReport, ExpertSidecarError> {
    validate_checkpoint_geometry(checkpoint)?;
    let fingerprint = source_fingerprint(checkpoint)?;
    let expected = expected_header(fingerprint, true, EXPERT_SIDECAR_RECORDS)?;
    let header = read_header(path)?;
    validate_header(&header, &expected, true)?;
    let file = File::open(path)?;
    if file.metadata()?.len() != header.total_bytes {
        return Err(ExpertSidecarError::Invalid(format!(
            "sidecar has {} bytes, expected {}",
            file.metadata()?.len(),
            header.total_bytes
        )));
    }
    let mut record = vec![0_u8; usize_from_u64(header.record_bytes)?];
    for index in 0..header.record_count {
        read_record(&file, &header, index, &mut record)?;
        let expected_checksum = read_checksum(&file, &header, index)?;
        let actual_checksum = crc64_ecma(&record);
        if actual_checksum != expected_checksum {
            return Err(ExpertSidecarError::Checksum {
                record: index,
                expected: expected_checksum,
                actual: actual_checksum,
            });
        }
        if (index + 1) % DEFAULT_SYNC_RECORDS == 0 || index + 1 == header.record_count {
            progress(ExpertSidecarProgress {
                completed_records: index + 1,
                total_records: header.record_count,
                resumed_records: header.record_count,
            });
        }
    }
    Ok(ExpertSidecarBuildReport {
        records: header.record_count,
        bytes: header.total_bytes,
        resumed_records: header.record_count,
        already_complete: true,
        source_fingerprint: fingerprint,
    })
}

pub fn read_expert_sidecar_header(path: &Path) -> Result<ExpertSidecarHeader, ExpertSidecarError> {
    read_header(path)
}

/// Validate the immutable metadata needed by serving without scanning the
/// 63.3-GiB payload. Explicit offline verification owns the per-record scan.
pub fn validate_expert_sidecar_for_checkpoint(
    checkpoint: &FlashNextCheckpoint,
    path: &Path,
) -> Result<ExpertSidecarHeader, ExpertSidecarError> {
    validate_checkpoint_geometry(checkpoint)?;
    let fingerprint = source_fingerprint(checkpoint)?;
    let expected = expected_header(fingerprint, true, EXPERT_SIDECAR_RECORDS)?;
    let header = read_header(path)?;
    validate_header(&header, &expected, true)?;
    let actual_bytes = path.metadata()?.len();
    if actual_bytes != header.total_bytes {
        return Err(ExpertSidecarError::Invalid(format!(
            "sidecar {} has {actual_bytes} bytes, expected {}",
            path.display(),
            header.total_bytes
        )));
    }
    Ok(header)
}

fn validate_checkpoint_geometry(
    checkpoint: &FlashNextCheckpoint,
) -> Result<(), ExpertSidecarError> {
    if checkpoint.plan.config.layers != u64::from(EXPERT_SIDECAR_LAYERS)
        || checkpoint.plan.config.experts != u64::from(EXPERT_SIDECAR_EXPERTS_PER_LAYER)
    {
        return Err(ExpertSidecarError::Invalid(format!(
            "sidecar requires {} layers x {} experts, checkpoint has {} x {}",
            EXPERT_SIDECAR_LAYERS,
            EXPERT_SIDECAR_EXPERTS_PER_LAYER,
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
}

impl<'checkpoint> ExpertSource<'checkpoint> {
    fn new(checkpoint: &'checkpoint FlashNextCheckpoint) -> Self {
        Self {
            checkpoint,
            files: BTreeMap::new(),
            scale_scratch: vec![0_u8; 102_400],
        }
    }

    fn pack_record(
        &mut self,
        layer: u32,
        expert: u32,
        output: &mut [u8],
    ) -> Result<(), ExpertSidecarError> {
        if output.len() != usize_from_u64(EXPERT_SIDECAR_RECORD_BYTES)? {
            return Err(ExpertSidecarError::IntegerOverflow);
        }
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
        self.read_tensor(
            &format!("{prefix}.gate_proj.weight"),
            "U8",
            &[640, 1280],
            &mut output[0..819_200],
        )?;
        self.read_tensor(
            &format!("{prefix}.up_proj.weight"),
            "U8",
            &[640, 1280],
            &mut output[819_200..usize_from_u64(EXPERT_W13_WEIGHT_BYTES)?],
        )?;
        self.read_tensor(
            &format!("{prefix}.down_proj.weight"),
            "U8",
            &[2560, 320],
            slice_mut(output, EXPERT_W2_WEIGHT_OFFSET, EXPERT_W2_WEIGHT_BYTES)?,
        )?;
        self.read_interleaved_scale(
            &format!("{prefix}.gate_proj.weight_scale"),
            640,
            160,
            slice_mut(output, EXPERT_W13_SCALE_OFFSET, 102_400)?,
        )?;
        self.read_interleaved_scale(
            &format!("{prefix}.up_proj.weight_scale"),
            640,
            160,
            slice_mut(output, EXPERT_W13_SCALE_OFFSET + 102_400, 102_400)?,
        )?;
        self.read_interleaved_scale(
            &format!("{prefix}.down_proj.weight_scale"),
            2560,
            40,
            slice_mut(output, EXPERT_W2_SCALE_OFFSET, EXPERT_W2_SCALE_BYTES)?,
        )?;
        let gate_input = self.read_f32(&format!("{prefix}.gate_proj.input_scale"))?;
        let gate_weight = self.read_f32(&format!("{prefix}.gate_proj.weight_scale_2"))?;
        let down_input = self.read_f32(&format!("{prefix}.down_proj.input_scale"))?;
        let down_weight = self.read_f32(&format!("{prefix}.down_proj.weight_scale_2"))?;
        write_f32(
            output,
            EXPERT_W13_INPUT_GLOBAL_SCALE_OFFSET,
            1.0 / gate_input,
        )?;
        write_f32(output, EXPERT_W13_ALPHA_OFFSET, gate_input * gate_weight)?;
        write_f32(
            output,
            EXPERT_W2_INPUT_GLOBAL_SCALE_OFFSET,
            1.0 / down_input,
        )?;
        write_f32(output, EXPERT_W2_ALPHA_OFFSET, down_input * down_weight)?;
        Ok(())
    }

    fn read_tensor(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[u64],
        output: &mut [u8],
    ) -> Result<(), ExpertSidecarError> {
        let tensor = self.checkpoint.tensor(name)?.clone();
        validate_location(name, &tensor, dtype, shape, output.len() as u64)?;
        let file = self.file(&tensor.relative_file)?;
        file.read_exact_at(output, tensor.absolute_offset)?;
        Ok(())
    }

    fn read_interleaved_scale(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        output: &mut [u8],
    ) -> Result<(), ExpertSidecarError> {
        let bytes = rows
            .checked_mul(columns)
            .ok_or(ExpertSidecarError::IntegerOverflow)?;
        if output.len() != bytes || self.scale_scratch.len() < bytes {
            return Err(ExpertSidecarError::IntegerOverflow);
        }
        let tensor = self.checkpoint.tensor(name)?.clone();
        validate_location(
            name,
            &tensor,
            "F8_E4M3",
            &[rows as u64, columns as u64],
            bytes as u64,
        )?;
        let relative_file = tensor.relative_file.clone();
        let offset = tensor.absolute_offset;
        self.ensure_file(&relative_file)?;
        self.files
            .get(&relative_file)
            .expect("expert file inserted")
            .read_exact_at(&mut self.scale_scratch[..bytes], offset)?;
        interleave_128x4(&self.scale_scratch[..bytes], output, rows, columns);
        Ok(())
    }

    fn read_f32(&mut self, name: &str) -> Result<f32, ExpertSidecarError> {
        let mut bytes = [0_u8; 4];
        self.read_tensor(name, "F32", &[], &mut bytes)?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn file(&mut self, relative: &Path) -> Result<&File, ExpertSidecarError> {
        self.ensure_file(relative)?;
        Ok(self.files.get(relative).expect("expert file inserted"))
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

fn expected_header(
    source_fingerprint: [u8; 32],
    complete: bool,
    committed_records: u64,
) -> Result<ExpertSidecarHeader, ExpertSidecarError> {
    let checksum_bytes = EXPERT_SIDECAR_RECORDS
        .checked_mul(CHECKSUM_BYTES)
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    let records_offset = align_up(
        EXPERT_SIDECAR_HEADER_BYTES
            .checked_add(checksum_bytes)
            .ok_or(ExpertSidecarError::IntegerOverflow)?,
        4096,
    )?;
    let total_bytes = records_offset
        .checked_add(
            EXPERT_SIDECAR_RECORDS
                .checked_mul(EXPERT_SIDECAR_RECORD_BYTES)
                .ok_or(ExpertSidecarError::IntegerOverflow)?,
        )
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    Ok(ExpertSidecarHeader {
        state_complete: complete,
        record_bytes: EXPERT_SIDECAR_RECORD_BYTES,
        record_count: EXPERT_SIDECAR_RECORDS,
        checksums_offset: EXPERT_SIDECAR_HEADER_BYTES,
        records_offset,
        total_bytes,
        committed_records,
        source_fingerprint,
    })
}

fn serialize_header(header: &ExpertSidecarHeader) -> Result<[u8; 4096], ExpertSidecarError> {
    let mut bytes = [0_u8; 4096];
    bytes[..MAGIC.len()].copy_from_slice(&MAGIC);
    put_u32(&mut bytes, VERSION_OFFSET, EXPERT_SIDECAR_VERSION)?;
    put_u32(
        &mut bytes,
        HEADER_BYTES_OFFSET,
        u32::try_from(EXPERT_SIDECAR_HEADER_BYTES)
            .map_err(|_| ExpertSidecarError::IntegerOverflow)?,
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
    put_u32(&mut bytes, LAYERS_OFFSET, EXPERT_SIDECAR_LAYERS)?;
    put_u32(&mut bytes, EXPERTS_OFFSET, EXPERT_SIDECAR_EXPERTS_PER_LAYER)?;
    put_u64(&mut bytes, RECORD_BYTES_OFFSET, header.record_bytes)?;
    put_u64(&mut bytes, RECORD_COUNT_OFFSET, header.record_count)?;
    put_u64(&mut bytes, CHECKSUMS_OFFSET_OFFSET, header.checksums_offset)?;
    put_u64(&mut bytes, RECORDS_OFFSET_OFFSET, header.records_offset)?;
    put_u64(&mut bytes, TOTAL_BYTES_OFFSET, header.total_bytes)?;
    put_u64(
        &mut bytes,
        COMMITTED_RECORDS_OFFSET,
        header.committed_records,
    )?;
    bytes[SOURCE_FINGERPRINT_OFFSET..SOURCE_FINGERPRINT_OFFSET + 32]
        .copy_from_slice(&header.source_fingerprint);
    bytes[LAYOUT_NAME_OFFSET..LAYOUT_NAME_OFFSET + LAYOUT_NAME.len()].copy_from_slice(LAYOUT_NAME);
    let checksum = crc64_ecma(&bytes);
    put_u64(&mut bytes, HEADER_CHECKSUM_OFFSET, checksum)?;
    Ok(bytes)
}

fn parse_header(bytes: &[u8]) -> Result<ExpertSidecarHeader, ExpertSidecarError> {
    if bytes.len() != EXPERT_SIDECAR_HEADER_BYTES as usize || bytes[..16] != MAGIC {
        return Err(ExpertSidecarError::Invalid(
            "invalid Qwen expert sidecar magic or header size".into(),
        ));
    }
    let stored_checksum = get_u64(bytes, HEADER_CHECKSUM_OFFSET)?;
    let mut checksum_input = [0_u8; 4096];
    checksum_input.copy_from_slice(bytes);
    checksum_input[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 8].fill(0);
    if crc64_ecma(&checksum_input) != stored_checksum {
        return Err(ExpertSidecarError::Invalid(
            "Qwen expert sidecar header checksum mismatch".into(),
        ));
    }
    if get_u32(bytes, VERSION_OFFSET)? != EXPERT_SIDECAR_VERSION
        || get_u32(bytes, HEADER_BYTES_OFFSET)? != EXPERT_SIDECAR_HEADER_BYTES as u32
        || get_u32(bytes, LAYERS_OFFSET)? != EXPERT_SIDECAR_LAYERS
        || get_u32(bytes, EXPERTS_OFFSET)? != EXPERT_SIDECAR_EXPERTS_PER_LAYER
        || &bytes[LAYOUT_NAME_OFFSET..LAYOUT_NAME_OFFSET + LAYOUT_NAME.len()] != LAYOUT_NAME
    {
        return Err(ExpertSidecarError::Invalid(
            "unsupported Qwen expert sidecar schema".into(),
        ));
    }
    let state = get_u32(bytes, STATE_OFFSET)?;
    if state != STATE_BUILDING && state != STATE_COMPLETE {
        return Err(ExpertSidecarError::Invalid(
            "invalid Qwen expert sidecar state".into(),
        ));
    }
    let mut source_fingerprint = [0_u8; 32];
    source_fingerprint
        .copy_from_slice(&bytes[SOURCE_FINGERPRINT_OFFSET..SOURCE_FINGERPRINT_OFFSET + 32]);
    Ok(ExpertSidecarHeader {
        state_complete: state == STATE_COMPLETE,
        record_bytes: get_u64(bytes, RECORD_BYTES_OFFSET)?,
        record_count: get_u64(bytes, RECORD_COUNT_OFFSET)?,
        checksums_offset: get_u64(bytes, CHECKSUMS_OFFSET_OFFSET)?,
        records_offset: get_u64(bytes, RECORDS_OFFSET_OFFSET)?,
        total_bytes: get_u64(bytes, TOTAL_BYTES_OFFSET)?,
        committed_records: get_u64(bytes, COMMITTED_RECORDS_OFFSET)?,
        source_fingerprint,
    })
}

fn validate_header(
    actual: &ExpertSidecarHeader,
    expected: &ExpertSidecarHeader,
    require_complete: bool,
) -> Result<(), ExpertSidecarError> {
    if actual.record_bytes != expected.record_bytes
        || actual.record_count != expected.record_count
        || actual.checksums_offset != expected.checksums_offset
        || actual.records_offset != expected.records_offset
        || actual.total_bytes != expected.total_bytes
        || actual.source_fingerprint != expected.source_fingerprint
        || actual.committed_records > actual.record_count
        || (require_complete
            && (!actual.state_complete || actual.committed_records != actual.record_count))
        || (!require_complete && actual.state_complete)
    {
        return Err(ExpertSidecarError::Invalid(
            "Qwen expert sidecar metadata does not match this checkpoint or build".into(),
        ));
    }
    Ok(())
}

fn write_header(file: &mut File, header: &ExpertSidecarHeader) -> Result<(), ExpertSidecarError> {
    file.write_all_at(&serialize_header(header)?, 0)?;
    Ok(())
}

fn read_header(path: &Path) -> Result<ExpertSidecarHeader, ExpertSidecarError> {
    read_header_from(&File::open(path)?)
}

fn read_header_from(file: &File) -> Result<ExpertSidecarHeader, ExpertSidecarError> {
    let mut bytes = [0_u8; 4096];
    file.read_exact_at(&mut bytes, 0)?;
    parse_header(&bytes)
}

fn recover_last_committed_record(
    file: &mut File,
    header: &mut ExpertSidecarHeader,
) -> Result<(), ExpertSidecarError> {
    if header.committed_records == 0 {
        return Ok(());
    }
    let index = header.committed_records - 1;
    let mut record = vec![0_u8; usize_from_u64(header.record_bytes)?];
    read_record(file, header, index, &mut record)?;
    if crc64_ecma(&record) != read_checksum(file, header, index)? {
        header.committed_records = index;
        write_header(file, header)?;
        file.sync_data()?;
    }
    Ok(())
}

fn read_record(
    file: &File,
    header: &ExpertSidecarHeader,
    index: u64,
    output: &mut [u8],
) -> Result<(), ExpertSidecarError> {
    let offset = header
        .records_offset
        .checked_add(
            index
                .checked_mul(header.record_bytes)
                .ok_or(ExpertSidecarError::IntegerOverflow)?,
        )
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    file.read_exact_at(output, offset)?;
    Ok(())
}

fn read_checksum(
    file: &File,
    header: &ExpertSidecarHeader,
    index: u64,
) -> Result<u64, ExpertSidecarError> {
    let offset = header
        .checksums_offset
        .checked_add(
            index
                .checked_mul(CHECKSUM_BYTES)
                .ok_or(ExpertSidecarError::IntegerOverflow)?,
        )
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    let mut bytes = [0_u8; 8];
    file.read_exact_at(&mut bytes, offset)?;
    Ok(u64::from_le_bytes(bytes))
}

fn source_fingerprint(checkpoint: &FlashNextCheckpoint) -> Result<[u8; 32], ExpertSidecarError> {
    let mut lanes = [
        Crc64::new(0),
        Crc64::new(0x9e37_79b9_7f4a_7c15),
        Crc64::new(0xd1b5_4a32_d192_ed03),
        Crc64::new(0x94d0_49bb_1331_11eb),
    ];
    fingerprint_update(&mut lanes, b"spark.c/qwen-expert-sidecar/source-v1\0");
    for name in ["config.json", "model.safetensors.index.json"] {
        fingerprint_update(&mut lanes, name.as_bytes());
        let mut file = File::open(checkpoint.plan.root.join(name))?;
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let bytes = file.read(&mut buffer)?;
            if bytes == 0 {
                break;
            }
            fingerprint_update(&mut lanes, &buffer[..bytes]);
        }
    }
    let mut shard_sizes = BTreeMap::<PathBuf, u64>::new();
    for (name, tensor) in checkpoint
        .tensors
        .iter()
        .filter(|(name, _)| name.contains(".mlp.experts."))
    {
        fingerprint_update(&mut lanes, name.as_bytes());
        fingerprint_update(&mut lanes, tensor.relative_file.as_os_str().as_bytes());
        fingerprint_update(&mut lanes, &tensor.absolute_offset.to_le_bytes());
        fingerprint_update(&mut lanes, &tensor.data_bytes.to_le_bytes());
        fingerprint_update(&mut lanes, tensor.dtype.as_bytes());
        for dimension in &tensor.shape {
            fingerprint_update(&mut lanes, &dimension.to_le_bytes());
        }
        if !shard_sizes.contains_key(&tensor.relative_file) {
            shard_sizes.insert(
                tensor.relative_file.clone(),
                checkpoint
                    .plan
                    .root
                    .join(&tensor.relative_file)
                    .metadata()?
                    .len(),
            );
        }
    }
    for (file, bytes) in shard_sizes {
        fingerprint_update(&mut lanes, file.as_os_str().as_bytes());
        fingerprint_update(&mut lanes, &bytes.to_le_bytes());
    }
    let mut result = [0_u8; 32];
    for (index, lane) in lanes.into_iter().enumerate() {
        result[index * 8..index * 8 + 8].copy_from_slice(&lane.finish().to_le_bytes());
    }
    Ok(result)
}

fn fingerprint_update(lanes: &mut [Crc64; 4], bytes: &[u8]) {
    for (lane, domain) in lanes.iter_mut().zip(0_u64..) {
        lane.update(&domain.to_le_bytes());
        lane.update(&(bytes.len() as u64).to_le_bytes());
        lane.update(bytes);
    }
}

#[derive(Clone, Copy)]
struct Crc64(u64);

impl Crc64 {
    fn new(seed: u64) -> Self {
        Self(seed)
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
    let mut checksum = Crc64::new(0);
    checksum.update(bytes);
    checksum.finish()
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

fn write_f32(output: &mut [u8], offset: u64, value: f32) -> Result<(), ExpertSidecarError> {
    slice_mut(output, offset, 4)?.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn slice_mut(output: &mut [u8], offset: u64, bytes: u64) -> Result<&mut [u8], ExpertSidecarError> {
    let begin = usize_from_u64(offset)?;
    let end = begin
        .checked_add(usize_from_u64(bytes)?)
        .ok_or(ExpertSidecarError::IntegerOverflow)?;
    output
        .get_mut(begin..end)
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

fn align_up(value: u64, alignment: u64) -> Result<u64, ExpertSidecarError> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or(ExpertSidecarError::IntegerOverflow)
}

fn usize_from_u64(value: u64) -> Result<usize, ExpertSidecarError> {
    usize::try_from(value).map_err(|_| ExpertSidecarError::IntegerOverflow)
}

fn partial_path(output: &Path) -> Result<PathBuf, ExpertSidecarError> {
    let file_name = output
        .file_name()
        .ok_or_else(|| ExpertSidecarError::Invalid("sidecar output needs a file name".into()))?;
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

#[derive(Debug)]
pub enum ExpertSidecarError {
    Checkpoint(CheckpointError),
    Io(std::io::Error),
    IntegerOverflow,
    Invalid(String),
    Checksum {
        record: u64,
        expected: u64,
        actual: u64,
    },
}

impl From<CheckpointError> for ExpertSidecarError {
    fn from(error: CheckpointError) -> Self {
        Self::Checkpoint(error)
    }
}

impl From<std::io::Error> for ExpertSidecarError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl Display for ExpertSidecarError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::IntegerOverflow => formatter.write_str("Qwen expert sidecar integer overflow"),
            Self::Invalid(message) => formatter.write_str(message),
            Self::Checksum {
                record,
                expected,
                actual,
            } => write!(
                formatter,
                "Qwen expert sidecar record {record} checksum {actual:016x}, expected {expected:016x}"
            ),
        }
    }
}

impl std::error::Error for ExpertSidecarError {}

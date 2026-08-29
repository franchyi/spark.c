//! Fixed Qwen NVFP4 expert arenas and mmap-to-cache promotion.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use crate::checkpoint::{CheckpointError, FlashNextCheckpoint};
use crate::coherent::{CoherentRegionError, CoherentRegionOwner};
use crate::cuda::CudaStreamOwner;
use crate::fabric::ExpertLoad;
use crate::ffi::{
    QWEN_EXPERT_CAPACITY, QWEN_EXPERT_PACK_ABI_VERSION, QWEN_W13_SCALE_BYTES,
    QWEN_W13_WEIGHT_BYTES, QWEN_W2_SCALE_BYTES, QWEN_W2_WEIGHT_BYTES,
    QwenExpertPackArgs, Status, flash_qwen_expert_pack_launch,
};
use crate::qwen_weights::{FlashNextWeightMaps, QwenTensorView, QwenWeightError};

const ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenExpertHotViews {
    pub capacity: u32,
    pub w13_weights: u64,
    pub w2_weights: u64,
    pub w13_scales: u64,
    pub w2_scales: u64,
    pub w13_input_global_scales: u64,
    pub w13_alpha: u64,
    pub w2_input_global_scales: u64,
    pub w2_alpha: u64,
}

pub struct QwenExpertHotCache {
    w13_weights: CoherentRegionOwner,
    w2_weights: CoherentRegionOwner,
    w13_scales: CoherentRegionOwner,
    w2_scales: CoherentRegionOwner,
    w13_input_global_scales: CoherentRegionOwner,
    w13_alpha: CoherentRegionOwner,
    w2_input_global_scales: CoherentRegionOwner,
    w2_alpha: CoherentRegionOwner,
}

pub struct QwenExpertHotHost<'cache> {
    pub w13_weights: &'cache [u8],
    pub w2_weights: &'cache [u8],
    pub w13_scales: &'cache [u8],
    pub w2_scales: &'cache [u8],
    pub w13_input_global_scales: &'cache [u8],
    pub w13_alpha: &'cache [u8],
    pub w2_input_global_scales: &'cache [u8],
    pub w2_alpha: &'cache [u8],
}

struct QwenExpertStorageMut<'cache> {
    w13_weights: &'cache mut [u8],
    w2_weights: &'cache mut [u8],
    w13_scales: &'cache mut [u8],
    w2_scales: &'cache mut [u8],
    w13_input_global_scales: &'cache mut [u8],
    w13_alpha: &'cache mut [u8],
    w2_input_global_scales: &'cache mut [u8],
    w2_alpha: &'cache mut [u8],
}

/// A larger, fixed-size cache of expert bytes after the CPU-only ModelOpt
/// scale swizzle. The compact 16-slot hot cache remains the grouped-GEMM ABI;
/// this tier prevents repeated NVMe reads and repeated scale transforms when
/// an expert is reused by later tokens.
pub struct QwenPreparedExpertCache {
    capacity: u32,
    w13_weights: CoherentRegionOwner,
    w2_weights: CoherentRegionOwner,
    w13_scales: CoherentRegionOwner,
    w2_scales: CoherentRegionOwner,
    w13_input_global_scales: CoherentRegionOwner,
    w13_alpha: CoherentRegionOwner,
    w2_input_global_scales: CoherentRegionOwner,
    w2_alpha: CoherentRegionOwner,
    residents: BTreeMap<crate::fabric::ExpertKey, u32>,
    slot_keys: Vec<Option<crate::fabric::ExpertKey>>,
    last_used: Vec<u64>,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QwenPreparedExpertStats {
    pub capacity: u32,
    pub resident: u32,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// File-descriptor and scale-scratch owner for explicit expert fills. Unlike
/// `FlashNextWeightMaps`, this path never CUDA-registers a complete checkpoint
/// shard: it reads only the selected expert tensor ranges into the fixed hot
/// cache and keeps shard file descriptors open across tokens.
pub struct QwenExpertFileLoader {
    checkpoint: FlashNextCheckpoint,
    files: std::collections::BTreeMap<PathBuf, MappedExpertFile>,
    scale_scratch: Vec<u8>,
}

/// Read-only source mapping for expert tensors. It is deliberately not CUDA
/// registered: the checkpoint remains the single NVMe-backed source while a
/// bounded prepared tier owns only promoted experts. `posix_fadvise` is issued
/// for all misses before any copy so the kernel can queue their pages in one
/// I/O wave instead of servicing thousands of synchronous positional reads.
struct MappedExpertFile {
    file: File,
    address: *mut libc::c_void,
    bytes: usize,
}

impl MappedExpertFile {
    fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let bytes = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::other("expert source is too large to map"))?;
        if bytes == 0 {
            return Err(std::io::Error::other("expert source file is empty"));
        }
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        // Expert ids are router-dependent; explicit WILLNEED calls below own
        // readahead, while RANDOM prevents accidental whole-shard streaming.
        unsafe {
            libc::madvise(address, bytes, libc::MADV_RANDOM);
        }
        Ok(Self { file, address, bytes })
    }

    fn read_exact_at(&self, output: &mut [u8], offset: u64) -> std::io::Result<()> {
        let begin = usize::try_from(offset)
            .map_err(|_| std::io::Error::other("expert source offset overflow"))?;
        let end = begin
            .checked_add(output.len())
            .ok_or_else(|| std::io::Error::other("expert source range overflow"))?;
        if end > self.bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "expert tensor exceeds mapped source",
            ));
        }
        let source = unsafe {
            std::slice::from_raw_parts(self.address.cast::<u8>().add(begin), output.len())
        };
        output.copy_from_slice(source);
        Ok(())
    }

    fn will_need(&self, offset: u64, bytes: u64) -> std::io::Result<()> {
        let offset = i64::try_from(offset)
            .map_err(|_| std::io::Error::other("expert prefetch offset overflow"))?;
        let bytes = i64::try_from(bytes)
            .map_err(|_| std::io::Error::other("expert prefetch length overflow"))?;
        let status = unsafe {
            libc::posix_fadvise(
                self.file.as_raw_fd(),
                offset,
                bytes,
                libc::POSIX_FADV_WILLNEED,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(std::io::Error::from_raw_os_error(status))
        }
    }

    /// Populate the kernel page cache by reading this source mapping in large
    /// sequential windows. The scratch buffer is reused for every file and is
    /// discarded after startup; the persistent source remains file-backed.
    fn warm_sequential(&self, scratch: &mut [u8]) -> std::io::Result<u64> {
        if scratch.is_empty() {
            return Err(std::io::Error::other("expert warm scratch is empty"));
        }
        let length = i64::try_from(self.bytes)
            .map_err(|_| std::io::Error::other("expert file is too large"))?;
        let status = unsafe {
            libc::posix_fadvise(
                self.file.as_raw_fd(),
                0,
                length,
                libc::POSIX_FADV_SEQUENTIAL,
            )
        };
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status));
        }
        let mut offset = 0_usize;
        while offset < self.bytes {
            let bytes = scratch.len().min(self.bytes - offset);
            let mut filled = 0_usize;
            while filled < bytes {
                let read = self.file.read_at(
                    &mut scratch[filled..bytes],
                    u64::try_from(offset + filled)
                        .map_err(|_| std::io::Error::other("expert warm offset overflow"))?,
                )?;
                if read == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "expert source ended during warmup",
                    ));
                }
                filled += read;
            }
            offset += bytes;
        }
        Ok(self.bytes as u64)
    }
}

impl Drop for MappedExpertFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.address, self.bytes);
        }
    }
}

impl QwenExpertFileLoader {
    pub fn new(checkpoint: &FlashNextCheckpoint) -> Self {
        Self {
            checkpoint: checkpoint.clone(),
            files: std::collections::BTreeMap::new(),
            scale_scratch: vec![0_u8; 102_400],
        }
    }

    /// Warm only checkpoint files containing routed MoE experts. This explicit
    /// Spark policy trades deterministic startup I/O for page-resident first
    /// requests while PLE stays NVMe-paged. No anonymous model-sized shadow is
    /// retained: only the file-backed mmap/page-cache pages persist.
    pub fn warm_expert_source(&mut self) -> Result<u64, QwenExpertCacheError> {
        let files = self
            .checkpoint
            .tensors
            .iter()
            .filter(|(name, _)| name.contains(".mlp.experts."))
            .map(|(_, tensor)| tensor.relative_file.clone())
            .collect::<BTreeSet<_>>();
        let mut scratch = vec![0_u8; 8 * 1024 * 1024];
        let mut total = 0_u64;
        for relative_file in files {
            self.ensure_mapped(&relative_file)?;
            total = total
                .checked_add(
                    self.files
                        .get(&relative_file)
                        .expect("expert source mapping inserted")
                        .warm_sequential(&mut scratch)?,
                )
                .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        }
        Ok(total)
    }

    fn read_tensor(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[u64],
        output: &mut [u8],
    ) -> Result<(), QwenExpertCacheError> {
        let tensor = self.checkpoint.tensor(name)?.clone();
        validate_location(name, &tensor, dtype, shape, output.len() as u64)?;
        self.ensure_mapped(&tensor.relative_file)?;
        self.files
            .get(&tensor.relative_file)
            .expect("expert shard file inserted")
            .read_exact_at(output, tensor.absolute_offset)?;
        Ok(())
    }

    fn read_interleaved_scale(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        output: &mut [u8],
    ) -> Result<(), QwenExpertCacheError> {
        let bytes = rows
            .checked_mul(columns)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        if output.len() != bytes || self.scale_scratch.len() < bytes {
            return Err(QwenExpertCacheError::IntegerOverflow);
        }
        let tensor = self.checkpoint.tensor(name)?.clone();
        validate_location(
            name,
            &tensor,
            "F8_E4M3",
            &[rows as u64, columns as u64],
            bytes as u64,
        )?;
        self.ensure_mapped(&tensor.relative_file)?;
        self.files
            .get(&tensor.relative_file)
            .expect("expert shard file inserted")
            .read_exact_at(&mut self.scale_scratch[..bytes], tensor.absolute_offset)?;
        interleave_128x4(&self.scale_scratch[..bytes], output, rows, columns);
        Ok(())
    }

    fn read_f32(&mut self, name: &str) -> Result<f32, QwenExpertCacheError> {
        let mut bytes = [0_u8; 4];
        self.read_tensor(name, "F32", &[], &mut bytes)?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn ensure_mapped(&mut self, relative_file: &std::path::Path) -> Result<(), QwenExpertCacheError> {
        if !self.files.contains_key(relative_file) {
            let path = self.checkpoint.plan.root.join(relative_file);
            self.files.insert(
                relative_file.to_path_buf(),
                MappedExpertFile::open(&path)?,
            );
        }
        Ok(())
    }

    pub fn prefetch_experts(
        &mut self,
        keys: &[crate::fabric::ExpertKey],
    ) -> Result<(), QwenExpertCacheError> {
        const SUFFIXES: [&str; 10] = [
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
            "gate_proj.weight_scale",
            "up_proj.weight_scale",
            "down_proj.weight_scale",
            "gate_proj.input_scale",
            "gate_proj.weight_scale_2",
            "down_proj.input_scale",
            "down_proj.weight_scale_2",
        ];
        let mut ranges = BTreeMap::<PathBuf, Vec<(u64, u64)>>::new();
        for key in keys {
            let prefix = format!(
                "model.language_model.layers.{}.mlp.experts.{}",
                key.layer, key.expert
            );
            for suffix in SUFFIXES {
                let tensor = self.checkpoint.tensor(&format!("{prefix}.{suffix}"))?;
                ranges
                    .entry(tensor.relative_file.clone())
                    .or_default()
                    .push((tensor.absolute_offset, tensor.data_bytes));
            }
        }
        for (relative_file, mut file_ranges) in ranges {
            self.ensure_mapped(&relative_file)?;
            file_ranges.sort_unstable_by_key(|range| range.0);
            let file = self
                .files
                .get(&relative_file)
                .expect("expert shard file inserted");
            let mut merged = Vec::<(u64, u64)>::new();
            for (offset, bytes) in file_ranges {
                let end = offset
                    .checked_add(bytes)
                    .ok_or(QwenExpertCacheError::IntegerOverflow)?;
                if let Some((merged_offset, merged_bytes)) = merged.last_mut() {
                    let merged_end = merged_offset
                        .checked_add(*merged_bytes)
                        .ok_or(QwenExpertCacheError::IntegerOverflow)?;
                    // Safetensors stores one expert's three packed matrices,
                    // three scale planes, and scalar metadata in contiguous
                    // runs. Merge nearby runs so one routed token issues a few
                    // hundred advisory calls rather than several thousand.
                    if offset <= merged_end.saturating_add(64 * 1024) {
                        *merged_bytes = (*merged_bytes).max(end - *merged_offset);
                        continue;
                    }
                }
                merged.push((offset, bytes));
            }
            for (offset, bytes) in merged {
                file.will_need(offset, bytes)?;
            }
        }
        Ok(())
    }

    fn load_expert(
        &mut self,
        key: crate::fabric::ExpertKey,
        slot: u32,
        storage: &mut QwenExpertStorageMut<'_>,
    ) -> Result<(), QwenExpertCacheError> {
        let slot = slot as usize;
        let prefix = format!(
            "model.language_model.layers.{}.mlp.experts.{}",
            key.layer, key.expert
        );
        let w13_begin = slot * QWEN_W13_WEIGHT_BYTES as usize;
        let gate_end = w13_begin + QWEN_W13_WEIGHT_BYTES as usize / 2;
        let w13_end = w13_begin + QWEN_W13_WEIGHT_BYTES as usize;
        self.read_tensor(
            &format!("{prefix}.gate_proj.weight"),
            "U8",
            &[640, 1280],
            &mut storage.w13_weights[w13_begin..gate_end],
        )?;
        self.read_tensor(
            &format!("{prefix}.up_proj.weight"),
            "U8",
            &[640, 1280],
            &mut storage.w13_weights[gate_end..w13_end],
        )?;
        let w2_begin = slot * QWEN_W2_WEIGHT_BYTES as usize;
        let w2_end = w2_begin + QWEN_W2_WEIGHT_BYTES as usize;
        self.read_tensor(
            &format!("{prefix}.down_proj.weight"),
            "U8",
            &[2560, 320],
            &mut storage.w2_weights[w2_begin..w2_end],
        )?;

        let w13_scale_begin = slot * QWEN_W13_SCALE_BYTES as usize;
        let gate_scale_end = w13_scale_begin + QWEN_W13_SCALE_BYTES as usize / 2;
        let w13_scale_end = w13_scale_begin + QWEN_W13_SCALE_BYTES as usize;
        self.read_interleaved_scale(
            &format!("{prefix}.gate_proj.weight_scale"),
            640,
            160,
            &mut storage.w13_scales[w13_scale_begin..gate_scale_end],
        )?;
        self.read_interleaved_scale(
            &format!("{prefix}.up_proj.weight_scale"),
            640,
            160,
            &mut storage.w13_scales[gate_scale_end..w13_scale_end],
        )?;
        let w2_scale_begin = slot * QWEN_W2_SCALE_BYTES as usize;
        let w2_scale_end = w2_scale_begin + QWEN_W2_SCALE_BYTES as usize;
        self.read_interleaved_scale(
            &format!("{prefix}.down_proj.weight_scale"),
            2560,
            40,
            &mut storage.w2_scales[w2_scale_begin..w2_scale_end],
        )?;

        let gate_input = self.read_f32(&format!("{prefix}.gate_proj.input_scale"))?;
        let gate_weight = self.read_f32(&format!("{prefix}.gate_proj.weight_scale_2"))?;
        let down_input = self.read_f32(&format!("{prefix}.down_proj.input_scale"))?;
        let down_weight = self.read_f32(&format!("{prefix}.down_proj.weight_scale_2"))?;
        write_f32_slot(storage.w13_input_global_scales, slot, 1.0 / gate_input);
        write_f32_slot(storage.w13_alpha, slot, gate_input * gate_weight);
        write_f32_slot(storage.w2_input_global_scales, slot, 1.0 / down_input);
        write_f32_slot(storage.w2_alpha, slot, down_input * down_weight);
        Ok(())
    }
}

impl QwenExpertHotCache {
    pub fn create(flags: u32) -> Result<Self, QwenExpertCacheError> {
        let capacity = u64::from(QWEN_EXPERT_CAPACITY);
        Ok(Self {
            w13_weights: slab(capacity * QWEN_W13_WEIGHT_BYTES, flags)?,
            w2_weights: slab(capacity * QWEN_W2_WEIGHT_BYTES, flags)?,
            w13_scales: slab(capacity * QWEN_W13_SCALE_BYTES, flags)?,
            w2_scales: slab(capacity * QWEN_W2_SCALE_BYTES, flags)?,
            w13_input_global_scales: slab(capacity * 4, flags)?,
            w13_alpha: slab(capacity * 4, flags)?,
            w2_input_global_scales: slab(capacity * 4, flags)?,
            w2_alpha: slab(capacity * 4, flags)?,
        })
    }

    pub fn views(&self) -> QwenExpertHotViews {
        QwenExpertHotViews {
            capacity: QWEN_EXPERT_CAPACITY,
            w13_weights: self.w13_weights.device_address(),
            w2_weights: self.w2_weights.device_address(),
            w13_scales: self.w13_scales.device_address(),
            w2_scales: self.w2_scales.device_address(),
            w13_input_global_scales: self.w13_input_global_scales.device_address(),
            w13_alpha: self.w13_alpha.device_address(),
            w2_input_global_scales: self.w2_input_global_scales.device_address(),
            w2_alpha: self.w2_alpha.device_address(),
        }
    }

    /// Commit the registered hot-bank pages sequentially before random slot
    /// promotion. This converts scattered first-touch faults into bounded
    /// startup work and keeps request-time copies at memory bandwidth.
    pub fn prefault(&mut self) -> Result<u64, QwenExpertCacheError> {
        let mut bytes = 0_u64;
        for region in [
            &mut self.w13_weights,
            &mut self.w2_weights,
            &mut self.w13_scales,
            &mut self.w2_scales,
            &mut self.w13_input_global_scales,
            &mut self.w13_alpha,
            &mut self.w2_input_global_scales,
            &mut self.w2_alpha,
        ] {
            let payload = unsafe { region.host_payload_mut()? };
            payload.fill(0);
            bytes = bytes
                .checked_add(payload.len() as u64)
                .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        }
        Ok(bytes)
    }

    /// Read selected expert ranges from safetensors directly into the fixed
    /// cache slots. The caller must synchronize every CUDA user of this cache
    /// before the CPU aliases are written.
    ///
    /// # Safety
    ///
    /// No CUDA operation may read or write any hot-cache region until this
    /// method returns and the caller submits the next stream operation.
    pub unsafe fn load_misses_from_files(
        &mut self,
        loader: &mut QwenExpertFileLoader,
        loads: &[ExpertLoad],
    ) -> Result<(), QwenExpertCacheError> {
        if loads.is_empty() {
            return Ok(());
        }
        if loads.len() > QWEN_EXPERT_CAPACITY as usize {
            return Err(QwenExpertCacheError::TooManyFills(loads.len()));
        }
        let mut storage = unsafe { self.storage_mut()? };
        for load in loads {
            if load.address.slot >= QWEN_EXPERT_CAPACITY {
                return Err(QwenExpertCacheError::InvalidSlot(load.address.slot));
            }
            loader.load_expert(load.key, load.address.slot, &mut storage)?;
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        Ok(())
    }

    unsafe fn storage_mut(&mut self) -> Result<QwenExpertStorageMut<'_>, QwenExpertCacheError> {
        Ok(QwenExpertStorageMut {
            w13_weights: unsafe { self.w13_weights.host_payload_mut()? },
            w2_weights: unsafe { self.w2_weights.host_payload_mut()? },
            w13_scales: unsafe { self.w13_scales.host_payload_mut()? },
            w2_scales: unsafe { self.w2_scales.host_payload_mut()? },
            w13_input_global_scales: unsafe {
                self.w13_input_global_scales.host_payload_mut()?
            },
            w13_alpha: unsafe { self.w13_alpha.host_payload_mut()? },
            w2_input_global_scales: unsafe {
                self.w2_input_global_scales.host_payload_mut()?
            },
            w2_alpha: unsafe { self.w2_alpha.host_payload_mut()? },
        })
    }

    /// Inspect the coherent CPU aliases after the packing stream completes.
    /// Production execution passes the device views directly to grouped GEMM;
    /// this borrow exists for oracle parity and diagnostics.
    ///
    /// # Safety
    ///
    /// No CUDA writer may access these arenas for the returned lifetime.
    pub unsafe fn host_payloads(&self) -> Result<QwenExpertHotHost<'_>, QwenExpertCacheError> {
        Ok(QwenExpertHotHost {
            w13_weights: unsafe { self.w13_weights.host_payload()? },
            w2_weights: unsafe { self.w2_weights.host_payload()? },
            w13_scales: unsafe { self.w13_scales.host_payload()? },
            w2_scales: unsafe { self.w2_scales.host_payload()? },
            w13_input_global_scales: unsafe {
                self.w13_input_global_scales.host_payload()?
            },
            w13_alpha: unsafe { self.w13_alpha.host_payload()? },
            w2_input_global_scales: unsafe {
                self.w2_input_global_scales.host_payload()?
            },
            w2_alpha: unsafe { self.w2_alpha.host_payload()? },
        })
    }

    /// Enqueue every cache miss directly from CUDA-registered safetensors
    /// mappings. The caller keeps `weights` alive until stream completion and
    /// publishes the corresponding scheduler residency plan only afterward.
    pub fn pack_misses(
        &mut self,
        weights: &mut FlashNextWeightMaps,
        loads: &[ExpertLoad],
        stream: &mut CudaStreamOwner,
    ) -> Result<(), QwenExpertCacheError> {
        if loads.is_empty() {
            return Ok(());
        }
        if loads.len() > QWEN_EXPERT_CAPACITY as usize {
            return Err(QwenExpertCacheError::TooManyFills(loads.len()));
        }
        let mut slots = Vec::with_capacity(loads.len());
        let mut gate_weights = Vec::with_capacity(loads.len());
        let mut up_weights = Vec::with_capacity(loads.len());
        let mut down_weights = Vec::with_capacity(loads.len());
        let mut gate_scales = Vec::with_capacity(loads.len());
        let mut up_scales = Vec::with_capacity(loads.len());
        let mut down_scales = Vec::with_capacity(loads.len());
        let mut gate_input_scales = Vec::with_capacity(loads.len());
        let mut gate_weight_scale_2 = Vec::with_capacity(loads.len());
        let mut down_input_scales = Vec::with_capacity(loads.len());
        let mut down_weight_scale_2 = Vec::with_capacity(loads.len());
        for load in loads {
            let layer = load.key.layer;
            let expert = load.key.expert;
            if load.address.slot >= QWEN_EXPERT_CAPACITY {
                return Err(QwenExpertCacheError::InvalidSlot(load.address.slot));
            }
            let prefix = format!(
                "model.language_model.layers.{layer}.mlp.experts.{expert}"
            );
            slots.push(load.address.slot);
            gate_weights.push(byte_tensor(weights, &format!("{prefix}.gate_proj.weight"), "U8", &[640, 1280], 819_200)?);
            up_weights.push(byte_tensor(weights, &format!("{prefix}.up_proj.weight"), "U8", &[640, 1280], 819_200)?);
            down_weights.push(byte_tensor(weights, &format!("{prefix}.down_proj.weight"), "U8", &[2560, 320], 819_200)?);
            gate_scales.push(byte_tensor(weights, &format!("{prefix}.gate_proj.weight_scale"), "F8_E4M3", &[640, 160], 102_400)?);
            up_scales.push(byte_tensor(weights, &format!("{prefix}.up_proj.weight_scale"), "F8_E4M3", &[640, 160], 102_400)?);
            down_scales.push(byte_tensor(weights, &format!("{prefix}.down_proj.weight_scale"), "F8_E4M3", &[2560, 40], 102_400)?);
            gate_input_scales.push(float_tensor(weights, &format!("{prefix}.gate_proj.input_scale"))?);
            gate_weight_scale_2.push(float_tensor(weights, &format!("{prefix}.gate_proj.weight_scale_2"))?);
            down_input_scales.push(float_tensor(weights, &format!("{prefix}.down_proj.input_scale"))?);
            down_weight_scale_2.push(float_tensor(weights, &format!("{prefix}.down_proj.weight_scale_2"))?);
        }
        let views = self.views();
        let args = QwenExpertPackArgs {
            struct_size: u32::try_from(std::mem::size_of::<QwenExpertPackArgs>())
                .expect("Qwen expert-pack ABI fits u32"),
            abi_version: QWEN_EXPERT_PACK_ABI_VERSION,
            fills: loads.len() as u32,
            capacity: QWEN_EXPERT_CAPACITY,
            destination_slots: slots.as_ptr(),
            gate_weights: gate_weights.as_ptr(),
            up_weights: up_weights.as_ptr(),
            down_weights: down_weights.as_ptr(),
            gate_weight_scales: gate_scales.as_ptr(),
            up_weight_scales: up_scales.as_ptr(),
            down_weight_scales: down_scales.as_ptr(),
            gate_input_scales: gate_input_scales.as_ptr(),
            gate_weight_scale_2: gate_weight_scale_2.as_ptr(),
            down_input_scales: down_input_scales.as_ptr(),
            down_weight_scale_2: down_weight_scale_2.as_ptr(),
            w13_weights: pointer_mut(views.w13_weights),
            w2_weights: pointer_mut(views.w2_weights),
            w13_scales: pointer_mut(views.w13_scales),
            w2_scales: pointer_mut(views.w2_scales),
            w13_input_global_scales: pointer_mut(views.w13_input_global_scales),
            w13_alpha: pointer_mut(views.w13_alpha),
            w2_input_global_scales: pointer_mut(views.w2_input_global_scales),
            w2_alpha: pointer_mut(views.w2_alpha),
            cuda_stream: stream.raw(),
        };
        status_result(unsafe { flash_qwen_expert_pack_launch(&args) })
    }
}

impl QwenPreparedExpertCache {
    pub fn create(capacity: u32, flags: u32) -> Result<Self, QwenExpertCacheError> {
        if capacity == 0 {
            return Err(QwenExpertCacheError::InvalidPreparedCapacity(capacity));
        }
        let slots = u64::from(capacity);
        Ok(Self {
            capacity,
            w13_weights: slab(slots.checked_mul(QWEN_W13_WEIGHT_BYTES).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            w2_weights: slab(slots.checked_mul(QWEN_W2_WEIGHT_BYTES).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            w13_scales: slab(slots.checked_mul(QWEN_W13_SCALE_BYTES).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            w2_scales: slab(slots.checked_mul(QWEN_W2_SCALE_BYTES).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            w13_input_global_scales: slab(slots.checked_mul(4).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            w13_alpha: slab(slots.checked_mul(4).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            w2_input_global_scales: slab(slots.checked_mul(4).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            w2_alpha: slab(slots.checked_mul(4).ok_or(QwenExpertCacheError::IntegerOverflow)?, flags)?,
            residents: BTreeMap::new(),
            slot_keys: vec![None; capacity as usize],
            last_used: vec![0; capacity as usize],
            tick: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    pub fn stats(&self) -> QwenPreparedExpertStats {
        QwenPreparedExpertStats {
            capacity: self.capacity,
            resident: self.residents.len() as u32,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }

    /// Sequentially commit every registered prepared-tier page. The tier is a
    /// required fixed serving allocation, so deferring these faults to random
    /// routed slots only increases latency without saving steady-state memory.
    pub fn prefault(&mut self) -> Result<u64, QwenExpertCacheError> {
        let mut bytes = 0_u64;
        for region in [
            &mut self.w13_weights,
            &mut self.w2_weights,
            &mut self.w13_scales,
            &mut self.w2_scales,
            &mut self.w13_input_global_scales,
            &mut self.w13_alpha,
            &mut self.w2_input_global_scales,
            &mut self.w2_alpha,
        ] {
            let payload = unsafe { region.host_payload_mut()? };
            payload.fill(0);
            bytes = bytes
                .checked_add(payload.len() as u64)
                .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        }
        Ok(bytes)
    }

    /// Resolve selected `(layer, expert)` keys through the prepared LRU and
    /// copy their already-swizzled bytes into the compact grouped-GEMM slots.
    /// The large tier and the hot tier are both fixed CUDA-visible unified
    /// memory; there is no model-sized CPU shadow copy.
    ///
    /// # Safety
    ///
    /// The caller must synchronize all CUDA readers of both caches before
    /// entering. No CUDA work may access either cache until this method
    /// returns and the caller submits the next stream operation.
    pub unsafe fn prepare_and_promote(
        &mut self,
        loader: &mut QwenExpertFileLoader,
        hot: &mut QwenExpertHotCache,
        loads: &[ExpertLoad],
    ) -> Result<(), QwenExpertCacheError> {
        if loads.len() > QWEN_EXPERT_CAPACITY as usize {
            return Err(QwenExpertCacheError::TooManyFills(loads.len()));
        }
        for load in loads {
            if load.address.slot >= QWEN_EXPERT_CAPACITY {
                return Err(QwenExpertCacheError::InvalidSlot(load.address.slot));
            }
        }
        let keys = loads.iter().map(|load| load.key).collect::<Vec<_>>();
        let source_slots = unsafe {
            self.ensure_keys(loader, &keys, 0, self.capacity)?
        };

        let source = unsafe { self.host_payloads()? };
        let mut destination = unsafe { hot.storage_mut()? };
        for (source_slot, load) in source_slots.into_iter().zip(loads) {
            copy_expert_slot(
                &source,
                source_slot,
                &mut destination,
                load.address.slot,
            );
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Resolve and promote one layer through its fixed private prepared range.
    /// This prevents a long prefill chunk from evicting every earlier layer's
    /// experts from one global LRU, while the compact hot bank remains shared.
    ///
    /// # Safety
    ///
    /// The caller must synchronize CUDA readers of both cache tiers first.
    pub unsafe fn prepare_and_promote_layer(
        &mut self,
        loader: &mut QwenExpertFileLoader,
        hot: &mut QwenExpertHotCache,
        loads: &[ExpertLoad],
        slots_per_layer: u32,
    ) -> Result<(), QwenExpertCacheError> {
        if loads.is_empty() {
            return Ok(());
        }
        if loads.len() > QWEN_EXPERT_CAPACITY as usize {
            return Err(QwenExpertCacheError::TooManyFills(loads.len()));
        }
        let layer = loads[0].key.layer;
        for load in loads {
            if load.key.layer != layer {
                return Err(QwenExpertCacheError::MixedLayerPromotion);
            }
            if load.address.slot >= QWEN_EXPERT_CAPACITY {
                return Err(QwenExpertCacheError::InvalidSlot(load.address.slot));
            }
        }
        let first_slot = self.layer_first_slot(layer, slots_per_layer)?;
        let keys = loads.iter().map(|load| load.key).collect::<Vec<_>>();
        let source_slots = unsafe {
            self.ensure_keys(loader, &keys, first_slot, slots_per_layer)?
        };
        let source = unsafe { self.host_payloads()? };
        let mut destination = unsafe { hot.storage_mut()? };
        for (source_slot, load) in source_slots.into_iter().zip(loads) {
            copy_expert_slot(&source, source_slot, &mut destination, load.address.slot);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Keep a private, fixed-size slot range for each transformer layer and
    /// return the selected experts' local slot ids. FlashInfer can launch a
    /// fixed group count with empty groups, so callers can point the grouped
    /// GEMM directly at this layer range without repacking cache hits.
    ///
    /// # Safety
    ///
    /// The caller must synchronize all CUDA readers of this cache before a
    /// possible miss fill. No CUDA work may access the layer range until this
    /// method returns.
    pub unsafe fn ensure_layer(
        &mut self,
        loader: &mut QwenExpertFileLoader,
        layer: u16,
        slots_per_layer: u32,
        experts: &[u16],
    ) -> Result<Vec<u32>, QwenExpertCacheError> {
        let first_slot = self.layer_first_slot(layer, slots_per_layer)?;
        let keys = experts
            .iter()
            .copied()
            .map(|expert| crate::fabric::ExpertKey { layer, expert })
            .collect::<Vec<_>>();
        let global_slots = unsafe {
            self.ensure_keys(loader, &keys, first_slot, slots_per_layer)?
        };
        Ok(global_slots
            .into_iter()
            .map(|slot| slot - first_slot)
            .collect())
    }

    pub fn layer_views(
        &self,
        layer: u16,
        slots_per_layer: u32,
    ) -> Result<QwenExpertHotViews, QwenExpertCacheError> {
        let first_slot = self.layer_first_slot(layer, slots_per_layer)?;
        let first_slot = u64::from(first_slot);
        Ok(QwenExpertHotViews {
            capacity: slots_per_layer,
            w13_weights: component_address(
                self.w13_weights.device_address(),
                first_slot,
                QWEN_W13_WEIGHT_BYTES,
            )?,
            w2_weights: component_address(
                self.w2_weights.device_address(),
                first_slot,
                QWEN_W2_WEIGHT_BYTES,
            )?,
            w13_scales: component_address(
                self.w13_scales.device_address(),
                first_slot,
                QWEN_W13_SCALE_BYTES,
            )?,
            w2_scales: component_address(
                self.w2_scales.device_address(),
                first_slot,
                QWEN_W2_SCALE_BYTES,
            )?,
            w13_input_global_scales: component_address(
                self.w13_input_global_scales.device_address(),
                first_slot,
                4,
            )?,
            w13_alpha: component_address(
                self.w13_alpha.device_address(),
                first_slot,
                4,
            )?,
            w2_input_global_scales: component_address(
                self.w2_input_global_scales.device_address(),
                first_slot,
                4,
            )?,
            w2_alpha: component_address(
                self.w2_alpha.device_address(),
                first_slot,
                4,
            )?,
        })
    }

    fn layer_first_slot(
        &self,
        layer: u16,
        slots_per_layer: u32,
    ) -> Result<u32, QwenExpertCacheError> {
        if slots_per_layer == 0 {
            return Err(QwenExpertCacheError::InvalidLayerCache {
                layer,
                slots_per_layer,
                capacity: self.capacity,
            });
        }
        let first_slot = u32::from(layer)
            .checked_mul(slots_per_layer)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        let end = first_slot
            .checked_add(slots_per_layer)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        if end > self.capacity {
            return Err(QwenExpertCacheError::InvalidLayerCache {
                layer,
                slots_per_layer,
                capacity: self.capacity,
            });
        }
        Ok(first_slot)
    }

    unsafe fn ensure_keys(
        &mut self,
        loader: &mut QwenExpertFileLoader,
        keys: &[crate::fabric::ExpertKey],
        first_slot: u32,
        slot_count: u32,
    ) -> Result<Vec<u32>, QwenExpertCacheError> {
        if keys.len() > slot_count as usize {
            return Err(QwenExpertCacheError::PreparedCapacity {
                requested: keys.len(),
                capacity: slot_count,
            });
        }
        let end_slot = first_slot
            .checked_add(slot_count)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        if slot_count == 0 || end_slot > self.capacity {
            return Err(QwenExpertCacheError::InvalidPreparedRange {
                first_slot,
                slot_count,
                capacity: self.capacity,
            });
        }
        let mut requested = BTreeSet::new();
        for key in keys {
            if !requested.insert(*key) {
                return Err(QwenExpertCacheError::DuplicateExpert(*key));
            }
        }

        let mut working_tick = self.tick;
        let mut reserved = BTreeSet::new();
        let mut resolved = Vec::with_capacity(keys.len());
        let mut fills = Vec::<(
            crate::fabric::ExpertKey,
            u32,
            Option<crate::fabric::ExpertKey>,
        )>::new();
        let mut usage_updates = Vec::with_capacity(keys.len());
        let mut hit_count = 0_u64;
        let mut eviction_count = 0_u64;
        let range = first_slot as usize..end_slot as usize;

        for key in keys {
            working_tick = working_tick
                .checked_add(1)
                .ok_or(QwenExpertCacheError::IntegerOverflow)?;
            let prepared_slot = if let Some(&slot) = self.residents.get(key) {
                if slot < first_slot || slot >= end_slot {
                    return Err(QwenExpertCacheError::ResidentOutsideRange {
                        key: *key,
                        slot,
                        first_slot,
                        slot_count,
                    });
                }
                hit_count += 1;
                slot
            } else {
                let slot = range
                    .clone()
                    .find(|slot| {
                        self.slot_keys[*slot].is_none()
                            && !reserved.contains(&(*slot as u32))
                    })
                    .map(|slot| slot as u32)
                    .or_else(|| {
                        range
                            .clone()
                            .filter(|slot| {
                                !reserved.contains(&(*slot as u32))
                                    && self.slot_keys[*slot]
                                        .is_none_or(|resident| !requested.contains(&resident))
                            })
                            .min_by_key(|slot| self.last_used[*slot])
                            .map(|slot| slot as u32)
                    })
                    .ok_or(QwenExpertCacheError::NoPreparedVictim)?;
                let evicted = self.slot_keys[slot as usize];
                if evicted.is_some() {
                    eviction_count += 1;
                }
                fills.push((*key, slot, evicted));
                slot
            };
            reserved.insert(prepared_slot);
            usage_updates.push((prepared_slot, working_tick));
            resolved.push(prepared_slot);
        }

        let fill_result = {
            let fill_keys = fills.iter().map(|(key, _, _)| *key).collect::<Vec<_>>();
            loader.prefetch_experts(&fill_keys)?;
            let mut storage = unsafe { self.storage_mut()? };
            fills
                .iter()
                .try_for_each(|(key, slot, _)| loader.load_expert(*key, *slot, &mut storage))
        };
        if let Err(error) = fill_result {
            // A failed fill may have partially overwritten an old resident.
            // Invalidate only those touched slots; the untouched cache stays
            // live and no model-sized metadata rollback copy is required.
            for (_, slot, _) in fills {
                if let Some(old) = self.slot_keys[slot as usize].take() {
                    self.residents.remove(&old);
                }
                self.last_used[slot as usize] = 0;
            }
            return Err(error);
        }

        for (key, slot, evicted) in &fills {
            if let Some(evicted) = evicted {
                self.residents.remove(evicted);
            }
            self.slot_keys[*slot as usize] = Some(*key);
            self.residents.insert(*key, *slot);
        }
        for (slot, last_used) in usage_updates {
            self.last_used[slot as usize] = last_used;
        }
        self.tick = working_tick;
        self.hits = self
            .hits
            .checked_add(hit_count)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        self.misses = self
            .misses
            .checked_add(fills.len() as u64)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        self.evictions = self
            .evictions
            .checked_add(eviction_count)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        Ok(resolved)
    }

    unsafe fn storage_mut(&mut self) -> Result<QwenExpertStorageMut<'_>, QwenExpertCacheError> {
        Ok(QwenExpertStorageMut {
            w13_weights: unsafe { self.w13_weights.host_payload_mut()? },
            w2_weights: unsafe { self.w2_weights.host_payload_mut()? },
            w13_scales: unsafe { self.w13_scales.host_payload_mut()? },
            w2_scales: unsafe { self.w2_scales.host_payload_mut()? },
            w13_input_global_scales: unsafe { self.w13_input_global_scales.host_payload_mut()? },
            w13_alpha: unsafe { self.w13_alpha.host_payload_mut()? },
            w2_input_global_scales: unsafe { self.w2_input_global_scales.host_payload_mut()? },
            w2_alpha: unsafe { self.w2_alpha.host_payload_mut()? },
        })
    }

    unsafe fn host_payloads(&self) -> Result<QwenExpertHotHost<'_>, QwenExpertCacheError> {
        Ok(QwenExpertHotHost {
            w13_weights: unsafe { self.w13_weights.host_payload()? },
            w2_weights: unsafe { self.w2_weights.host_payload()? },
            w13_scales: unsafe { self.w13_scales.host_payload()? },
            w2_scales: unsafe { self.w2_scales.host_payload()? },
            w13_input_global_scales: unsafe { self.w13_input_global_scales.host_payload()? },
            w13_alpha: unsafe { self.w13_alpha.host_payload()? },
            w2_input_global_scales: unsafe { self.w2_input_global_scales.host_payload()? },
            w2_alpha: unsafe { self.w2_alpha.host_payload()? },
        })
    }
}

fn copy_expert_slot(
    source: &QwenExpertHotHost<'_>,
    source_slot: u32,
    destination: &mut QwenExpertStorageMut<'_>,
    destination_slot: u32,
) {
    copy_component(source.w13_weights, source_slot, destination.w13_weights, destination_slot, QWEN_W13_WEIGHT_BYTES as usize);
    copy_component(source.w2_weights, source_slot, destination.w2_weights, destination_slot, QWEN_W2_WEIGHT_BYTES as usize);
    copy_component(source.w13_scales, source_slot, destination.w13_scales, destination_slot, QWEN_W13_SCALE_BYTES as usize);
    copy_component(source.w2_scales, source_slot, destination.w2_scales, destination_slot, QWEN_W2_SCALE_BYTES as usize);
    copy_component(source.w13_input_global_scales, source_slot, destination.w13_input_global_scales, destination_slot, 4);
    copy_component(source.w13_alpha, source_slot, destination.w13_alpha, destination_slot, 4);
    copy_component(source.w2_input_global_scales, source_slot, destination.w2_input_global_scales, destination_slot, 4);
    copy_component(source.w2_alpha, source_slot, destination.w2_alpha, destination_slot, 4);
}

fn copy_component(
    source: &[u8],
    source_slot: u32,
    destination: &mut [u8],
    destination_slot: u32,
    slot_bytes: usize,
) {
    let source_begin = source_slot as usize * slot_bytes;
    let destination_begin = destination_slot as usize * slot_bytes;
    destination[destination_begin..destination_begin + slot_bytes]
        .copy_from_slice(&source[source_begin..source_begin + slot_bytes]);
}

fn component_address(
    base: u64,
    first_slot: u64,
    slot_bytes: u64,
) -> Result<u64, QwenExpertCacheError> {
    base.checked_add(
        first_slot
            .checked_mul(slot_bytes)
            .ok_or(QwenExpertCacheError::IntegerOverflow)?,
    )
    .ok_or(QwenExpertCacheError::IntegerOverflow)
}

fn slab(bytes: u64, flags: u32) -> Result<CoherentRegionOwner, CoherentRegionError> {
    CoherentRegionOwner::slab(bytes, ALIGNMENT, flags)
}

fn byte_tensor(
    weights: &mut FlashNextWeightMaps,
    name: &str,
    dtype: &str,
    shape: &[u64],
    bytes: u64,
) -> Result<*const u8, QwenExpertCacheError> {
    let tensor = weights.tensor(name, 1)?;
    validate_tensor(name, &tensor, dtype, shape, bytes)?;
    Ok(pointer_const(tensor.device_address))
}

fn float_tensor(
    weights: &mut FlashNextWeightMaps,
    name: &str,
) -> Result<*const f32, QwenExpertCacheError> {
    let tensor = weights.tensor(name, 4)?;
    validate_tensor(name, &tensor, "F32", &[], 4)?;
    Ok(pointer_const(tensor.device_address))
}

fn validate_tensor(
    name: &str,
    tensor: &QwenTensorView,
    dtype: &str,
    shape: &[u64],
    bytes: u64,
) -> Result<(), QwenExpertCacheError> {
    if tensor.dtype != dtype || tensor.shape != shape || tensor.data_bytes != bytes {
        return Err(QwenExpertCacheError::TensorGeometry {
            tensor: name.to_owned(),
            expected: format!("{dtype} {shape:?} {bytes} bytes"),
            actual: format!(
                "{} {:?} {} bytes",
                tensor.dtype, tensor.shape, tensor.data_bytes
            ),
        });
    }
    Ok(())
}

fn validate_location(
    name: &str,
    tensor: &crate::checkpoint::SafetensorLocation,
    dtype: &str,
    shape: &[u64],
    bytes: u64,
) -> Result<(), QwenExpertCacheError> {
    if tensor.dtype != dtype || tensor.shape != shape || tensor.data_bytes != bytes {
        return Err(QwenExpertCacheError::TensorGeometry {
            tensor: name.to_owned(),
            expected: format!("{dtype} {shape:?} {bytes} bytes"),
            actual: format!(
                "{} {:?} {} bytes",
                tensor.dtype, tensor.shape, tensor.data_bytes
            ),
        });
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
                    let destination = (((row_block * column_blocks + column_block) * 32
                        + row_lane)
                        * 4
                        + row_four)
                        * 4;
                    output[destination..destination + 4]
                        .copy_from_slice(&input[source..source + 4]);
                }
            }
        }
    }
}

fn write_f32_slot(output: &mut [u8], slot: usize, value: f32) {
    let begin = slot * 4;
    output[begin..begin + 4].copy_from_slice(&value.to_ne_bytes());
}

fn pointer_const<T>(address: u64) -> *const T {
    address as usize as *const T
}

fn pointer_mut<T>(address: u64) -> *mut T {
    address as usize as *mut T
}

fn status_result(status: Status) -> Result<(), QwenExpertCacheError> {
    if status.code == 0 {
        return Ok(());
    }
    let message = if status.message.is_null() {
        "native Qwen expert pack failed".to_owned()
    } else {
        unsafe { CStr::from_ptr(status.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(QwenExpertCacheError::Native {
        code: status.code,
        message,
    })
}

#[derive(Debug)]
pub enum QwenExpertCacheError {
    Coherent(CoherentRegionError),
    Checkpoint(CheckpointError),
    Io(std::io::Error),
    Weight(QwenWeightError),
    IntegerOverflow,
    TooManyFills(usize),
    InvalidPreparedCapacity(u32),
    PreparedCapacity { requested: usize, capacity: u32 },
    NoPreparedVictim,
    DuplicateExpert(crate::fabric::ExpertKey),
    InvalidPreparedRange { first_slot: u32, slot_count: u32, capacity: u32 },
    InvalidLayerCache { layer: u16, slots_per_layer: u32, capacity: u32 },
    ResidentOutsideRange {
        key: crate::fabric::ExpertKey,
        slot: u32,
        first_slot: u32,
        slot_count: u32,
    },
    InvalidSlot(u32),
    MixedLayerPromotion,
    TensorGeometry {
        tensor: String,
        expected: String,
        actual: String,
    },
    Native { code: i32, message: String },
}

impl From<CoherentRegionError> for QwenExpertCacheError {
    fn from(error: CoherentRegionError) -> Self { Self::Coherent(error) }
}

impl From<CheckpointError> for QwenExpertCacheError {
    fn from(error: CheckpointError) -> Self { Self::Checkpoint(error) }
}

impl From<std::io::Error> for QwenExpertCacheError {
    fn from(error: std::io::Error) -> Self { Self::Io(error) }
}

impl From<QwenWeightError> for QwenExpertCacheError {
    fn from(error: QwenWeightError) -> Self { Self::Weight(error) }
}

impl Display for QwenExpertCacheError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coherent(error) => write!(formatter, "{error}"),
            Self::Checkpoint(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Weight(error) => write!(formatter, "{error}"),
            Self::IntegerOverflow => formatter.write_str("Qwen expert loader integer overflow"),
            Self::TooManyFills(fills) => write!(formatter, "Qwen expert cache has {fills} fills, maximum is 16"),
            Self::InvalidPreparedCapacity(capacity) => write!(formatter, "Qwen prepared expert cache capacity {capacity} is invalid"),
            Self::PreparedCapacity { requested, capacity } => write!(formatter, "Qwen prepared expert cache request has {requested} experts, capacity is {capacity}"),
            Self::NoPreparedVictim => formatter.write_str("Qwen prepared expert cache has no available victim slot"),
            Self::DuplicateExpert(key) => write!(formatter, "Qwen prepared expert request duplicates layer {} expert {}", key.layer, key.expert),
            Self::InvalidPreparedRange { first_slot, slot_count, capacity } => write!(formatter, "Qwen prepared expert slot range {first_slot}..{} exceeds capacity {capacity}", first_slot.saturating_add(*slot_count)),
            Self::InvalidLayerCache { layer, slots_per_layer, capacity } => write!(formatter, "Qwen layer {layer} with {slots_per_layer} expert slots does not fit prepared capacity {capacity}"),
            Self::ResidentOutsideRange { key, slot, first_slot, slot_count } => write!(formatter, "Qwen layer {} expert {} is resident in slot {slot}, outside requested range {first_slot}..{}", key.layer, key.expert, first_slot.saturating_add(*slot_count)),
            Self::InvalidSlot(slot) => write!(formatter, "Qwen expert cache slot {slot} is out of range"),
            Self::MixedLayerPromotion => formatter.write_str("Qwen hot promotion mixes experts from different layers"),
            Self::TensorGeometry { tensor, expected, actual } => write!(formatter, "Qwen tensor {tensor} expected {expected}, got {actual}"),
            Self::Native { code, message } => write!(formatter, "{message} (native status {code})"),
        }
    }
}

impl std::error::Error for QwenExpertCacheError {}

#[cfg(test)]
mod tests {
    use super::interleave_128x4;

    #[test]
    fn optimized_scale_interleave_matches_index_oracle() {
        for (rows, columns) in [(128_usize, 40_usize), (640, 160), (2560, 40)] {
            let input = (0..rows * columns)
                .map(|index| (index.wrapping_mul(131) & 0xff) as u8)
                .collect::<Vec<_>>();
            let mut actual = vec![0_u8; input.len()];
            let mut expected = vec![0_u8; input.len()];
            interleave_128x4(&input, &mut actual, rows, columns);
            let column_blocks = columns / 4;
            for (index, value) in input.iter().copied().enumerate() {
                let row = index / columns;
                let column = index % columns;
                let destination = (((row / 128 * column_blocks + column / 4) * 32
                    + row % 32)
                    * 4
                    + row % 128 / 32)
                    * 4
                    + column % 4;
                expected[destination] = value;
            }
            assert_eq!(actual, expected, "scale interleave differs for {rows}x{columns}");
        }
    }
}

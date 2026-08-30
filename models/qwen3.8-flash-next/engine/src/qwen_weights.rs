//! Direct safetensors-to-CUDA weight views for Qwen3.8 Flash-Next.
//!
//! A shard is mapped and CUDA-registered once. Tensor addresses are offsets
//! into that original file mapping, so the serving path does not create a
//! second resident copy of the 72.5-GiB base checkpoint. Routed experts may be
//! copied from these views into the small fixed hot cache selected by the
//! router; PLE tensors continue to use the dedicated NVMe page cache.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use crate::checkpoint::{CheckpointError, FlashNextCheckpoint};
use crate::coherent::{CoherentRegionError, CoherentRegionOwner};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenTensorView {
    pub device_address: u64,
    pub data_bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

pub struct FlashNextWeightMaps {
    checkpoint: FlashNextCheckpoint,
    map_flags: u32,
    shards: BTreeMap<PathBuf, CoherentRegionOwner>,
}

impl FlashNextWeightMaps {
    /// Own checkpoint metadata so CUDA-registered shard mappings can remain
    /// alive for the complete engine lifetime without a self-reference.
    pub fn new(checkpoint: &FlashNextCheckpoint, map_flags: u32) -> Self {
        Self {
            checkpoint: checkpoint.clone(),
            map_flags,
            shards: BTreeMap::new(),
        }
    }

    pub fn mapped_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor(
        &mut self,
        name: &str,
        required_alignment: u64,
    ) -> Result<QwenTensorView, QwenWeightError> {
        if required_alignment == 0 || !required_alignment.is_power_of_two() {
            return Err(QwenWeightError::InvalidAlignment(required_alignment));
        }
        let location = self.checkpoint.tensor(name)?.clone();
        if is_nonresident_tensor(name) {
            return Err(QwenWeightError::NonResidentTensor(name.to_owned()));
        }
        self.ensure_mapped(&location.relative_file)?;
        let base = self
            .shards
            .get(&location.relative_file)
            .expect("mapped shard must exist")
            .device_address();
        let device_address = base
            .checked_add(location.absolute_offset)
            .ok_or(QwenWeightError::AddressOverflow)?;
        if !device_address.is_multiple_of(required_alignment) {
            return Err(QwenWeightError::TensorAlignment {
                tensor: name.to_owned(),
                address: device_address,
                required: required_alignment,
            });
        }
        Ok(QwenTensorView {
            device_address,
            data_bytes: location.data_bytes,
            dtype: location.dtype,
            shape: location.shape,
        })
    }

    fn ensure_mapped(&mut self, relative_file: &Path) -> Result<(), QwenWeightError> {
        if self.shards.contains_key(relative_file) {
            return Ok(());
        }
        let path = self.checkpoint.plan.root.join(relative_file);
        let bytes = std::fs::metadata(&path)?.len();
        let mapping = CoherentRegionOwner::file_read_only(&path, 0, bytes, 1, self.map_flags)?;
        self.shards.insert(relative_file.to_path_buf(), mapping);
        Ok(())
    }
}

fn is_nonresident_tensor(name: &str) -> bool {
    (name.contains(".ngram_embedding.shard_") && name.ends_with(".weight"))
        || name.starts_with("model.visual.")
        || name.contains(".visual.")
}

#[derive(Debug)]
pub enum QwenWeightError {
    Checkpoint(CheckpointError),
    Coherent(CoherentRegionError),
    Io(std::io::Error),
    InvalidAlignment(u64),
    NonResidentTensor(String),
    AddressOverflow,
    TensorAlignment {
        tensor: String,
        address: u64,
        required: u64,
    },
}

impl Display for QwenWeightError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint(error) => write!(formatter, "{error}"),
            Self::Coherent(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidAlignment(alignment) => {
                write!(formatter, "weight alignment must be a power of two, got {alignment}")
            }
            Self::NonResidentTensor(tensor) => {
                write!(formatter, "tensor {tensor} belongs to an offloaded checkpoint class")
            }
            Self::AddressOverflow => formatter.write_str("weight device address overflow"),
            Self::TensorAlignment {
                tensor,
                address,
                required,
            } => write!(
                formatter,
                "tensor {tensor} at device address {address:#x} is not {required}-byte aligned"
            ),
        }
    }
}

impl std::error::Error for QwenWeightError {}

impl From<CheckpointError> for QwenWeightError {
    fn from(error: CheckpointError) -> Self {
        Self::Checkpoint(error)
    }
}

impl From<CoherentRegionError> for QwenWeightError {
    fn from(error: CoherentRegionError) -> Self {
        Self::Coherent(error)
    }
}

impl From<std::io::Error> for QwenWeightError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

//! Explicit GLM GGUF residency and direct coherent file mappings.
//!
//! Routed expert tensors are never registered wholesale: they remain NVMe
//! sources for the fixed expert cache. The base serving path also excludes the
//! MTP block. Remaining immutable tensors are coalesced only across page-sized
//! padding gaps, then mapped from their original files exactly once.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[cfg(feature = "native-fabric")]
use crate::coherent::{CoherentRegionError, CoherentRegionOwner};
#[cfg(feature = "native-fabric")]
use crate::ffi::COHERENT_REGION_PREFAULT;
use crate::gguf::{GgmlTensorType, GgufSet};
use crate::glm_topology::{GlmTopology, GlmTopologyError};

pub const GLM_RESIDENT_COALESCE_GAP: u64 = 4096;
pub const GLM_RESIDENT_ALIGNMENT: u64 = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmResidentRange {
    pub shard: usize,
    pub path: PathBuf,
    pub file_offset: u64,
    pub payload_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmResidentTensor {
    pub region: usize,
    pub offset: u64,
    pub data_bytes: u64,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgmlTensorType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmResidentPlan {
    ranges: Vec<GlmResidentRange>,
    tensors: BTreeMap<String, GlmResidentTensor>,
    tensor_bytes: u64,
    mapped_payload_bytes: u64,
    excluded_expert_bytes: u64,
    excluded_mtp_bytes: u64,
}

impl GlmResidentPlan {
    pub fn from_gguf(set: &GgufSet, include_mtp: bool) -> Result<Self, GlmWeightError> {
        let topology = GlmTopology::from_gguf(set).map_err(GlmWeightError::Topology)?;
        let trunk_layers = topology.trunk_layer_count();
        let mut selected = Vec::<(String, usize, u64, u64, Vec<u64>, GgmlTensorType)>::new();
        let mut tensor_bytes = 0_u64;
        let mut excluded_expert_bytes = 0_u64;
        let mut excluded_mtp_bytes = 0_u64;

        for (name, tensor) in &set.tensors {
            let is_mtp = tensor_layer(name).is_some_and(|layer| layer >= trunk_layers);
            if is_mtp && !include_mtp {
                excluded_mtp_bytes = excluded_mtp_bytes
                    .checked_add(tensor.data_bytes)
                    .ok_or(GlmWeightError::Overflow)?;
                continue;
            }
            if is_routed_expert(name) {
                excluded_expert_bytes = excluded_expert_bytes
                    .checked_add(tensor.data_bytes)
                    .ok_or(GlmWeightError::Overflow)?;
                continue;
            }
            let end = tensor
                .absolute_offset
                .checked_add(tensor.data_bytes)
                .ok_or(GlmWeightError::Overflow)?;
            let shard = set
                .shards
                .get(tensor.shard)
                .ok_or(GlmWeightError::InvalidShard(tensor.shard))?;
            if end > shard.file_bytes {
                return Err(GlmWeightError::TensorOutsideShard(name.clone()));
            }
            tensor_bytes = tensor_bytes
                .checked_add(tensor.data_bytes)
                .ok_or(GlmWeightError::Overflow)?;
            selected.push((
                name.clone(),
                tensor.shard,
                tensor.absolute_offset,
                end,
                tensor.dimensions.clone(),
                tensor.tensor_type,
            ));
        }
        selected.sort_by_key(|(_, shard, start, _, _, _)| (*shard, *start));

        let mut ranges = Vec::<GlmResidentRange>::new();
        for (_, shard, start, end, _, _) in &selected {
            let merge = ranges.last_mut().filter(|range| {
                if range.shard != *shard {
                    return false;
                }
                let range_end = range.file_offset + range.payload_bytes;
                *start <= range_end.saturating_add(GLM_RESIDENT_COALESCE_GAP)
            });
            if let Some(range) = merge {
                let range_end = range.file_offset + range.payload_bytes;
                range.payload_bytes = (*end).max(range_end) - range.file_offset;
            } else {
                ranges.push(GlmResidentRange {
                    shard: *shard,
                    path: set.shards[*shard].path.clone(),
                    file_offset: *start,
                    payload_bytes: end - start,
                });
            }
        }

        let mut tensors = BTreeMap::new();
        for (name, shard, start, end, dimensions, tensor_type) in selected {
            let (region, range) = ranges
                .iter()
                .enumerate()
                .find(|(_, range)| {
                    range.shard == shard
                        && start >= range.file_offset
                        && end <= range.file_offset + range.payload_bytes
                })
                .ok_or_else(|| GlmWeightError::UnmappedTensor(name.clone()))?;
            tensors.insert(
                name,
                GlmResidentTensor {
                    region,
                    offset: start - range.file_offset,
                    data_bytes: end - start,
                    dimensions,
                    tensor_type,
                },
            );
        }
        let mapped_payload_bytes = ranges.iter().try_fold(0_u64, |total, range| {
            total
                .checked_add(range.payload_bytes)
                .ok_or(GlmWeightError::Overflow)
        })?;

        Ok(Self {
            ranges,
            tensors,
            tensor_bytes,
            mapped_payload_bytes,
            excluded_expert_bytes,
            excluded_mtp_bytes,
        })
    }

    pub fn ranges(&self) -> &[GlmResidentRange] {
        &self.ranges
    }

    pub fn tensors(&self) -> &BTreeMap<String, GlmResidentTensor> {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&GlmResidentTensor> {
        self.tensors.get(name)
    }

    pub fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }

    pub fn mapped_payload_bytes(&self) -> u64 {
        self.mapped_payload_bytes
    }

    pub fn excluded_expert_bytes(&self) -> u64 {
        self.excluded_expert_bytes
    }

    pub fn excluded_mtp_bytes(&self) -> u64 {
        self.excluded_mtp_bytes
    }

    pub fn mapped_page_bytes(&self, page_bytes: u64) -> Result<u64, GlmWeightError> {
        if page_bytes == 0 || !page_bytes.is_power_of_two() {
            return Err(GlmWeightError::InvalidPageSize(page_bytes));
        }
        self.ranges.iter().try_fold(0_u64, |total, range| {
            let first = range.file_offset / page_bytes * page_bytes;
            let end = range
                .file_offset
                .checked_add(range.payload_bytes)
                .and_then(|value| value.checked_add(page_bytes - 1))
                .ok_or(GlmWeightError::Overflow)?
                / page_bytes
                * page_bytes;
            total
                .checked_add(end - first)
                .ok_or(GlmWeightError::Overflow)
        })
    }
}

#[cfg(feature = "native-fabric")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmMappedTensor {
    pub host_address: u64,
    pub device_address: u64,
    pub data_bytes: u64,
    pub tensor_type: GgmlTensorType,
}

#[cfg(feature = "native-fabric")]
pub struct GlmResidentWeights {
    plan: GlmResidentPlan,
    regions: Vec<CoherentRegionOwner>,
}

#[cfg(feature = "native-fabric")]
impl GlmResidentWeights {
    pub fn open(
        set: &GgufSet,
        include_mtp: bool,
        prefault: bool,
    ) -> Result<Self, GlmWeightError> {
        let plan = GlmResidentPlan::from_gguf(set, include_mtp)?;
        let flags = if prefault { COHERENT_REGION_PREFAULT } else { 0 };
        let regions = plan
            .ranges()
            .iter()
            .map(|range| {
                CoherentRegionOwner::file_read_only(
                    &range.path,
                    range.file_offset,
                    range.payload_bytes,
                    GLM_RESIDENT_ALIGNMENT,
                    flags,
                )
                .map_err(GlmWeightError::Coherent)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { plan, regions })
    }

    pub fn plan(&self) -> &GlmResidentPlan {
        &self.plan
    }

    pub fn tensor(&self, name: &str) -> Result<GlmMappedTensor, GlmWeightError> {
        let tensor = self
            .plan
            .tensor(name)
            .ok_or_else(|| GlmWeightError::MissingResidentTensor(name.to_owned()))?;
        let region = self
            .regions
            .get(tensor.region)
            .ok_or(GlmWeightError::InvalidRegion(tensor.region))?;
        let host_address = region.view().host_pointer as usize as u64;
        Ok(GlmMappedTensor {
            host_address: host_address
                .checked_add(tensor.offset)
                .ok_or(GlmWeightError::Overflow)?,
            device_address: region
                .device_address()
                .checked_add(tensor.offset)
                .ok_or(GlmWeightError::Overflow)?,
            data_bytes: tensor.data_bytes,
            tensor_type: tensor.tensor_type,
        })
    }
}

fn tensor_layer(name: &str) -> Option<u16> {
    let rest = name.strip_prefix("blk.")?;
    rest.split_once('.')?.0.parse().ok()
}

fn is_routed_expert(name: &str) -> bool {
    name.ends_with(".ffn_gate_exps.weight")
        || name.ends_with(".ffn_up_exps.weight")
        || name.ends_with(".ffn_down_exps.weight")
}

#[derive(Debug)]
pub enum GlmWeightError {
    Topology(GlmTopologyError),
    InvalidShard(usize),
    InvalidRegion(usize),
    InvalidPageSize(u64),
    TensorOutsideShard(String),
    UnmappedTensor(String),
    MissingResidentTensor(String),
    Overflow,
    #[cfg(feature = "native-fabric")]
    Coherent(CoherentRegionError),
}

impl Display for GlmWeightError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(error) => write!(formatter, "invalid GLM topology: {error}"),
            Self::InvalidShard(shard) => write!(formatter, "invalid GLM shard {shard}"),
            Self::InvalidRegion(region) => write!(formatter, "invalid GLM mapping region {region}"),
            Self::InvalidPageSize(bytes) => write!(formatter, "invalid GLM mapping page size {bytes}"),
            Self::TensorOutsideShard(name) => write!(formatter, "GLM tensor {name} exceeds its shard"),
            Self::UnmappedTensor(name) => write!(formatter, "GLM tensor {name} has no resident range"),
            Self::MissingResidentTensor(name) => write!(formatter, "GLM tensor {name} is not resident"),
            Self::Overflow => formatter.write_str("GLM resident mapping size overflow"),
            #[cfg(feature = "native-fabric")]
            Self::Coherent(error) => write!(formatter, "cannot map GLM resident weights: {error}"),
        }
    }
}

impl std::error::Error for GlmWeightError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::gguf::{
        GgufMetadataEntry, GgufMetadataValue, GgufShard, GgufTensorLocation,
        GgufValueType,
    };

    use super::*;

    fn metadata(value: u64) -> GgufMetadataEntry {
        GgufMetadataEntry {
            value_type: GgufValueType::Uint64,
            value: GgufMetadataValue::Unsigned(value),
        }
    }

    fn location(shard: usize, offset: u64, bytes: u64) -> GgufTensorLocation {
        GgufTensorLocation {
            shard,
            absolute_offset: offset,
            data_bytes: bytes,
            dimensions: vec![32],
            tensor_type: GgmlTensorType::Q8_0,
        }
    }

    #[test]
    fn maps_only_base_resident_ranges_and_coalesces_padding() {
        let mut metadata_map = BTreeMap::new();
        metadata_map.insert("glm5next.block_count".into(), metadata(3));
        metadata_map.insert("glm5next.nextn_predict_layers".into(), metadata(1));
        metadata_map.insert("glm5next.leading_dense_block_count".into(), metadata(1));
        let shard = GgufShard {
            path: PathBuf::from("model-00001.gguf"),
            version: 3,
            alignment: 32,
            data_offset: 4096,
            file_bytes: 1 << 20,
            metadata: metadata_map,
            tensors: Vec::new(),
        };
        let mut tensors = BTreeMap::new();
        tensors.insert("output.weight".into(), location(0, 4096, 1024));
        tensors.insert("blk.0.ssm_a".into(), location(0, 5120, 256));
        tensors.insert("blk.0.ffn_gate.weight".into(), location(0, 5376, 1024));
        tensors.insert("blk.1.indexer.proj.weight".into(), location(0, 20_000, 1024));
        tensors.insert(
            "blk.1.ffn_gate_exps.weight".into(),
            location(0, 30_000, 100_000),
        );
        tensors.insert("blk.2.ssm_a".into(), location(0, 140_000, 256));
        tensors.insert(
            "blk.2.ffn_gate_exps.weight".into(),
            location(0, 150_000, 100_000),
        );
        let set = GgufSet {
            architecture: "glm5next".into(),
            shards: vec![shard],
            tensors,
        };

        let plan = GlmResidentPlan::from_gguf(&set, false).expect("resident plan");
        assert!(plan.tensor("output.weight").is_some());
        assert!(plan.tensor("blk.1.indexer.proj.weight").is_some());
        assert!(plan.tensor("blk.1.ffn_gate_exps.weight").is_none());
        assert!(plan.tensor("blk.2.ssm_a").is_none());
        assert_eq!(plan.excluded_expert_bytes(), 100_000);
        assert_eq!(plan.excluded_mtp_bytes(), 100_256);
        assert_eq!(plan.ranges().len(), 2);
        assert_eq!(plan.ranges()[0].file_offset, 4096);
        assert_eq!(plan.ranges()[0].payload_bytes, 2304);
        assert_eq!(
            plan.mapped_page_bytes(4096).expect("mapped page bytes"),
            12_288
        );
    }
}

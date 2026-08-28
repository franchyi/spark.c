//! GGUF expert-slice to fixed-cache I/O planning.
//!
//! Quantized expert tensors stay encoded. Each projection component owns one
//! fixed structure-of-arrays arena `[physical_slot, rows, K-block]`, so the
//! borrowed MMVQ kernel can consume cache-slot IDs directly without a gather or
//! repack. This module plans source and destination byte ranges only; Rust's I/O
//! engine and residency transaction decide when those bytes become visible.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
#[cfg(target_os = "linux")]
use std::fs::File;

use crate::fabric::{ExpertKey, ExpertResidencyPlan, RegionSpec};
use crate::ggml_quant::mmvq_geometry;
use crate::gguf::{GgmlTensorType, GgufSet, GgufTensorLocation};
#[cfg(target_os = "linux")]
use crate::uring::{FixedBufferReader, FixedRead};

const COMPONENT_ARENA_ALIGNMENT: u64 = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufExpertTensorSpec {
    pub component: String,
    pub tensor_name: String,
}

impl GgufExpertTensorSpec {
    pub fn new(component: impl Into<String>, tensor_name: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            tensor_name: tensor_name.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufLayerExpertSpec {
    pub layer: u16,
    pub tensors: Vec<GgufExpertTensorSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufExpertComponent {
    pub name: String,
    pub k: u64,
    pub rows: u64,
    /// Largest encoded slice used by any layer for this component.
    pub max_slice_bytes: u64,
    /// Fixed byte distance between cache slots. It is a common multiple of
    /// every quant block size used by this component, so the raw donor can
    /// address mixed-quant slots without repacking.
    pub slot_stride_bytes: u64,
    pub quant_types: Vec<GgmlTensorType>,
    /// Fixed offset of `[physical_slot, rows, K-block]` inside the cache arena.
    pub arena_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorSource {
    shard: usize,
    absolute_offset: u64,
    slice_bytes: u64,
    tensor_type: GgmlTensorType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufExpertRead {
    pub key: ExpertKey,
    pub physical_slot: u32,
    pub component: usize,
    pub shard: usize,
    pub source_offset: u64,
    pub destination_offset: u64,
    pub bytes: u64,
    pub tensor_type: GgmlTensorType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufExpertLaunchLayout {
    pub quant_type: GgmlTensorType,
    pub weights_offset: u64,
    pub weight_slot_stride_bytes: u64,
    pub k: u64,
    pub rows: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufExpertCatalog {
    experts: u32,
    physical_slots: u32,
    components: Vec<GgufExpertComponent>,
    sources: BTreeMap<(u16, usize), TensorSource>,
    layers: BTreeSet<u16>,
    max_useful_expert_bytes: u64,
    arena_bytes: u64,
    total_source_bytes: u64,
}

impl GgufExpertCatalog {
    /// Build the exact GLM-5-next MoE catalog from GGUF metadata and canonical
    /// tensor names. Dense leading blocks are deliberately excluded.
    pub fn build_glm53(set: &GgufSet, physical_slots: u32) -> Result<Self, GgufPagingError> {
        Self::build_glm53_mode(set, physical_slots, true)
    }

    /// Build only the 45-layer base-model catalog. MTP expert tensors remain
    /// entirely off the serving path unless speculative decoding is enabled.
    pub fn build_glm53_trunk(
        set: &GgufSet,
        physical_slots: u32,
    ) -> Result<Self, GgufPagingError> {
        Self::build_glm53_mode(set, physical_slots, false)
    }

    fn build_glm53_mode(
        set: &GgufSet,
        physical_slots: u32,
        include_mtp: bool,
    ) -> Result<Self, GgufPagingError> {
        if set.architecture != "glm5next" {
            return Err(GgufPagingError::InvalidMetadata("general.architecture"));
        }
        let metadata = set
            .shards
            .first()
            .ok_or(GgufPagingError::InvalidMetadata("shards"))?;
        let value = |key: &'static str| {
            metadata
                .metadata(key)
                .and_then(|entry| entry.as_u64())
                .ok_or(GgufPagingError::InvalidMetadata(key))
        };
        let blocks = value("glm5next.block_count")?;
        let serving_blocks = if include_mtp {
            blocks
        } else {
            blocks
                .checked_sub(value("glm5next.nextn_predict_layers")?)
                .ok_or(GgufPagingError::InvalidMetadata("glm5next.nextn_predict_layers"))?
        };
        let dense_blocks = value("glm5next.leading_dense_block_count")?;
        let experts = value("glm5next.expert_count")?;
        if dense_blocks >= serving_blocks
            || serving_blocks > u16::MAX as u64
            || experts > u32::MAX as u64
        {
            return Err(GgufPagingError::InvalidMetadata("glm5next MoE geometry"));
        }
        let layers = (dense_blocks..serving_blocks)
            .map(|layer| GgufLayerExpertSpec {
                layer: layer as u16,
                tensors: vec![
                    GgufExpertTensorSpec::new("gate", format!("blk.{layer}.ffn_gate_exps.weight")),
                    GgufExpertTensorSpec::new("up", format!("blk.{layer}.ffn_up_exps.weight")),
                    GgufExpertTensorSpec::new("down", format!("blk.{layer}.ffn_down_exps.weight")),
                ],
            })
            .collect::<Vec<_>>();
        Self::build(set, experts as u32, physical_slots, &layers)
    }

    pub fn build(
        set: &GgufSet,
        experts: u32,
        physical_slots: u32,
        layers: &[GgufLayerExpertSpec],
    ) -> Result<Self, GgufPagingError> {
        if experts == 0 || experts > u16::MAX as u32 || physical_slots == 0 || layers.is_empty() {
            return Err(GgufPagingError::InvalidConfig);
        }
        let first = &layers[0];
        if first.tensors.is_empty() {
            return Err(GgufPagingError::InvalidConfig);
        }
        let mut component_shapes = Vec::with_capacity(first.tensors.len());
        let mut component_names = BTreeSet::new();
        for tensor in &first.tensors {
            if tensor.component.is_empty() || !component_names.insert(tensor.component.clone()) {
                return Err(GgufPagingError::DuplicateComponent(
                    tensor.component.clone(),
                ));
            }
            let location = tensor_location(set, tensor)?;
            component_shapes.push(component_shape(location, experts, &tensor.tensor_name)?);
        }
        let mut component_max_slice_bytes = vec![0_u64; component_shapes.len()];
        let mut component_block_multiples = vec![1_u64; component_shapes.len()];
        let mut component_quant_types = vec![BTreeSet::new(); component_shapes.len()];

        let mut seen_layers = BTreeSet::new();
        let mut seen_tensors = BTreeSet::new();
        let mut sources = BTreeMap::new();
        let mut total_source_bytes = 0_u64;
        let mut max_useful_expert_bytes = 0_u64;
        for layer in layers {
            if !seen_layers.insert(layer.layer) {
                return Err(GgufPagingError::DuplicateLayer(layer.layer));
            }
            if layer.tensors.len() != first.tensors.len() {
                return Err(GgufPagingError::ComponentSetMismatch(layer.layer));
            }
            let mut layer_useful_expert_bytes = 0_u64;
            for (component_index, tensor) in layer.tensors.iter().enumerate() {
                if tensor.component != first.tensors[component_index].component {
                    return Err(GgufPagingError::ComponentSetMismatch(layer.layer));
                }
                if !seen_tensors.insert(tensor.tensor_name.clone()) {
                    return Err(GgufPagingError::DuplicateTensor(tensor.tensor_name.clone()));
                }
                let location = tensor_location(set, tensor)?;
                let shape = component_shape(location, experts, &tensor.tensor_name)?;
                if shape.k != component_shapes[component_index].k
                    || shape.rows != component_shapes[component_index].rows
                {
                    return Err(GgufPagingError::ShapeMismatch {
                        component: tensor.component.clone(),
                        tensor: tensor.tensor_name.clone(),
                    });
                }
                component_max_slice_bytes[component_index] =
                    component_max_slice_bytes[component_index].max(shape.slice_bytes);
                component_block_multiples[component_index] = lcm(
                    component_block_multiples[component_index],
                    shape.block_bytes,
                )?;
                component_quant_types[component_index].insert(shape.tensor_type);
                layer_useful_expert_bytes = layer_useful_expert_bytes
                    .checked_add(shape.slice_bytes)
                    .ok_or(GgufPagingError::IntegerOverflow)?;
                total_source_bytes = total_source_bytes
                    .checked_add(location.data_bytes)
                    .ok_or(GgufPagingError::IntegerOverflow)?;
                sources.insert(
                    (layer.layer, component_index),
                    TensorSource {
                        shard: location.shard,
                        absolute_offset: location.absolute_offset,
                        slice_bytes: shape.slice_bytes,
                        tensor_type: shape.tensor_type,
                    },
                );
            }
            max_useful_expert_bytes = max_useful_expert_bytes.max(layer_useful_expert_bytes);
        }

        let mut arena_bytes = 0_u64;
        let mut components = Vec::with_capacity(component_shapes.len());
        for (component_index, (tensor, shape)) in
            first.tensors.iter().zip(component_shapes).enumerate()
        {
            arena_bytes = align_up(arena_bytes, COMPONENT_ARENA_ALIGNMENT)?;
            let arena_offset = arena_bytes;
            let slot_alignment = lcm(
                component_block_multiples[component_index],
                COMPONENT_ARENA_ALIGNMENT,
            )?;
            let slot_stride_bytes =
                align_up(component_max_slice_bytes[component_index], slot_alignment)?;
            arena_bytes = arena_bytes
                .checked_add(
                    slot_stride_bytes
                        .checked_mul(u64::from(physical_slots))
                        .ok_or(GgufPagingError::IntegerOverflow)?,
                )
                .ok_or(GgufPagingError::IntegerOverflow)?;
            components.push(GgufExpertComponent {
                name: tensor.component.clone(),
                k: shape.k,
                rows: shape.rows,
                max_slice_bytes: component_max_slice_bytes[component_index],
                slot_stride_bytes,
                quant_types: component_quant_types[component_index]
                    .iter()
                    .copied()
                    .collect(),
                arena_offset,
            });
        }
        arena_bytes = align_up(arena_bytes, COMPONENT_ARENA_ALIGNMENT)?;

        Ok(Self {
            experts,
            physical_slots,
            components,
            sources,
            layers: seen_layers,
            max_useful_expert_bytes,
            arena_bytes,
            total_source_bytes,
        })
    }

    pub fn components(&self) -> &[GgufExpertComponent] {
        &self.components
    }

    pub fn component(&self, name: &str) -> Option<(usize, &GgufExpertComponent)> {
        self.components
            .iter()
            .enumerate()
            .find(|(_, component)| component.name == name)
    }

    pub fn physical_slots(&self) -> u32 {
        self.physical_slots
    }

    pub fn contains_layer(&self, layer: u16) -> bool {
        self.layers.contains(&layer)
    }

    /// Maximum exact source bytes loaded for one cold logical expert across
    /// all components and layers. Individual reads retain their exact size.
    pub fn useful_expert_bytes(&self) -> u64 {
        self.max_useful_expert_bytes
    }

    /// Actual fixed allocation, including inter-component alignment.
    pub fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    pub fn expert_source_bytes(&self) -> u64 {
        self.total_source_bytes
    }

    pub fn region_spec(&self, name: impl Into<String>) -> RegionSpec {
        RegionSpec::expert_cache(
            name,
            self.total_source_bytes,
            self.arena_bytes,
            COMPONENT_ARENA_ALIGNMENT,
        )
    }

    /// Kernel view for one layer/component. The quant tag comes from that
    /// layer's GGUF tensor while the slot stride remains graph-stable.
    pub fn launch_layout(
        &self,
        layer: u16,
        component_name: &str,
    ) -> Result<GgufExpertLaunchLayout, GgufPagingError> {
        let (component_index, component) = self
            .component(component_name)
            .ok_or_else(|| GgufPagingError::MissingComponent(component_name.to_owned()))?;
        let source = self
            .sources
            .get(&(layer, component_index))
            .ok_or(GgufPagingError::ResidencyMismatch)?;
        Ok(GgufExpertLaunchLayout {
            quant_type: source.tensor_type,
            weights_offset: component.arena_offset,
            weight_slot_stride_bytes: component.slot_stride_bytes,
            k: component.k,
            rows: component.rows,
        })
    }

    /// Expand logical cache misses into exact source-to-SoA destination reads.
    /// The residency transaction must be committed only after every returned
    /// read and checksum succeeds.
    pub fn reads_for(
        &self,
        residency: &ExpertResidencyPlan,
    ) -> Result<Vec<GgufExpertRead>, GgufPagingError> {
        let mut reads = Vec::with_capacity(residency.loads().len() * self.components.len());
        for load in residency.loads() {
            if !self.layers.contains(&load.key.layer)
                || u32::from(load.key.expert) >= self.experts
                || load.address.slot >= self.physical_slots
            {
                return Err(GgufPagingError::ResidencyMismatch);
            }
            for (component_index, component) in self.components.iter().enumerate() {
                let source = self
                    .sources
                    .get(&(load.key.layer, component_index))
                    .ok_or(GgufPagingError::ResidencyMismatch)?;
                let source_offset = source
                    .absolute_offset
                    .checked_add(
                        u64::from(load.key.expert)
                            .checked_mul(source.slice_bytes)
                            .ok_or(GgufPagingError::IntegerOverflow)?,
                    )
                    .ok_or(GgufPagingError::IntegerOverflow)?;
                let destination_offset = component
                    .arena_offset
                    .checked_add(
                        u64::from(load.address.slot)
                            .checked_mul(component.slot_stride_bytes)
                            .ok_or(GgufPagingError::IntegerOverflow)?,
                    )
                    .ok_or(GgufPagingError::IntegerOverflow)?;
                let destination_end = destination_offset
                    .checked_add(source.slice_bytes)
                    .ok_or(GgufPagingError::IntegerOverflow)?;
                if destination_end > self.arena_bytes {
                    return Err(GgufPagingError::ResidencyMismatch);
                }
                reads.push(GgufExpertRead {
                    key: load.key,
                    physical_slot: load.address.slot,
                    component: component_index,
                    shard: source.shard,
                    source_offset,
                    destination_offset,
                    bytes: source.slice_bytes,
                    tensor_type: source.tensor_type,
                });
            }
        }
        Ok(reads)
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GgufPagingStats {
    pub fill_batches: u64,
    pub read_operations: u64,
    pub requested_bytes: u64,
    pub useful_bytes: u64,
}

#[cfg(target_os = "linux")]
impl GgufPagingStats {
    pub fn requested_amplification(self) -> f64 {
        if self.useful_bytes == 0 {
            0.0
        } else {
            self.requested_bytes as f64 / self.useful_bytes as f64
        }
    }
}

/// Linux fixed-buffer reader that fills the same coherent cache arena consumed
/// by CUDA. It owns file descriptors, not weight bytes; the registered slab is
/// borrowed for the reader's lifetime and no intermediate payload is created.
#[cfg(target_os = "linux")]
pub struct GgufExpertReader<'a> {
    files: Vec<File>,
    reader: FixedBufferReader<'a>,
    stats: GgufPagingStats,
}

#[cfg(target_os = "linux")]
impl<'a> GgufExpertReader<'a> {
    pub fn open(
        set: &GgufSet,
        catalog: &GgufExpertCatalog,
        coherent_arena: &'a mut [u8],
        queue_depth: usize,
        max_batch: usize,
    ) -> Result<Self, GgufPagingError> {
        let required =
            usize::try_from(catalog.arena_bytes).map_err(|_| GgufPagingError::IntegerOverflow)?;
        if coherent_arena.len() < required {
            return Err(GgufPagingError::ArenaTooSmall {
                required: catalog.arena_bytes,
                available: coherent_arena.len() as u64,
            });
        }
        let files = set
            .shards
            .iter()
            .map(|shard| {
                File::open(&shard.path).map_err(|error| GgufPagingError::Io(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reader = FixedBufferReader::new(coherent_arena, queue_depth, max_batch)
            .map_err(|error| GgufPagingError::Io(error.to_string()))?;
        Ok(Self {
            files,
            reader,
            stats: GgufPagingStats::default(),
        })
    }

    /// Fill every miss destination. The caller still owns the two-phase
    /// residency value and must commit it only after this method succeeds.
    pub fn fill(
        &mut self,
        catalog: &GgufExpertCatalog,
        residency: &ExpertResidencyPlan,
    ) -> Result<GgufPagingStats, GgufPagingError> {
        let required =
            usize::try_from(catalog.arena_bytes).map_err(|_| GgufPagingError::IntegerOverflow)?;
        if self.reader.buffer().len() < required {
            return Err(GgufPagingError::ArenaTooSmall {
                required: catalog.arena_bytes,
                available: self.reader.buffer().len() as u64,
            });
        }
        let operations = catalog.reads_for(residency)?;
        let mut fixed_reads = Vec::with_capacity(operations.len());
        for operation in &operations {
            let file = self
                .files
                .get(operation.shard)
                .ok_or(GgufPagingError::ShardOutOfRange(operation.shard))?;
            fixed_reads.push(FixedRead {
                file,
                file_offset: operation.source_offset,
                buffer_offset: usize::try_from(operation.destination_offset)
                    .map_err(|_| GgufPagingError::IntegerOverflow)?,
                bytes: usize::try_from(operation.bytes)
                    .map_err(|_| GgufPagingError::IntegerOverflow)?,
            });
        }
        let result = self
            .reader
            .read(&fixed_reads)
            .map_err(|error| GgufPagingError::Io(error.to_string()))?;
        let useful_bytes = operations.iter().try_fold(0_u64, |total, operation| {
            total
                .checked_add(operation.bytes)
                .ok_or(GgufPagingError::IntegerOverflow)
        })?;
        self.stats.fill_batches = self.stats.fill_batches.saturating_add(1);
        self.stats.read_operations = self.stats.read_operations.saturating_add(result.operations);
        self.stats.requested_bytes = self.stats.requested_bytes.saturating_add(result.bytes);
        self.stats.useful_bytes = self.stats.useful_bytes.saturating_add(useful_bytes);
        Ok(self.stats)
    }

    pub fn stats(&self) -> GgufPagingStats {
        self.stats
    }

    pub fn arena(&self) -> &[u8] {
        self.reader.buffer()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ComponentShape {
    k: u64,
    rows: u64,
    slice_bytes: u64,
    block_bytes: u64,
    tensor_type: GgmlTensorType,
}

fn tensor_location<'a>(
    set: &'a GgufSet,
    tensor: &GgufExpertTensorSpec,
) -> Result<&'a GgufTensorLocation, GgufPagingError> {
    set.tensors
        .get(&tensor.tensor_name)
        .ok_or_else(|| GgufPagingError::MissingTensor(tensor.tensor_name.clone()))
}

fn component_shape(
    location: &GgufTensorLocation,
    experts: u32,
    tensor_name: &str,
) -> Result<ComponentShape, GgufPagingError> {
    if location.dimensions.len() != 3 {
        return Err(GgufPagingError::UnsupportedTensor(tensor_name.to_owned()));
    }
    let (block_elements, block_bytes) = mmvq_geometry(location.tensor_type)
        .ok_or_else(|| GgufPagingError::UnsupportedTensor(tensor_name.to_owned()))?;
    let k = location.dimensions[0];
    let rows = location.dimensions[1];
    if location.dimensions[2] != u64::from(experts)
        || k == 0
        || rows == 0
        || !k.is_multiple_of(block_elements)
    {
        return Err(GgufPagingError::UnsupportedTensor(tensor_name.to_owned()));
    }
    let slice_bytes = k
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(rows))
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(GgufPagingError::IntegerOverflow)?;
    let expected_bytes = slice_bytes
        .checked_mul(u64::from(experts))
        .ok_or(GgufPagingError::IntegerOverflow)?;
    if location.data_bytes != expected_bytes {
        return Err(GgufPagingError::UnsupportedTensor(tensor_name.to_owned()));
    }
    Ok(ComponentShape {
        k,
        rows,
        slice_bytes,
        block_bytes,
        tensor_type: location.tensor_type,
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufPagingError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(GgufPagingError::IntegerOverflow)
}

fn lcm(left: u64, right: u64) -> Result<u64, GgufPagingError> {
    let mut a = left;
    let mut b = right;
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    left.checked_div(a)
        .and_then(|quotient| quotient.checked_mul(right))
        .ok_or(GgufPagingError::IntegerOverflow)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufPagingError {
    InvalidConfig,
    InvalidMetadata(&'static str),
    MissingTensor(String),
    MissingComponent(String),
    DuplicateTensor(String),
    DuplicateLayer(u16),
    DuplicateComponent(String),
    ComponentSetMismatch(u16),
    UnsupportedTensor(String),
    ShapeMismatch { component: String, tensor: String },
    ResidencyMismatch,
    ArenaTooSmall { required: u64, available: u64 },
    ShardOutOfRange(usize),
    Io(String),
    IntegerOverflow,
}

impl Display for GgufPagingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig => formatter.write_str("invalid GGUF expert paging config"),
            Self::InvalidMetadata(key) => write!(formatter, "invalid GLM GGUF metadata: {key}"),
            Self::MissingTensor(tensor) => write!(formatter, "GGUF tensor {tensor} is missing"),
            Self::MissingComponent(component) => {
                write!(formatter, "GGUF expert component {component} is missing")
            }
            Self::DuplicateTensor(tensor) => {
                write!(formatter, "GGUF expert tensor {tensor} is reused")
            }
            Self::DuplicateLayer(layer) => write!(formatter, "duplicate GGUF layer {layer}"),
            Self::DuplicateComponent(component) => {
                write!(formatter, "duplicate GGUF expert component {component}")
            }
            Self::ComponentSetMismatch(layer) => {
                write!(
                    formatter,
                    "GGUF layer {layer} has a different component set"
                )
            }
            Self::UnsupportedTensor(tensor) => write!(
                formatter,
                "GGUF tensor {tensor} is not a supported contiguous [K, rows, experts] MMVQ type"
            ),
            Self::ShapeMismatch { component, tensor } => write!(
                formatter,
                "GGUF tensor {tensor} changes the {component} expert shape"
            ),
            Self::ResidencyMismatch => {
                formatter.write_str("expert residency plan does not match the GGUF catalog")
            }
            Self::ArenaTooSmall {
                required,
                available,
            } => write!(
                formatter,
                "GGUF expert arena needs {required} bytes, only {available} are available"
            ),
            Self::ShardOutOfRange(shard) => write!(formatter, "GGUF shard {shard} is missing"),
            Self::Io(message) => write!(formatter, "GGUF expert I/O failed: {message}"),
            Self::IntegerOverflow => formatter.write_str("GGUF expert paging size overflow"),
        }
    }
}

impl std::error::Error for GgufPagingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::FixedExpertCache;
    use crate::gguf::GgufTensorLocation;

    fn location(
        shard: usize,
        absolute_offset: u64,
        dimensions: Vec<u64>,
        tensor_type: GgmlTensorType,
    ) -> GgufTensorLocation {
        let (block_elements, block_bytes) = mmvq_geometry(tensor_type).expect("MMVQ type");
        let blocks = dimensions.iter().product::<u64>() / block_elements;
        GgufTensorLocation {
            shard,
            absolute_offset,
            data_bytes: blocks * block_bytes,
            dimensions,
            tensor_type,
        }
    }

    fn set() -> GgufSet {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "blk.0.gate".into(),
            location(0, 1_000, vec![256, 4, 3], GgmlTensorType::Iq2S),
        );
        tensors.insert(
            "blk.0.down".into(),
            location(1, 2_000, vec![512, 2, 3], GgmlTensorType::Iq3S),
        );
        tensors.insert(
            "blk.1.gate".into(),
            location(1, 3_000, vec![256, 4, 3], GgmlTensorType::Q2K),
        );
        tensors.insert(
            "blk.1.down".into(),
            location(1, 4_000, vec![512, 2, 3], GgmlTensorType::Iq4Xs),
        );
        GgufSet {
            architecture: "glm5next".into(),
            shards: Vec::new(),
            tensors,
        }
    }

    fn specs() -> Vec<GgufLayerExpertSpec> {
        vec![
            GgufLayerExpertSpec {
                layer: 0,
                tensors: vec![
                    GgufExpertTensorSpec::new("gate", "blk.0.gate"),
                    GgufExpertTensorSpec::new("down", "blk.0.down"),
                ],
            },
            GgufLayerExpertSpec {
                layer: 1,
                tensors: vec![
                    GgufExpertTensorSpec::new("gate", "blk.1.gate"),
                    GgufExpertTensorSpec::new("down", "blk.1.down"),
                ],
            },
        ]
    }

    #[test]
    fn plans_exact_soa_reads_for_transactional_cache_misses() {
        let catalog = GgufExpertCatalog::build(&set(), 3, 2, &specs()).expect("catalog");
        assert_eq!(catalog.useful_expert_bytes(), 880);
        assert_eq!(catalog.arena_bytes(), 919_552);
        assert_eq!(catalog.component("gate").expect("gate").1.arena_offset, 0);
        assert_eq!(
            catalog.component("down").expect("down").1.arena_offset,
            440_832
        );
        assert_eq!(
            catalog
                .launch_layout(1, "gate")
                .expect("gate launch layout"),
            GgufExpertLaunchLayout {
                quant_type: GgmlTensorType::Q2K,
                weights_offset: 0,
                weight_slot_stride_bytes: 220_416,
                k: 256,
                rows: 4,
            }
        );

        let cache = FixedExpertCache::new(2, catalog.useful_expert_bytes()).expect("cache");
        let key = ExpertKey {
            layer: 1,
            expert: 2,
        };
        let residency = cache.prepare(&[key]).expect("residency");
        let reads = catalog.reads_for(&residency).expect("reads");
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[0].source_offset, 3_000 + 2 * 336);
        assert_eq!(reads[0].destination_offset, 0);
        assert_eq!(reads[0].bytes, 336);
        assert_eq!(reads[0].tensor_type, GgmlTensorType::Q2K);
        assert_eq!(reads[1].source_offset, 4_000 + 2 * 544);
        assert_eq!(reads[1].destination_offset, 440_832);
        assert_eq!(reads[1].bytes, 544);
        assert_eq!(reads.iter().map(|read| read.bytes).sum::<u64>(), 880);
    }

    #[test]
    fn cache_region_charges_the_real_aligned_allocation_not_the_source_model() {
        let catalog = GgufExpertCatalog::build(&set(), 3, 2, &specs()).expect("catalog");
        let region = catalog.region_spec("glm-mixed-quant-experts");
        assert_eq!(region.source_bytes, 4_944);
        assert_eq!(region.committed_bytes, 919_552);
        assert!(!region.keeps_shadow_copy);
    }

    #[test]
    fn rejects_noncontiguous_or_wrong_quantized_expert_tensors() {
        let mut invalid = set();
        invalid
            .tensors
            .get_mut("blk.1.down")
            .expect("tensor")
            .dimensions = vec![512, 3, 2];
        assert!(matches!(
            GgufExpertCatalog::build(&invalid, 3, 2, &specs()),
            Err(GgufPagingError::UnsupportedTensor(_))
        ));
    }
}

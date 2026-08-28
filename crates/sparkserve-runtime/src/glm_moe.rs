//! GLM-5-next routed-MoE control plane over the native GGUF pager and borrowed
//! mixed-quant MMVQ ABI. Rust owns logical-to-physical expert remapping and the
//! two-phase I/O publication; kernel launches receive only fixed cache slots.

use std::fmt::{Display, Formatter};

use crate::fabric::ExpertCacheStats;
use crate::gguf::{GgufMetadataValue, GgufSet};
use crate::gguf_paging::{
    GgufExpertCatalog, GgufExpertLaunchLayout, GgufExpertRead, GgufPagingError,
};
#[cfg(target_os = "linux")]
use crate::gguf_paging::{GgufExpertReader, GgufPagingStats};
use crate::routing::RoutePlan;
use crate::scheduler::{
    MoeScheduleError, MoeSchedulerConfig, PendingMoeStep, ReadyMoeStep, RoutedMoeScheduler,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeLaunchPlan<'a> {
    pub layer: u16,
    /// Token-major physical cache-slot IDs consumed by all three MMVQ calls.
    pub execution_route: &'a RoutePlan,
    pub gate: GgufExpertLaunchLayout,
    pub up: GgufExpertLaunchLayout,
    pub down: GgufExpertLaunchLayout,
}

/// One in-flight GLM routed layer at a time. The fixed expert arena and its
/// `GgufExpertReader` are owned outside this value so CUDA and io_uring can
/// share the same coherent allocation without a second pointer.
pub struct GlmMoePlanner {
    catalog: GgufExpertCatalog,
    routes: RoutedMoeScheduler,
}

impl GlmMoePlanner {
    pub fn new(set: &GgufSet, physical_slots: u32) -> Result<Self, GlmMoeError> {
        let catalog = GgufExpertCatalog::build_glm53(set, physical_slots)?;
        let layers = metadata_u32(set, "glm5next.block_count")?;
        let experts = metadata_u32(set, "glm5next.expert_count")?;
        let top_k = metadata_u32(set, "glm5next.expert_used_count")?;
        let hidden_size = metadata_u64(set, "glm5next.embedding_length")?;
        let config = MoeSchedulerConfig {
            layers: u16::try_from(layers).map_err(|_| GlmMoeError::InvalidMetadata)?,
            num_experts: experts,
            top_k,
            hidden_size,
            expert_slots: physical_slots,
            expert_slot_bytes: catalog.useful_expert_bytes(),
        };
        Ok(Self {
            catalog,
            routes: RoutedMoeScheduler::new(config)?,
        })
    }

    pub fn catalog(&self) -> &GgufExpertCatalog {
        &self.catalog
    }

    pub fn prepare(
        &self,
        layer: u16,
        num_tokens: u32,
        coherent_expert_ids: &[i32],
    ) -> Result<PendingMoeStep, GlmMoeError> {
        if !self.catalog.contains_layer(layer) {
            return Err(GlmMoeError::NotSparseLayer(layer));
        }
        Ok(self
            .routes
            .prepare_gate_step(layer, num_tokens, coherent_expert_ids)?)
    }

    pub fn reads_for(&self, pending: &PendingMoeStep) -> Result<Vec<GgufExpertRead>, GlmMoeError> {
        Ok(self.catalog.reads_for(pending.residency_plan())?)
    }

    /// Publish fixed cache slots only after the caller has completed every
    /// read returned by `reads_for` and any requested checksum validation.
    pub fn commit(&mut self, pending: PendingMoeStep) -> Result<ReadyMoeStep, GlmMoeError> {
        Ok(self.routes.commit_step(pending)?)
    }

    /// Linux convenience path: io_uring fills the same coherent arena and the
    /// residency transaction is committed only if all reads succeed.
    #[cfg(target_os = "linux")]
    pub fn fill_and_commit(
        &mut self,
        reader: &mut GgufExpertReader<'_>,
        pending: PendingMoeStep,
    ) -> Result<(ReadyMoeStep, GgufPagingStats), GlmMoeError> {
        let stats = reader.fill(&self.catalog, pending.residency_plan())?;
        let ready = self.routes.commit_step(pending)?;
        Ok((ready, stats))
    }

    pub fn launch_plan<'a>(
        &'a self,
        ready: &'a ReadyMoeStep,
    ) -> Result<GlmMoeLaunchPlan<'a>, GlmMoeError> {
        Ok(GlmMoeLaunchPlan {
            layer: ready.layer,
            execution_route: &ready.execution_route,
            gate: self.catalog.launch_layout(ready.layer, "gate")?,
            up: self.catalog.launch_layout(ready.layer, "up")?,
            down: self.catalog.launch_layout(ready.layer, "down")?,
        })
    }

    pub fn cache_stats(&self) -> ExpertCacheStats {
        self.routes.cache_stats()
    }
}

fn metadata_u64(set: &GgufSet, key: &'static str) -> Result<u64, GlmMoeError> {
    set.shards
        .first()
        .and_then(|shard| shard.metadata(key))
        .and_then(GgufMetadataValue::as_u64)
        .ok_or(GlmMoeError::InvalidMetadata)
}

fn metadata_u32(set: &GgufSet, key: &'static str) -> Result<u32, GlmMoeError> {
    u32::try_from(metadata_u64(set, key)?).map_err(|_| GlmMoeError::InvalidMetadata)
}

#[derive(Debug)]
pub enum GlmMoeError {
    InvalidMetadata,
    NotSparseLayer(u16),
    Paging(GgufPagingError),
    Schedule(MoeScheduleError),
}

impl From<GgufPagingError> for GlmMoeError {
    fn from(error: GgufPagingError) -> Self {
        Self::Paging(error)
    }
}

impl From<MoeScheduleError> for GlmMoeError {
    fn from(error: MoeScheduleError) -> Self {
        Self::Schedule(error)
    }
}

impl Display for GlmMoeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMetadata => formatter.write_str("invalid GLM MoE metadata"),
            Self::NotSparseLayer(layer) => {
                write!(formatter, "GLM layer {layer} is not a routed MoE layer")
            }
            Self::Paging(error) => write!(formatter, "GLM expert paging failed: {error}"),
            Self::Schedule(error) => write!(formatter, "GLM expert scheduling failed: {error}"),
        }
    }
}

impl std::error::Error for GlmMoeError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use crate::ggml_quant::mmvq_geometry;
    use crate::gguf::{
        GgmlTensorType, GgufMetadataEntry, GgufShard, GgufTensorLocation, GgufValueType,
    };

    fn metadata(value: u64) -> GgufMetadataEntry {
        GgufMetadataEntry {
            value_type: GgufValueType::Uint64,
            value: GgufMetadataValue::Unsigned(value),
        }
    }

    fn tensor(tensor_type: GgmlTensorType, offset: u64) -> GgufTensorLocation {
        let dimensions = vec![256, 4, 3];
        let (block_elements, block_bytes) = mmvq_geometry(tensor_type).expect("MMVQ type");
        GgufTensorLocation {
            shard: 0,
            absolute_offset: offset,
            data_bytes: dimensions.iter().product::<u64>() / block_elements * block_bytes,
            dimensions,
            tensor_type,
        }
    }

    fn set() -> GgufSet {
        let mut shard_metadata = BTreeMap::new();
        shard_metadata.insert("glm5next.block_count".into(), metadata(2));
        shard_metadata.insert("glm5next.leading_dense_block_count".into(), metadata(1));
        shard_metadata.insert("glm5next.expert_count".into(), metadata(3));
        shard_metadata.insert("glm5next.expert_used_count".into(), metadata(2));
        shard_metadata.insert("glm5next.embedding_length".into(), metadata(256));
        let shard = GgufShard {
            path: PathBuf::from("fixture.gguf"),
            version: 3,
            alignment: 32,
            data_offset: 0,
            file_bytes: 10_000,
            metadata: shard_metadata,
            tensors: Vec::new(),
        };
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "blk.1.ffn_gate_exps.weight".into(),
            tensor(GgmlTensorType::Iq2S, 1_000),
        );
        tensors.insert(
            "blk.1.ffn_up_exps.weight".into(),
            tensor(GgmlTensorType::Iq2S, 2_000),
        );
        tensors.insert(
            "blk.1.ffn_down_exps.weight".into(),
            tensor(GgmlTensorType::Iq3S, 3_000),
        );
        GgufSet {
            architecture: "glm5next".into(),
            shards: vec![shard],
            tensors,
        }
    }

    #[test]
    fn storage_reads_are_committed_before_physical_launch_ids_exist() {
        let set = set();
        let mut planner = GlmMoePlanner::new(&set, 2).expect("planner");
        let pending = planner.prepare(1, 1, &[0, 2]).expect("pending route");
        let reads = planner.reads_for(&pending).expect("exact reads");
        assert_eq!(reads.len(), 6);
        assert_eq!(reads.iter().map(|read| read.bytes).sum::<u64>(), 2_192);

        let ready = planner.commit(pending).expect("published slots");
        let launch = planner.launch_plan(&ready).expect("launch plan");
        assert_eq!(launch.gate.quant_type, GgmlTensorType::Iq2S);
        assert_eq!(launch.down.quant_type, GgmlTensorType::Iq3S);
        assert_eq!(launch.execution_route.route_experts.len(), 2);
        assert!(
            launch
                .execution_route
                .route_experts
                .iter()
                .all(|slot| *slot < 2)
        );
        assert!(matches!(
            planner.prepare(0, 1, &[0, 2]),
            Err(GlmMoeError::NotSparseLayer(0))
        ));
    }
}

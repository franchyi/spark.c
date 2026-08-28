//! One-copy GGUF projection plans for GLM-5.3 DSA/MLA decode.
//!
//! SGLang defines the projection graph, while the pinned llama.cpp MMVQ
//! adapter consumes the checkpoint's encoded blocks directly. In particular,
//! GGUF already stores MLA's per-head K/V matrices as 64 contiguous Q8_0
//! slices, so the runtime must not materialize SGLang's derived BF16 copies.

use std::fmt::{Display, Formatter};

use crate::ggml_quant::{QuantDenseSpec, QuantPlanError, QuantRoutedSpec};
use crate::gguf::{GgmlTensorType, GgufSet};
use crate::glm_dsa::{BF16_BYTES, GlmDsaError, GlmDsaSpec};

pub const GLM_DSA_QUANT_ALIGNMENT: u64 = 256;
pub const GLM_DSA_HEAD_CHUNK: u64 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmDsaQuantRole {
    QueryA,
    QueryB,
    KvA,
    IndexQuery,
    IndexKey,
    CompressorGate,
    AbsorbKey,
    ExpandValue,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmDsaResidentRole {
    QueryNorm,
    KvNorm,
    IndexKeyNormWeight,
    IndexKeyNormBias,
    CompressorApe,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmDsaTensorSlice {
    pub name: String,
    pub shard: usize,
    pub absolute_offset: u64,
    pub data_bytes: u64,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgmlTensorType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaHeadChunk {
    pub first_head: u16,
    pub heads: u16,
    pub input_element_offset: u64,
    pub output_element_offset: u64,
    pub expert_ids: [i32; GLM_DSA_HEAD_CHUNK as usize],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlmDsaQuantDispatch {
    Dense(QuantDenseSpec),
    HeadBatched {
        spec: QuantRoutedSpec,
        chunks: Vec<GlmDsaHeadChunk>,
    },
}

impl GlmDsaQuantDispatch {
    fn memory(&self) -> Result<crate::ggml_quant::QuantMemoryPlan, GlmDsaQuantError> {
        match self {
            Self::Dense(spec) => spec.plan().map_err(GlmDsaQuantError::Quant),
            Self::HeadBatched { spec, .. } => spec.plan().map_err(GlmDsaQuantError::Quant),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmDsaQuantOperation {
    pub role: GlmDsaQuantRole,
    pub tensor: GlmDsaTensorSlice,
    pub dispatch: GlmDsaQuantDispatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmDsaResidentTensor {
    pub role: GlmDsaResidentRole,
    pub tensor: GlmDsaTensorSlice,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmDsaF32ProjectionPlan {
    pub tensor: GlmDsaTensorSlice,
    pub rows: u64,
    pub k: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaQuantWorkspacePlan {
    pub q8_scratch_offset: u64,
    pub q8_scratch_bytes: u64,
    pub output_offset: u64,
    pub output_bytes: u64,
    pub total_bytes: u64,
}

impl GlmDsaQuantWorkspacePlan {
    fn from_operations(operations: &[GlmDsaQuantOperation]) -> Result<Self, GlmDsaQuantError> {
        let mut q8_scratch_bytes = 0;
        let mut output_bytes = 0;
        for operation in operations {
            let memory = operation.dispatch.memory()?;
            q8_scratch_bytes = q8_scratch_bytes.max(memory.q8_scratch_bytes);
            output_bytes = output_bytes.max(memory.output_bytes);
        }
        let output_offset = align_up(q8_scratch_bytes, GLM_DSA_QUANT_ALIGNMENT)?;
        let total_bytes = align_up(
            output_offset
                .checked_add(output_bytes)
                .ok_or(GlmDsaQuantError::Overflow)?,
            GLM_DSA_QUANT_ALIGNMENT,
        )?;
        Ok(Self {
            q8_scratch_offset: 0,
            q8_scratch_bytes,
            output_offset,
            output_bytes,
            total_bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmDsaQuantLayerPlan {
    pub layer: u16,
    pub operations: Vec<GlmDsaQuantOperation>,
    pub resident_tensors: Vec<GlmDsaResidentTensor>,
    pub head_gate: GlmDsaF32ProjectionPlan,
    pub workspace: GlmDsaQuantWorkspacePlan,
}

impl GlmDsaQuantLayerPlan {
    pub fn from_gguf(set: &GgufSet, layer: u16) -> Result<Self, GlmDsaQuantError> {
        let spec = GlmDsaSpec::from_gguf(set).map_err(GlmDsaQuantError::Dsa)?;
        if !spec.dsa_layers.contains(&layer) {
            return Err(GlmDsaQuantError::NotDsaLayer(layer));
        }
        Self::build(set, &spec, layer)
    }

    fn build(set: &GgufSet, spec: &GlmDsaSpec, layer: u16) -> Result<Self, GlmDsaQuantError> {
        let prefix = format!("blk.{layer}");
        let mut operations = Vec::with_capacity(9);
        for (role, suffix) in [
            (GlmDsaQuantRole::QueryA, "attn_q_a.weight"),
            (GlmDsaQuantRole::KvA, "attn_kv_a_mqa.weight"),
            (GlmDsaQuantRole::QueryB, "attn_q_b.weight"),
            (GlmDsaQuantRole::IndexQuery, "indexer.attn_q_b.weight"),
            (GlmDsaQuantRole::IndexKey, "indexer.attn_k.weight"),
            (
                GlmDsaQuantRole::CompressorGate,
                "indexer_compressor_gate.weight",
            ),
        ] {
            let tensor = tensor_slice(set, &format!("{prefix}.{suffix}"))?;
            let (&k, &rows) = match tensor.dimensions.as_slice() {
                [k, rows] => (k, rows),
                _ => return Err(GlmDsaQuantError::InvalidTensor(tensor.name)),
            };
            let dispatch = GlmDsaQuantDispatch::Dense(QuantDenseSpec {
                quant_type: tensor.tensor_type,
                vectors: 1,
                rows,
                k,
            });
            dispatch.memory()?;
            operations.push(GlmDsaQuantOperation {
                role,
                tensor,
                dispatch,
            });
        }

        operations.push(head_operation(
            set,
            &format!("{prefix}.attn_k_b.weight"),
            GlmDsaQuantRole::AbsorbKey,
            spec.heads,
        )?);
        operations.push(head_operation(
            set,
            &format!("{prefix}.attn_v_b.weight"),
            GlmDsaQuantRole::ExpandValue,
            spec.heads,
        )?);

        let output_tensor = tensor_slice(set, &format!("{prefix}.attn_output.weight"))?;
        let (&output_k, &output_rows) = match output_tensor.dimensions.as_slice() {
            [k, rows] => (k, rows),
            _ => return Err(GlmDsaQuantError::InvalidTensor(output_tensor.name)),
        };
        let output_dispatch = GlmDsaQuantDispatch::Dense(QuantDenseSpec {
            quant_type: output_tensor.tensor_type,
            vectors: 1,
            rows: output_rows,
            k: output_k,
        });
        output_dispatch.memory()?;
        operations.push(GlmDsaQuantOperation {
            role: GlmDsaQuantRole::Output,
            tensor: output_tensor,
            dispatch: output_dispatch,
        });

        let resident_tensors = [
            (GlmDsaResidentRole::QueryNorm, "attn_q_a_norm.weight"),
            (GlmDsaResidentRole::KvNorm, "attn_kv_a_norm.weight"),
            (
                GlmDsaResidentRole::IndexKeyNormWeight,
                "indexer.k_norm.weight",
            ),
            (GlmDsaResidentRole::IndexKeyNormBias, "indexer.k_norm.bias"),
            (
                GlmDsaResidentRole::CompressorApe,
                "indexer_compressor_ape.weight",
            ),
        ]
        .into_iter()
        .map(|(role, suffix)| {
            Ok(GlmDsaResidentTensor {
                role,
                tensor: tensor_slice(set, &format!("{prefix}.{suffix}"))?,
            })
        })
        .collect::<Result<Vec<_>, GlmDsaQuantError>>()?;

        let head_gate_tensor = tensor_slice(set, &format!("{prefix}.indexer.proj.weight"))?;
        let (&head_gate_k, &head_gate_rows) = match head_gate_tensor.dimensions.as_slice() {
            [k, rows] => (k, rows),
            _ => return Err(GlmDsaQuantError::InvalidTensor(head_gate_tensor.name)),
        };
        let head_gate = GlmDsaF32ProjectionPlan {
            tensor: head_gate_tensor,
            rows: head_gate_rows,
            k: head_gate_k,
            input_bytes: product(&[head_gate_k, 4])?,
            output_bytes: product(&[head_gate_rows, 4])?,
        };
        let workspace = GlmDsaQuantWorkspacePlan::from_operations(&operations)?;
        Ok(Self {
            layer,
            operations,
            resident_tensors,
            head_gate,
            workspace,
        })
    }

    pub fn operation(&self, role: GlmDsaQuantRole) -> Option<&GlmDsaQuantOperation> {
        self.operations
            .iter()
            .find(|operation| operation.role == role)
    }

    pub fn quantized_weight_bytes(&self) -> Result<u64, GlmDsaQuantError> {
        self.operations.iter().try_fold(0_u64, |total, operation| {
            total
                .checked_add(operation.tensor.data_bytes)
                .ok_or(GlmDsaQuantError::Overflow)
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmDsaQuantModelPlan {
    pub layers: Vec<GlmDsaQuantLayerPlan>,
    pub workspace: GlmDsaQuantWorkspacePlan,
    pub avoided_bf16_kv_projection_bytes: u64,
}

impl GlmDsaQuantModelPlan {
    pub fn from_gguf(set: &GgufSet) -> Result<Self, GlmDsaQuantError> {
        let spec = GlmDsaSpec::from_gguf(set).map_err(GlmDsaQuantError::Dsa)?;
        Self::from_spec(set, spec)
    }

    pub fn from_gguf_trunk(set: &GgufSet) -> Result<Self, GlmDsaQuantError> {
        let spec = GlmDsaSpec::from_gguf_trunk(set).map_err(GlmDsaQuantError::Dsa)?;
        Self::from_spec(set, spec)
    }

    fn from_spec(set: &GgufSet, spec: GlmDsaSpec) -> Result<Self, GlmDsaQuantError> {
        let layers = spec
            .dsa_layers
            .iter()
            .map(|&layer| GlmDsaQuantLayerPlan::build(set, &spec, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let workspace = layers
            .first()
            .map(|layer| layer.workspace)
            .ok_or(GlmDsaQuantError::NoDsaLayers)?;
        for layer in &layers {
            if layer.workspace != workspace {
                return Err(GlmDsaQuantError::InconsistentWorkspace);
            }
        }
        let avoided_bf16_kv_projection_bytes = product(&[
            layers.len() as u64,
            2,
            spec.heads,
            spec.qk_nope_head_dim,
            spec.kv_lora_rank,
            BF16_BYTES,
        ])?;
        Ok(Self {
            layers,
            workspace,
            avoided_bf16_kv_projection_bytes,
        })
    }
}

fn head_operation(
    set: &GgufSet,
    name: &str,
    role: GlmDsaQuantRole,
    expected_heads: u64,
) -> Result<GlmDsaQuantOperation, GlmDsaQuantError> {
    let tensor = tensor_slice(set, name)?;
    let (&k, &rows, &heads) = match tensor.dimensions.as_slice() {
        [k, rows, heads] => (k, rows, heads),
        _ => return Err(GlmDsaQuantError::InvalidTensor(tensor.name)),
    };
    if heads != expected_heads || !heads.is_multiple_of(GLM_DSA_HEAD_CHUNK) {
        return Err(GlmDsaQuantError::InvalidHeadGeometry);
    }
    let slice_bytes = encoded_bytes(&[k, rows], tensor.tensor_type)?;
    let spec = QuantRoutedSpec {
        quant_type: tensor.tensor_type,
        tokens: GLM_DSA_HEAD_CHUNK,
        top_k: 1,
        weight_slots: heads,
        rows,
        k,
        weight_slot_stride_bytes: slice_bytes,
    };
    spec.plan().map_err(GlmDsaQuantError::Quant)?;
    let mut chunks = Vec::with_capacity((heads / GLM_DSA_HEAD_CHUNK) as usize);
    for first_head in (0..heads).step_by(GLM_DSA_HEAD_CHUNK as usize) {
        let mut expert_ids = [0_i32; GLM_DSA_HEAD_CHUNK as usize];
        for (offset, expert_id) in expert_ids.iter_mut().enumerate() {
            *expert_id = i32::try_from(first_head + offset as u64)
                .map_err(|_| GlmDsaQuantError::Overflow)?;
        }
        chunks.push(GlmDsaHeadChunk {
            first_head: u16::try_from(first_head).map_err(|_| GlmDsaQuantError::Overflow)?,
            heads: GLM_DSA_HEAD_CHUNK as u16,
            input_element_offset: product(&[first_head, k])?,
            output_element_offset: product(&[first_head, rows])?,
            expert_ids,
        });
    }
    Ok(GlmDsaQuantOperation {
        role,
        tensor,
        dispatch: GlmDsaQuantDispatch::HeadBatched { spec, chunks },
    })
}

fn tensor_slice(set: &GgufSet, name: &str) -> Result<GlmDsaTensorSlice, GlmDsaQuantError> {
    let tensor = set
        .tensors
        .get(name)
        .ok_or_else(|| GlmDsaQuantError::MissingTensor(name.to_owned()))?;
    let expected_bytes = encoded_bytes(&tensor.dimensions, tensor.tensor_type)?;
    if tensor.data_bytes != expected_bytes {
        return Err(GlmDsaQuantError::InvalidTensor(name.to_owned()));
    }
    let shard = set
        .shards
        .get(tensor.shard)
        .ok_or(GlmDsaQuantError::MissingShard(tensor.shard))?;
    let end = tensor
        .absolute_offset
        .checked_add(tensor.data_bytes)
        .ok_or(GlmDsaQuantError::Overflow)?;
    if tensor.absolute_offset < shard.data_offset || end > shard.file_bytes {
        return Err(GlmDsaQuantError::InvalidTensorExtent(name.to_owned()));
    }
    Ok(GlmDsaTensorSlice {
        name: name.to_owned(),
        shard: tensor.shard,
        absolute_offset: tensor.absolute_offset,
        data_bytes: tensor.data_bytes,
        dimensions: tensor.dimensions.clone(),
        tensor_type: tensor.tensor_type,
    })
}

fn encoded_bytes(dimensions: &[u64], tensor_type: GgmlTensorType) -> Result<u64, GlmDsaQuantError> {
    let elements = product(dimensions)?;
    let (block_elements, block_bytes) = tensor_type.block_geometry();
    if dimensions.is_empty() || !dimensions[0].is_multiple_of(block_elements) {
        return Err(GlmDsaQuantError::InvalidBlockGeometry);
    }
    elements
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(GlmDsaQuantError::Overflow)
}

fn product(values: &[u64]) -> Result<u64, GlmDsaQuantError> {
    values.iter().try_fold(1_u64, |total, value| {
        total.checked_mul(*value).ok_or(GlmDsaQuantError::Overflow)
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GlmDsaQuantError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(GlmDsaQuantError::Overflow)
}

#[derive(Debug, PartialEq)]
pub enum GlmDsaQuantError {
    Dsa(GlmDsaError),
    Quant(QuantPlanError),
    NotDsaLayer(u16),
    NoDsaLayers,
    MissingTensor(String),
    InvalidTensor(String),
    MissingShard(usize),
    InvalidTensorExtent(String),
    InvalidBlockGeometry,
    InvalidHeadGeometry,
    InconsistentWorkspace,
    Overflow,
}

impl Display for GlmDsaQuantError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dsa(error) => write!(formatter, "invalid GLM DSA contract: {error}"),
            Self::Quant(error) => write!(formatter, "invalid GLM MMVQ contract: {error}"),
            Self::NotDsaLayer(layer) => write!(formatter, "GLM layer {layer} is not DSA/MLA"),
            Self::NoDsaLayers => formatter.write_str("GLM model has no DSA/MLA layers"),
            Self::MissingTensor(name) => write!(formatter, "missing GLM DSA tensor {name}"),
            Self::InvalidTensor(name) => write!(formatter, "invalid GLM DSA tensor {name}"),
            Self::MissingShard(shard) => write!(formatter, "missing GGUF shard {shard}"),
            Self::InvalidTensorExtent(name) => {
                write!(formatter, "GLM DSA tensor {name} escapes its GGUF shard")
            }
            Self::InvalidBlockGeometry => formatter.write_str("invalid GGUF block geometry"),
            Self::InvalidHeadGeometry => formatter.write_str("invalid GLM MLA head geometry"),
            Self::InconsistentWorkspace => {
                formatter.write_str("GLM DSA layers require inconsistent MMVQ workspaces")
            }
            Self::Overflow => formatter.write_str("GLM DSA quant plan overflows u64"),
        }
    }
}

impl std::error::Error for GlmDsaQuantError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use crate::gguf::{
        GgufMetadataEntry, GgufMetadataValue, GgufShard, GgufTensorLocation, GgufValueType,
    };

    fn metadata(value: u64) -> GgufMetadataEntry {
        GgufMetadataEntry {
            value_type: GgufValueType::Uint32,
            value: GgufMetadataValue::Unsigned(value),
        }
    }

    fn fixture() -> GgufSet {
        let mut metadata_map = BTreeMap::new();
        for (key, value) in [
            ("glm5next.block_count", 4),
            ("glm5next.nextn_predict_layers", 1),
            ("glm5next.leading_dense_block_count", 1),
            ("glm5next.attention.head_count", 64),
            ("glm5next.attention.q_lora_rank", 1536),
            ("glm5next.attention.kv_lora_rank", 512),
            ("glm5next.attention.key_length_mla", 256),
            ("glm5next.attention.value_length_mla", 256),
            ("glm5next.rope.dimension_count", 0),
            ("glm5next.attention.indexer.head_count", 32),
            ("glm5next.attention.indexer.key_length", 128),
            ("glm5next.attention.indexer.top_k", 2048),
            ("glm5next.attention.indexer.kpool", 4),
        ] {
            metadata_map.insert(key.to_owned(), metadata(value));
        }
        metadata_map.insert(
            "glm5next.attention.layer_norm_epsilon".to_owned(),
            GgufMetadataEntry {
                value_type: GgufValueType::Float32,
                value: GgufMetadataValue::Float(1.0e-6),
            },
        );

        let mut tensors = BTreeMap::new();
        let mut cursor = 4096_u64;
        let mut add = |name: String, dimensions: &[u64], tensor_type: GgmlTensorType| {
            let data_bytes = encoded_bytes(dimensions, tensor_type).expect("fixture bytes");
            tensors.insert(
                name,
                GgufTensorLocation {
                    shard: 0,
                    absolute_offset: cursor,
                    data_bytes,
                    dimensions: dimensions.to_vec(),
                    tensor_type,
                },
            );
            cursor += data_bytes;
        };
        for layer in 0..4 {
            if layer != 3 {
                add(format!("blk.{layer}.ssm_a"), &[1], GgmlTensorType::F32);
            }
            add(
                if layer == 0 {
                    format!("blk.{layer}.ffn_gate.weight")
                } else {
                    format!("blk.{layer}.ffn_gate_exps.weight")
                },
                &[1],
                GgmlTensorType::F32,
            );
        }
        for (suffix, dimensions, tensor_type) in [
            ("attn_q_a.weight", &[4096, 1536][..], GgmlTensorType::Q6K),
            ("attn_q_a_norm.weight", &[1536][..], GgmlTensorType::F32),
            ("attn_q_b.weight", &[1536, 16384][..], GgmlTensorType::Q8_0),
            (
                "attn_kv_a_mqa.weight",
                &[4096, 512][..],
                GgmlTensorType::Q8_0,
            ),
            ("attn_kv_a_norm.weight", &[512][..], GgmlTensorType::F32),
            ("attn_k_b.weight", &[256, 512, 64][..], GgmlTensorType::Q8_0),
            ("attn_v_b.weight", &[512, 256, 64][..], GgmlTensorType::Q8_0),
            (
                "attn_output.weight",
                &[16384, 4096][..],
                GgmlTensorType::Q6K,
            ),
            (
                "indexer.attn_q_b.weight",
                &[1536, 4096][..],
                GgmlTensorType::Q8_0,
            ),
            (
                "indexer.attn_k.weight",
                &[4096, 128][..],
                GgmlTensorType::Q8_0,
            ),
            ("indexer.k_norm.weight", &[128][..], GgmlTensorType::F32),
            ("indexer.k_norm.bias", &[128][..], GgmlTensorType::F32),
            ("indexer.proj.weight", &[4096, 32][..], GgmlTensorType::F32),
            (
                "indexer_compressor_ape.weight",
                &[128, 4][..],
                GgmlTensorType::F32,
            ),
            (
                "indexer_compressor_gate.weight",
                &[4096, 128][..],
                GgmlTensorType::Q8_0,
            ),
        ] {
            add(format!("blk.3.{suffix}"), dimensions, tensor_type);
        }
        let shard = GgufShard {
            path: PathBuf::from("fixture.gguf"),
            version: 3,
            alignment: 32,
            data_offset: 4096,
            file_bytes: cursor + 4096,
            metadata: metadata_map,
            tensors: Vec::new(),
        };
        GgufSet {
            architecture: "glm5next".to_owned(),
            shards: vec![shard],
            tensors,
        }
    }

    #[test]
    fn builds_direct_quant_projection_graph() {
        let plan = GlmDsaQuantLayerPlan::from_gguf(&fixture(), 3).expect("projection plan");
        assert_eq!(plan.operations.len(), 9);
        assert_eq!(plan.resident_tensors.len(), 5);
        assert_eq!(plan.head_gate.rows, 32);
        assert_eq!(plan.head_gate.k, 4096);
        assert_eq!(plan.head_gate.tensor.data_bytes, 524_288);
        // The 16,384-wide output projection, rather than the head-batched
        // absorb/expand pair, determines the shared Q8_1 scratch maximum.
        assert_eq!(plan.workspace.q8_scratch_bytes, 18_432);
        assert_eq!(plan.workspace.output_bytes, 65_536);
        assert_eq!(plan.workspace.total_bytes, 83_968);
    }

    #[test]
    fn splits_q8_mla_heads_into_eight_donor_safe_launches() {
        let plan = GlmDsaQuantLayerPlan::from_gguf(&fixture(), 3).expect("projection plan");
        let absorb = plan
            .operation(GlmDsaQuantRole::AbsorbKey)
            .expect("absorb operation");
        assert_eq!(absorb.tensor.data_bytes, 8_912_896);
        let GlmDsaQuantDispatch::HeadBatched { spec, chunks } = &absorb.dispatch else {
            panic!("head-batched dispatch")
        };
        assert_eq!(spec.weight_slot_stride_bytes, 139_264);
        assert_eq!(spec.tokens, 8);
        assert_eq!(chunks.len(), 8);
        assert_eq!(chunks[0].expert_ids, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(chunks[7].expert_ids, [56, 57, 58, 59, 60, 61, 62, 63]);
        assert_eq!(chunks[7].input_element_offset, 56 * 256);
        assert_eq!(chunks[7].output_element_offset, 56 * 512);
    }

    #[test]
    fn model_plan_counts_the_derived_bf16_copy_we_avoid() {
        let plan = GlmDsaQuantModelPlan::from_gguf(&fixture()).expect("model plan");
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.avoided_bf16_kv_projection_bytes, 33_554_432);
    }

    #[test]
    fn rejects_declared_payload_length_drift() {
        let mut set = fixture();
        set.tensors
            .get_mut("blk.3.attn_k_b.weight")
            .expect("tensor")
            .data_bytes -= 1;
        assert_eq!(
            GlmDsaQuantLayerPlan::from_gguf(&set, 3),
            Err(GlmDsaQuantError::InvalidTensor(
                "blk.3.attn_k_b.weight".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_non_dsa_layer_and_out_of_shard_extent() {
        let set = fixture();
        assert_eq!(
            GlmDsaQuantLayerPlan::from_gguf(&set, 2),
            Err(GlmDsaQuantError::NotDsaLayer(2))
        );
        let mut invalid = set;
        invalid.shards[0].file_bytes = 4096;
        assert!(matches!(
            GlmDsaQuantLayerPlan::from_gguf(&invalid, 3),
            Err(GlmDsaQuantError::InvalidTensorExtent(_))
        ));
    }
}

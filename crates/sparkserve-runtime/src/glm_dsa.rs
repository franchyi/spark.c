//! Strict GLM-5.3 DSA/MLA and pooled-indexer geometry. SGLang is the pinned
//! semantic oracle, while Rust owns cache sizing, fixed addresses, and graph
//! admission without importing its runtime.

use std::fmt::{Display, Formatter};

use crate::ffi::{QsaExpandPlan, QsaTopkPlan};
use crate::gguf::{GgmlTensorType, GgufMetadataValue, GgufSet};
use crate::glm_mqa::GLM_MQA_SCHEDULE_BYTES;
use crate::glm_topology::{GlmTopology, GlmTopologyError};

pub const GLM_DSA_PAGE_SIZE: u64 = 64;
pub const BF16_BYTES: u64 = 2;
pub const FP8_BYTES: u64 = 1;
pub const FP32_BYTES: u64 = 4;
pub const I32_BYTES: u64 = 4;
pub const GLM_SPARSE_MLA_TOKEN_BYTES: u64 = 656;
pub const GLM_SPARSE_MLA_COMPAT_ROPE_DIM: u64 = 64;
pub const GLM_SPARSE_MLA_HISTORY_TOPK: u64 = 2048;
pub const GLM_SPARSE_MLA_TAIL_TOPK: u64 = 128;
pub const GLM_SPARSE_MLA_HISTORY_SPLITS: u64 = 32;
pub const GLM_SPARSE_MLA_TAIL_SPLITS: u64 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct GlmDsaSpec {
    pub dsa_layers: Vec<u16>,
    pub heads: u64,
    pub q_lora_rank: u64,
    pub kv_lora_rank: u64,
    pub qk_nope_head_dim: u64,
    pub qk_rope_head_dim: u64,
    pub v_head_dim: u64,
    pub index_heads: u64,
    pub index_head_dim: u64,
    pub index_topk: u64,
    pub index_kpool: u64,
    pub layer_norm_epsilon: f32,
}

impl GlmDsaSpec {
    pub fn from_gguf(set: &GgufSet) -> Result<Self, GlmDsaError> {
        Self::from_gguf_mode(set, true)
    }

    pub fn from_gguf_trunk(set: &GgufSet) -> Result<Self, GlmDsaError> {
        Self::from_gguf_mode(set, false)
    }

    fn from_gguf_mode(set: &GgufSet, include_mtp: bool) -> Result<Self, GlmDsaError> {
        let topology = GlmTopology::from_gguf(set).map_err(GlmDsaError::Topology)?;
        let spec = Self {
            dsa_layers: if include_mtp {
                topology.dsa_layers().collect()
            } else {
                topology.trunk_dsa_layers().collect()
            },
            heads: metadata_u64(set, "glm5next.attention.head_count")?,
            q_lora_rank: metadata_u64(set, "glm5next.attention.q_lora_rank")?,
            kv_lora_rank: metadata_u64(set, "glm5next.attention.kv_lora_rank")?,
            qk_nope_head_dim: metadata_u64(set, "glm5next.attention.key_length_mla")?,
            qk_rope_head_dim: metadata_u64(set, "glm5next.rope.dimension_count")?,
            v_head_dim: metadata_u64(set, "glm5next.attention.value_length_mla")?,
            index_heads: metadata_u64(set, "glm5next.attention.indexer.head_count")?,
            index_head_dim: metadata_u64(set, "glm5next.attention.indexer.key_length")?,
            index_topk: metadata_u64(set, "glm5next.attention.indexer.top_k")?,
            index_kpool: metadata_u64(set, "glm5next.attention.indexer.kpool")?,
            layer_norm_epsilon: metadata_float(set, "glm5next.attention.layer_norm_epsilon")?
                as f32,
        };
        spec.validate_locked_geometry()?;
        for &layer in &spec.dsa_layers {
            validate_layer_tensors(set, layer)?;
        }
        Ok(spec)
    }

    fn validate_locked_geometry(&self) -> Result<(), GlmDsaError> {
        for (name, actual, expected) in [
            ("heads", self.heads, 64),
            ("q_lora_rank", self.q_lora_rank, 1536),
            ("kv_lora_rank", self.kv_lora_rank, 512),
            ("qk_nope_head_dim", self.qk_nope_head_dim, 256),
            ("qk_rope_head_dim", self.qk_rope_head_dim, 0),
            ("v_head_dim", self.v_head_dim, 256),
            ("index_heads", self.index_heads, 32),
            ("index_head_dim", self.index_head_dim, 128),
            ("index_topk", self.index_topk, 2048),
            ("index_kpool", self.index_kpool, 4),
        ] {
            if actual != expected {
                return Err(GlmDsaError::UnexpectedGeometry {
                    name,
                    actual,
                    expected,
                });
            }
        }
        if self.dsa_layers.is_empty()
            || self.index_topk % self.index_kpool != 0
            || GLM_DSA_PAGE_SIZE % self.index_kpool != 0
            || !self.layer_norm_epsilon.is_finite()
            || self.layer_norm_epsilon <= 0.0
        {
            return Err(GlmDsaError::InvalidPoolContract);
        }
        Ok(())
    }

    pub fn plan(
        &self,
        sequences: u64,
        context_tokens: u64,
    ) -> Result<GlmDsaMemoryPlan, GlmDsaError> {
        if sequences == 0 || context_tokens == 0 {
            return Err(GlmDsaError::InvalidRequestGeometry);
        }
        let layers = u64::try_from(self.dsa_layers.len()).map_err(|_| GlmDsaError::Overflow)?;
        let pooled_entries = context_tokens
            .checked_add(self.index_kpool - 1)
            .ok_or(GlmDsaError::Overflow)?
            / self.index_kpool;
        // FlashInfer GLM_NSA consumes 512 FP8 values, four FP32 scales and a
        // 64-BF16 compatibility tail. GLM-5.3 has qk_rope_head_dim=0, so the
        // final 128 bytes are exact zeros while preserving the donor ABI.
        let mla_cache_bytes = product(&[
            sequences,
            layers,
            context_tokens,
            GLM_SPARSE_MLA_TOKEN_BYTES,
        ])?;
        let pooled_entry_bytes = self
            .index_head_dim
            .checked_mul(FP8_BYTES)
            .and_then(|bytes| bytes.checked_add(FP32_BYTES))
            .ok_or(GlmDsaError::Overflow)?;
        let pooled_index_cache_bytes =
            product(&[sequences, layers, pooled_entries, pooled_entry_bytes])?;
        let tail_bytes = product(&[
            sequences,
            layers,
            self.index_kpool,
            self.index_head_dim,
            2,
            BF16_BYTES,
        ])?;
        let selected = self
            .index_topk
            .checked_add(self.index_kpool - 1)
            .ok_or(GlmDsaError::Overflow)?;
        let raw_attention_query = self
            .heads
            .checked_mul(self.qk_nope_head_dim)
            .ok_or(GlmDsaError::Overflow)?;
        let index_query = self
            .index_heads
            .checked_mul(self.index_head_dim)
            .ok_or(GlmDsaError::Overflow)?;
        let absorbed_attention_query = self
            .heads
            .checked_mul(self.kv_lora_rank)
            .ok_or(GlmDsaError::Overflow)?;
        let padded_attention_query = self
            .heads
            .checked_mul(
                self.kv_lora_rank
                    .checked_add(GLM_SPARSE_MLA_COMPAT_ROPE_DIM)
                    .ok_or(GlmDsaError::Overflow)?,
            )
            .ok_or(GlmDsaError::Overflow)?;
        let query_width = self
            .q_lora_rank
            .checked_add(raw_attention_query)
            .and_then(|width| width.checked_add(index_query))
            .and_then(|width| width.checked_add(absorbed_attention_query))
            .and_then(|width| width.checked_add(padded_attention_query))
            .ok_or(GlmDsaError::Overflow)?;
        let query_workspace_bytes = product(&[sequences, query_width, BF16_BYTES])?;
        let topk_bytes = product(&[sequences, selected, I32_BYTES])?;
        let history_indices_bytes = product(&[
            sequences,
            GLM_SPARSE_MLA_HISTORY_TOPK,
            I32_BYTES,
        ])?;
        let tail_indices_bytes = product(&[
            sequences,
            GLM_SPARSE_MLA_TAIL_TOPK,
            I32_BYTES,
        ])?;
        let selection_lengths_bytes = product(&[sequences, I32_BYTES])?;
        let history_mid_out_bytes = product(&[
            sequences,
            self.heads,
            GLM_SPARSE_MLA_HISTORY_SPLITS,
            self.kv_lora_rank,
            BF16_BYTES,
        ])?;
        let history_mid_lse_bytes = product(&[
            sequences,
            self.heads,
            GLM_SPARSE_MLA_HISTORY_SPLITS,
            FP32_BYTES,
        ])?;
        let attention_output_bytes =
            product(&[sequences, self.heads, self.kv_lora_rank, BF16_BYTES])?;
        let attention_lse_bytes = product(&[sequences, self.heads, FP32_BYTES])?;
        let tail_mid_out_bytes = product(&[
            sequences,
            self.heads,
            GLM_SPARSE_MLA_TAIL_SPLITS,
            self.kv_lora_rank,
            BF16_BYTES,
        ])?;
        let tail_mid_lse_bytes = product(&[
            sequences,
            self.heads,
            GLM_SPARSE_MLA_TAIL_SPLITS,
            FP32_BYTES,
        ])?;
        Ok(GlmDsaMemoryPlan {
            sequences,
            context_tokens,
            layers,
            pooled_entries,
            mla_cache_bytes,
            pooled_index_cache_bytes,
            tail_bytes,
            query_workspace_bytes,
            topk_bytes,
            history_indices_bytes,
            tail_indices_bytes,
            selection_lengths_bytes,
            history_mid_out_bytes,
            history_mid_lse_bytes,
            attention_output_bytes,
            attention_lse_bytes,
            tail_mid_out_bytes,
            tail_mid_lse_bytes,
            mqa_schedule_bytes: GLM_MQA_SCHEDULE_BYTES,
        })
    }

    /// GLM KPool has the exact 512-group -> 2048-token + 3-tail geometry of
    /// the already extracted SGLang QSA radix/expansion kernels, so no second
    /// top-k implementation is needed.
    pub fn cuda_topk_plan(
        &self,
        rows: u32,
        pooled_columns: u32,
    ) -> Result<QsaTopkPlan, GlmDsaError> {
        if rows == 0 || pooled_columns == 0 || self.index_topk / self.index_kpool != 512 {
            return Err(GlmDsaError::InvalidRequestGeometry);
        }
        Ok(QsaTopkPlan::pooled_history(
            rows,
            pooled_columns,
            u64::from(pooled_columns),
        ))
    }

    pub fn cuda_expand_plan(&self, rows: u32) -> Result<QsaExpandPlan, GlmDsaError> {
        if rows == 0 || self.index_topk / self.index_kpool != 512 || self.index_kpool != 4 {
            return Err(GlmDsaError::InvalidRequestGeometry);
        }
        Ok(QsaExpandPlan::pooled_history(rows))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmDsaMemoryPlan {
    pub sequences: u64,
    pub context_tokens: u64,
    pub layers: u64,
    pub pooled_entries: u64,
    pub mla_cache_bytes: u64,
    pub pooled_index_cache_bytes: u64,
    pub tail_bytes: u64,
    pub query_workspace_bytes: u64,
    pub topk_bytes: u64,
    pub history_indices_bytes: u64,
    pub tail_indices_bytes: u64,
    /// One i32 length vector. The fixed arena allocates one for history and
    /// one for the unpooled tail.
    pub selection_lengths_bytes: u64,
    pub history_mid_out_bytes: u64,
    pub history_mid_lse_bytes: u64,
    pub attention_output_bytes: u64,
    pub attention_lse_bytes: u64,
    pub tail_mid_out_bytes: u64,
    pub tail_mid_lse_bytes: u64,
    pub mqa_schedule_bytes: u64,
}

impl GlmDsaMemoryPlan {
    pub fn persistent_bytes(self) -> Result<u64, GlmDsaError> {
        checked_sum(&[
            self.mla_cache_bytes,
            self.pooled_index_cache_bytes,
            self.tail_bytes,
        ])
    }

    pub fn decode_workspace_bytes(self) -> Result<u64, GlmDsaError> {
        checked_sum(&[
            self.query_workspace_bytes,
            self.topk_bytes,
            self.history_indices_bytes,
            self.tail_indices_bytes,
            self.selection_lengths_bytes,
            self.selection_lengths_bytes,
            self.history_mid_out_bytes,
            self.history_mid_lse_bytes,
            self.attention_output_bytes,
            self.attention_lse_bytes,
            self.tail_mid_out_bytes,
            self.tail_mid_lse_bytes,
            self.attention_output_bytes,
            self.attention_lse_bytes,
            self.mqa_schedule_bytes,
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmKPoolSelection {
    /// Sequence-relative token positions used for continuation/replay checks.
    pub raw: Vec<i32>,
    /// Runtime positions after an optional page-table or ragged-offset map.
    pub mapped: Vec<i32>,
}

/// Deterministic CPU oracle for SGLang's pooled-history selection. Production
/// CUDA may choose a different order for exact score ties, so fixtures avoid
/// ties and compare selected sets where the donor documents set stability.
pub fn select_pooled_history(
    spec: &GlmDsaSpec,
    logits: &[f32],
    group_length: u64,
    sequence_length: u64,
    page_table: Option<&[i32]>,
    topk_offset: Option<i32>,
) -> Result<GlmKPoolSelection, GlmDsaError> {
    if page_table.is_some() && topk_offset.is_some() {
        return Err(GlmDsaError::AmbiguousTopkMapping);
    }
    let groups = usize::try_from(group_length).map_err(|_| GlmDsaError::Overflow)?;
    if groups > logits.len()
        || group_length
            .checked_mul(spec.index_kpool)
            .ok_or(GlmDsaError::Overflow)?
            != sequence_length - sequence_length % spec.index_kpool
        || logits[..groups].iter().any(|score| !score.is_finite())
    {
        return Err(GlmDsaError::InvalidTopkInput);
    }
    if let Some(table) = page_table {
        let sequence = usize::try_from(sequence_length).map_err(|_| GlmDsaError::Overflow)?;
        if table.len() < sequence {
            return Err(GlmDsaError::InvalidTopkInput);
        }
    }
    let group_budget = spec.index_topk / spec.index_kpool;
    let valid_groups = group_length.min(group_budget);
    let mut selected = (0..groups).collect::<Vec<_>>();
    selected.sort_unstable_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    selected.truncate(usize::try_from(valid_groups).map_err(|_| GlmDsaError::Overflow)?);

    let output_width = spec
        .index_topk
        .checked_add(spec.index_kpool - 1)
        .ok_or(GlmDsaError::Overflow)?;
    let output_width = usize::try_from(output_width).map_err(|_| GlmDsaError::Overflow)?;
    let history_width = usize::try_from(
        group_length
            .checked_mul(spec.index_kpool)
            .ok_or(GlmDsaError::Overflow)?
            .min(spec.index_topk),
    )
    .map_err(|_| GlmDsaError::Overflow)?;
    let mut raw = vec![-1; output_width];
    for (rank, group) in selected.into_iter().enumerate() {
        for offset in 0..spec.index_kpool {
            let destination = rank
                .checked_mul(usize::try_from(spec.index_kpool).map_err(|_| GlmDsaError::Overflow)?)
                .and_then(|base| base.checked_add(usize::try_from(offset).ok()?))
                .ok_or(GlmDsaError::Overflow)?;
            let token = u64::try_from(group)
                .map_err(|_| GlmDsaError::Overflow)?
                .checked_mul(spec.index_kpool)
                .and_then(|base| base.checked_add(offset))
                .ok_or(GlmDsaError::Overflow)?;
            raw[destination] = i32::try_from(token).map_err(|_| GlmDsaError::Overflow)?;
        }
    }
    let tail_start = group_length
        .checked_mul(spec.index_kpool)
        .ok_or(GlmDsaError::Overflow)?;
    let tail_count = sequence_length % spec.index_kpool;
    for offset in 0..tail_count {
        raw[history_width + usize::try_from(offset).map_err(|_| GlmDsaError::Overflow)?] =
            i32::try_from(tail_start + offset).map_err(|_| GlmDsaError::Overflow)?;
    }
    let mapped = raw
        .iter()
        .map(|token| {
            if *token < 0 {
                return Ok(-1);
            }
            if let Some(table) = page_table {
                return table
                    .get(usize::try_from(*token).map_err(|_| GlmDsaError::Overflow)?)
                    .copied()
                    .ok_or(GlmDsaError::InvalidTopkInput);
            }
            token
                .checked_add(topk_offset.unwrap_or(0))
                .ok_or(GlmDsaError::Overflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GlmKPoolSelection { raw, mapped })
}

fn validate_layer_tensors(set: &GgufSet, layer: u16) -> Result<(), GlmDsaError> {
    let prefix = format!("blk.{layer}");
    for (suffix, dimensions, types) in [
        (
            "attn_q_a.weight",
            &[4096, 1536][..],
            &[GgmlTensorType::Q6K, GgmlTensorType::Q8_0][..],
        ),
        (
            "attn_q_a_norm.weight",
            &[1536][..],
            &[GgmlTensorType::F32][..],
        ),
        (
            "attn_q_b.weight",
            &[1536, 16384][..],
            &[GgmlTensorType::Q8_0][..],
        ),
        (
            "attn_kv_a_mqa.weight",
            &[4096, 512][..],
            &[GgmlTensorType::Q8_0][..],
        ),
        (
            "attn_kv_a_norm.weight",
            &[512][..],
            &[GgmlTensorType::F32][..],
        ),
        (
            "attn_k_b.weight",
            &[256, 512, 64][..],
            &[GgmlTensorType::Q8_0][..],
        ),
        (
            "attn_v_b.weight",
            &[512, 256, 64][..],
            &[GgmlTensorType::Q8_0][..],
        ),
        (
            "attn_output.weight",
            &[16384, 4096][..],
            &[GgmlTensorType::Q6K, GgmlTensorType::Q8_0][..],
        ),
        (
            "indexer.attn_q_b.weight",
            &[1536, 4096][..],
            &[GgmlTensorType::Q8_0][..],
        ),
        (
            "indexer.attn_k.weight",
            &[4096, 128][..],
            &[GgmlTensorType::Q8_0][..],
        ),
        (
            "indexer.k_norm.weight",
            &[128][..],
            &[GgmlTensorType::F32][..],
        ),
        (
            "indexer.k_norm.bias",
            &[128][..],
            &[GgmlTensorType::F32][..],
        ),
        (
            "indexer.proj.weight",
            &[4096, 32][..],
            &[GgmlTensorType::F32][..],
        ),
        (
            "indexer_compressor_ape.weight",
            &[128, 4][..],
            &[GgmlTensorType::F32][..],
        ),
        (
            "indexer_compressor_gate.weight",
            &[4096, 128][..],
            &[GgmlTensorType::Q8_0][..],
        ),
    ] {
        require_tensor(set, &format!("{prefix}.{suffix}"), dimensions, types)?;
    }
    Ok(())
}

fn require_tensor(
    set: &GgufSet,
    name: &str,
    dimensions: &[u64],
    types: &[GgmlTensorType],
) -> Result<(), GlmDsaError> {
    let tensor = set
        .tensors
        .get(name)
        .ok_or_else(|| GlmDsaError::MissingTensor(name.to_owned()))?;
    if tensor.dimensions != dimensions || !types.contains(&tensor.tensor_type) {
        return Err(GlmDsaError::InvalidTensor(name.to_owned()));
    }
    Ok(())
}

fn metadata_u64(set: &GgufSet, key: &'static str) -> Result<u64, GlmDsaError> {
    set.shards
        .first()
        .and_then(|shard| shard.metadata(key))
        .and_then(GgufMetadataValue::as_u64)
        .ok_or(GlmDsaError::InvalidMetadata(key))
}

fn metadata_float(set: &GgufSet, key: &'static str) -> Result<f64, GlmDsaError> {
    match set.shards.first().and_then(|shard| shard.metadata(key)) {
        Some(GgufMetadataValue::Float(value)) => Ok(*value),
        _ => Err(GlmDsaError::InvalidMetadata(key)),
    }
}

fn product(values: &[u64]) -> Result<u64, GlmDsaError> {
    values.iter().try_fold(1_u64, |accumulator, value| {
        accumulator.checked_mul(*value).ok_or(GlmDsaError::Overflow)
    })
}

fn checked_sum(values: &[u64]) -> Result<u64, GlmDsaError> {
    values.iter().try_fold(0_u64, |accumulator, value| {
        accumulator.checked_add(*value).ok_or(GlmDsaError::Overflow)
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum GlmDsaError {
    Topology(GlmTopologyError),
    InvalidMetadata(&'static str),
    UnexpectedGeometry {
        name: &'static str,
        actual: u64,
        expected: u64,
    },
    InvalidPoolContract,
    MissingTensor(String),
    InvalidTensor(String),
    InvalidRequestGeometry,
    InvalidTopkInput,
    AmbiguousTopkMapping,
    Overflow,
}

impl Display for GlmDsaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(error) => write!(formatter, "invalid GLM topology: {error}"),
            Self::InvalidMetadata(key) => write!(formatter, "invalid GLM DSA metadata: {key}"),
            Self::UnexpectedGeometry {
                name,
                actual,
                expected,
            } => write!(
                formatter,
                "GLM DSA {name} is {actual}, expected locked value {expected}"
            ),
            Self::InvalidPoolContract => formatter.write_str("invalid GLM KPool contract"),
            Self::MissingTensor(name) => write!(formatter, "missing GLM DSA tensor: {name}"),
            Self::InvalidTensor(name) => write!(formatter, "invalid GLM DSA tensor: {name}"),
            Self::InvalidRequestGeometry => {
                formatter.write_str("GLM DSA sequences and context must be positive")
            }
            Self::InvalidTopkInput => formatter.write_str("invalid GLM KPool top-k input"),
            Self::AmbiguousTopkMapping => {
                formatter.write_str("GLM KPool page-table and ragged-offset maps are exclusive")
            }
            Self::Overflow => formatter.write_str("GLM DSA memory geometry overflows u64"),
        }
    }
}

impl std::error::Error for GlmDsaError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use crate::gguf::{GgufMetadataEntry, GgufShard, GgufTensorLocation, GgufValueType};

    fn metadata_u32(value: u64) -> GgufMetadataEntry {
        GgufMetadataEntry {
            value_type: GgufValueType::Uint32,
            value: GgufMetadataValue::Unsigned(value),
        }
    }

    fn tensor(dimensions: &[u64], tensor_type: GgmlTensorType) -> GgufTensorLocation {
        GgufTensorLocation {
            shard: 0,
            absolute_offset: 0,
            data_bytes: 32,
            dimensions: dimensions.to_vec(),
            tensor_type,
        }
    }

    fn set() -> GgufSet {
        let mut metadata = BTreeMap::new();
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
            metadata.insert(key.to_owned(), metadata_u32(value));
        }
        metadata.insert(
            "glm5next.attention.layer_norm_epsilon".to_owned(),
            GgufMetadataEntry {
                value_type: GgufValueType::Float32,
                value: GgufMetadataValue::Float(1.0e-6),
            },
        );
        let shard = GgufShard {
            path: PathBuf::from("fixture.gguf"),
            version: 3,
            alignment: 32,
            data_offset: 0,
            file_bytes: 0,
            metadata,
            tensors: Vec::new(),
        };
        let mut tensors = BTreeMap::new();
        for layer in 0..4 {
            tensors.insert(
                if layer == 3 {
                    format!("blk.{layer}.indexer.proj.weight")
                } else {
                    format!("blk.{layer}.ssm_a")
                },
                tensor(&[1], GgmlTensorType::F32),
            );
            tensors.insert(
                if layer == 0 {
                    format!("blk.{layer}.ffn_gate.weight")
                } else {
                    format!("blk.{layer}.ffn_gate_exps.weight")
                },
                tensor(&[1], GgmlTensorType::F32),
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
            tensors.insert(format!("blk.3.{suffix}"), tensor(dimensions, tensor_type));
        }
        // The topology marker above is also the real indexer projection.
        tensors.insert(
            "blk.3.indexer.proj.weight".to_owned(),
            tensor(&[4096, 32], GgmlTensorType::F32),
        );
        GgufSet {
            architecture: "glm5next".to_owned(),
            shards: vec![shard],
            tensors,
        }
    }

    #[test]
    fn freezes_locked_dsa_tensor_and_pool_contract() {
        let spec = GlmDsaSpec::from_gguf(&set()).expect("valid DSA set");
        assert_eq!(spec.dsa_layers, vec![3]);
        assert_eq!(spec.index_topk, 2048);
        assert_eq!(spec.index_kpool, 4);
    }

    #[test]
    fn exact_batch_one_32k_plan_for_twelve_layers() {
        let mut spec = GlmDsaSpec::from_gguf(&set()).expect("valid DSA set");
        spec.dsa_layers = vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45];
        let plan = spec.plan(1, 32 * 1024).expect("valid plan");
        assert_eq!(plan.pooled_entries, 8192);
        assert_eq!(spec.qk_rope_head_dim, 0);
        assert_eq!(plan.mla_cache_bytes, 257_949_696);
        assert_eq!(plan.pooled_index_cache_bytes, 12_976_128);
        assert_eq!(plan.tail_bytes, 24_576);
        assert_eq!(plan.persistent_bytes(), Ok(270_950_400));
        assert_eq!(plan.query_workspace_bytes, 183_296);
        assert_eq!(plan.history_mid_out_bytes, 2_097_152);
        assert_eq!(plan.tail_mid_out_bytes, 131_072);
        assert_eq!(plan.mqa_schedule_bytes, 392);
        assert_eq!(plan.decode_workspace_bytes(), Ok(2_569_116));
    }

    #[test]
    fn rejects_dsa_tensor_shape_drift() {
        let mut invalid = set();
        invalid
            .tensors
            .get_mut("blk.3.indexer.attn_k.weight")
            .expect("tensor")
            .dimensions = vec![4096, 64];
        assert_eq!(
            GlmDsaSpec::from_gguf(&invalid),
            Err(GlmDsaError::InvalidTensor(
                "blk.3.indexer.attn_k.weight".to_owned()
            ))
        );
    }

    #[test]
    fn pooled_topk_expands_groups_and_appends_the_unpooled_tail() {
        let spec = GlmDsaSpec::from_gguf(&set()).expect("valid DSA set");
        let selection = select_pooled_history(&spec, &[0.1, 0.9, 0.4], 3, 14, None, Some(100))
            .expect("selection");
        assert_eq!(
            &selection.raw[..15],
            &[4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 12, 13, -1]
        );
        assert_eq!(
            &selection.mapped[..15],
            &[
                104, 105, 106, 107, 108, 109, 110, 111, 100, 101, 102, 103, 112, 113, -1
            ]
        );
        assert_eq!(selection.raw.len(), 2051);
    }

    #[test]
    fn pooled_topk_uses_one_selection_for_raw_and_page_mapped_outputs() {
        let spec = GlmDsaSpec::from_gguf(&set()).expect("valid DSA set");
        let page_table = (0..10).map(|token| 1000 + token * 7).collect::<Vec<_>>();
        let selection = select_pooled_history(&spec, &[0.3, 0.7], 2, 10, Some(&page_table), None)
            .expect("selection");
        assert_eq!(&selection.raw[..11], &[4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1]);
        assert_eq!(
            &selection.mapped[..11],
            &[
                1028, 1035, 1042, 1049, 1000, 1007, 1014, 1021, 1056, 1063, -1
            ]
        );
    }

    #[test]
    fn glm_reuses_the_existing_framework_free_radix_and_expansion_kernels() {
        let spec = GlmDsaSpec::from_gguf(&set()).expect("valid DSA set");
        let topk = spec.cuda_topk_plan(2, 8192).expect("topk plan");
        let expand = spec.cuda_expand_plan(2).expect("expand plan");
        assert_eq!(topk.topk, 512);
        assert_eq!(topk.columns, 8192);
        assert_eq!(expand.block_topk, 512);
        assert_eq!(expand.compress_ratio, 4);
        assert_eq!(expand.token_topk, 2048);
        assert_eq!(expand.final_topk, 2051);
    }
}

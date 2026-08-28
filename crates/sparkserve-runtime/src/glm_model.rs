//! Locked GLM-5.3-Flash model contract derived from the real GGUF headers.
//!
//! This is intentionally model-specific. SparkServe supports one GLM graph,
//! not an open-ended Transformers interpreter, so every runtime-relevant
//! scalar, per-layer array, tensor family, and the 45-trunk + 1-MTP split is
//! rejected unless it matches the pinned `UD-IQ3_XXS` checkpoint.

use std::fmt::{Display, Formatter};

use crate::ggml_quant::mmvq_geometry;
use crate::gguf::{GgmlTensorType, GgufError, GgufMetadataValue, GgufSet};
use crate::gguf_paging::GgufExpertCatalog;
use crate::glm_dsa::GlmDsaSpec;
use crate::glm_topology::{GlmAttentionKind, GlmTopology, GlmTopologyError};

pub const GLM53_TRUNK_LAYERS: u16 = 45;
pub const GLM53_MTP_LAYERS: u16 = 1;
pub const GLM53_HIDDEN: u64 = 4096;
pub const GLM53_VOCAB: u64 = 154_880;
pub const GLM53_CONTEXT: u64 = 1_048_576;

#[derive(Clone, Debug, PartialEq)]
pub struct Glm53ModelSpec {
    pub topology: GlmTopology,
    pub hidden_size: u64,
    pub vocabulary: u64,
    pub context_length: u64,
    pub dense_intermediate_size: u64,
    pub expert_intermediate_size: u64,
    pub experts: u64,
    pub routed_experts: u64,
    pub shared_experts: u64,
    pub expert_weight_scale: f32,
    pub rms_norm_epsilon: f32,
    pub attention_norm_epsilon: f32,
    pub hyper_connections: u64,
    pub hyper_connection_epsilon: f32,
    pub hyper_connection_sinkhorn_iterations: u64,
    pub kda_head_dim: u64,
    pub kda_gate_lower_bound: f32,
    pub kda_conv_width: u64,
    pub attention_kv_heads: Vec<i32>,
    pub swiglu_clamp_expert: Vec<f32>,
    pub swiglu_clamp_shared: Vec<f32>,
}

impl Glm53ModelSpec {
    pub fn from_gguf(set: &GgufSet) -> Result<Self, Glm53ModelError> {
        if set.architecture != "glm5next" {
            return Err(Glm53ModelError::WrongArchitecture);
        }
        let topology = GlmTopology::from_gguf(set).map_err(Glm53ModelError::Topology)?;
        expect_u64(
            "glm5next.block_count",
            topology.layers().len() as u64,
            u64::from(GLM53_TRUNK_LAYERS + GLM53_MTP_LAYERS),
        )?;
        expect_u64(
            "glm5next trunk layers",
            u64::from(topology.trunk_layer_count()),
            u64::from(GLM53_TRUNK_LAYERS),
        )?;
        expect_u64(
            "glm5next MTP layers",
            u64::from(topology.mtp_layer_count()),
            u64::from(GLM53_MTP_LAYERS),
        )?;

        let hidden_size = locked_u64(set, "glm5next.embedding_length", GLM53_HIDDEN)?;
        let vocabulary = locked_u64(set, "glm5next.vocab_size", GLM53_VOCAB)?;
        let context_length = locked_u64(set, "glm5next.context_length", GLM53_CONTEXT)?;
        let dense_intermediate_size =
            locked_u64(set, "glm5next.feed_forward_length", 12_288)?;
        let expert_intermediate_size = locked_u64(
            set,
            "glm5next.expert_feed_forward_length",
            2_048,
        )?;
        locked_u64(
            set,
            "glm5next.expert_shared_feed_forward_length",
            expert_intermediate_size,
        )?;
        let experts = locked_u64(set, "glm5next.expert_count", 288)?;
        let routed_experts = locked_u64(set, "glm5next.expert_used_count", 8)?;
        let shared_experts = locked_u64(set, "glm5next.expert_shared_count", 1)?;
        locked_u64(set, "glm5next.expert_group_count", 1)?;
        locked_u64(set, "glm5next.expert_group_used_count", 1)?;
        locked_u64(set, "glm5next.expert_gating_func", 2)?;
        locked_bool(set, "glm5next.expert_weights_norm", true)?;
        let expert_weight_scale = locked_f32(set, "glm5next.expert_weights_scale", 2.5)?;
        let rms_norm_epsilon = locked_f32(
            set,
            "glm5next.attention.layer_norm_rms_epsilon",
            1.0e-5,
        )?;
        let attention_norm_epsilon = locked_f32(
            set,
            "glm5next.attention.layer_norm_epsilon",
            1.0e-6,
        )?;
        let hyper_connections =
            locked_u64(set, "glm5next.hyper_connection.count", 4)?;
        let hyper_connection_epsilon = locked_f32(
            set,
            "glm5next.hyper_connection.epsilon",
            1.0e-6,
        )?;
        let hyper_connection_sinkhorn_iterations = locked_u64(
            set,
            "glm5next.hyper_connection.sinkhorn_iterations",
            20,
        )?;
        let kda_head_dim = locked_u64(set, "glm5next.kda.head_dim", 128)?;
        let kda_gate_lower_bound =
            locked_f32(set, "glm5next.kda.gate_lower_bound", -5.0)?;
        let kda_conv_width = locked_u64(set, "glm5next.ssm.conv_kernel", 4)?;
        locked_u64(set, "glm5next.nextn_predict_layers", 1)?;

        let first = set.shards.first().ok_or(Glm53ModelError::NoShard)?;
        let attention_kv_heads = first
            .metadata_i32_array("glm5next.attention.head_count_kv")
            .map_err(Glm53ModelError::Gguf)?;
        let swiglu_clamp_expert = first
            .metadata_f32_array("glm5next.swiglu_clamp_exp")
            .map_err(Glm53ModelError::Gguf)?;
        let swiglu_clamp_shared = first
            .metadata_f32_array("glm5next.swiglu_clamp_shexp")
            .map_err(Glm53ModelError::Gguf)?;
        let blocks = topology.layers().len();
        if attention_kv_heads.len() != blocks
            || swiglu_clamp_expert.len() != blocks
            || swiglu_clamp_shared.len() != blocks
        {
            return Err(Glm53ModelError::LayerArrayLength {
                blocks,
                kv_heads: attention_kv_heads.len(),
                expert_clamps: swiglu_clamp_expert.len(),
                shared_clamps: swiglu_clamp_shared.len(),
            });
        }
        for layer in topology.layers() {
            let expected_kv_heads = match layer.attention {
                GlmAttentionKind::Kda => 0,
                GlmAttentionKind::DsaMla => 1,
            };
            let ordinal = usize::from(layer.layer);
            if attention_kv_heads[ordinal] != expected_kv_heads {
                return Err(Glm53ModelError::LayerKvHeads {
                    layer: layer.layer,
                    actual: attention_kv_heads[ordinal],
                    expected: expected_kv_heads,
                });
            }
            for (role, value) in [
                ("routed expert", swiglu_clamp_expert[ordinal]),
                ("shared expert", swiglu_clamp_shared[ordinal]),
            ] {
                if value.to_bits() != 10.0_f32.to_bits() {
                    return Err(Glm53ModelError::LayerClamp {
                        layer: layer.layer,
                        role,
                        actual: value,
                    });
                }
            }
        }

        validate_tensor_inventory(set, &topology)?;
        GlmDsaSpec::from_gguf(set).map_err(|error| {
            Glm53ModelError::Component("DSA/MLA", error.to_string())
        })?;
        GgufExpertCatalog::build_glm53(set, 16).map_err(|error| {
            Glm53ModelError::Component("expert paging", error.to_string())
        })?;

        Ok(Self {
            topology,
            hidden_size,
            vocabulary,
            context_length,
            dense_intermediate_size,
            expert_intermediate_size,
            experts,
            routed_experts,
            shared_experts,
            expert_weight_scale,
            rms_norm_epsilon,
            attention_norm_epsilon,
            hyper_connections,
            hyper_connection_epsilon,
            hyper_connection_sinkhorn_iterations,
            kda_head_dim,
            kda_gate_lower_bound,
            kda_conv_width,
            attention_kv_heads,
            swiglu_clamp_expert,
            swiglu_clamp_shared,
        })
    }
}

fn validate_tensor_inventory(
    set: &GgufSet,
    topology: &GlmTopology,
) -> Result<(), Glm53ModelError> {
    tensor(set, "token_embd.weight", &[4096, 154_880], GgmlTensorType::Q6K)?;
    tensor(set, "output.weight", &[4096, 154_880], GgmlTensorType::Q6K)?;
    tensor(set, "output_norm.weight", &[4096], GgmlTensorType::F32)?;

    for layer in topology.trunk_layers() {
        let prefix = format!("blk.{}", layer.layer);
        tensor(
            set,
            &format!("{prefix}.attn_norm.weight"),
            &[4096],
            GgmlTensorType::F32,
        )?;
        tensor(
            set,
            &format!("{prefix}.ffn_norm.weight"),
            &[4096],
            GgmlTensorType::F32,
        )?;
        for stage in ["attn", "ffn"] {
            tensor(
                set,
                &format!("{prefix}.hc_{stage}_base.weight"),
                &[24],
                GgmlTensorType::F32,
            )?;
            tensor(
                set,
                &format!("{prefix}.hc_{stage}_scale.weight"),
                &[3],
                GgmlTensorType::F32,
            )?;
            tensor(
                set,
                &format!("{prefix}.hc_{stage}_fn.weight"),
                &[16_384, 24],
                GgmlTensorType::Q8_0,
            )?;
        }
        match layer.attention {
            GlmAttentionKind::Kda => validate_kda_tensors(set, layer.layer)?,
            GlmAttentionKind::DsaMla => {}
        }
    }

    let mtp = topology
        .mtp_layers()
        .first()
        .ok_or(Glm53ModelError::MissingMtp)?;
    let prefix = format!("blk.{}.nextn", mtp.layer);
    tensor(
        set,
        &format!("{prefix}.eh_proj.weight"),
        &[8192, 4096],
        GgmlTensorType::Q8_0,
    )?;
    for name in ["enorm", "hnorm", "shared_head_norm"] {
        tensor(
            set,
            &format!("{prefix}.{name}.weight"),
            &[4096],
            GgmlTensorType::F32,
        )?;
    }
    Ok(())
}

fn validate_kda_tensors(set: &GgufSet, layer: u16) -> Result<(), Glm53ModelError> {
    let prefix = format!("blk.{layer}");
    for name in ["attn_q", "attn_k", "attn_v"] {
        tensor(
            set,
            &format!("{prefix}.{name}.weight"),
            &[4096, 8192],
            GgmlTensorType::Q6K,
        )?;
    }
    tensor(
        set,
        &format!("{prefix}.attn_output.weight"),
        &[8192, 4096],
        GgmlTensorType::Q6K,
    )?;
    tensor(
        set,
        &format!("{prefix}.ssm_a"),
        &[64],
        GgmlTensorType::F32,
    )?;
    tensor(
        set,
        &format!("{prefix}.ssm_beta.weight"),
        &[4096, 64],
        GgmlTensorType::Q8_0,
    )?;
    for name in ["ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"] {
        tensor(
            set,
            &format!("{prefix}.{name}.weight"),
            &[4, 1, 8192],
            GgmlTensorType::F32,
        )?;
    }
    tensor(
        set,
        &format!("{prefix}.ssm_dt.bias"),
        &[8192],
        GgmlTensorType::F32,
    )?;
    for name in ["ssm_f_a", "ssm_g_a"] {
        tensor(
            set,
            &format!("{prefix}.{name}.weight"),
            &[4096, 128],
            GgmlTensorType::Q8_0,
        )?;
    }
    for name in ["ssm_f_b", "ssm_g_b"] {
        tensor(
            set,
            &format!("{prefix}.{name}.weight"),
            &[128, 8192],
            GgmlTensorType::Q8_0,
        )?;
    }
    tensor(
        set,
        &format!("{prefix}.ssm_norm.weight"),
        &[128],
        GgmlTensorType::F32,
    )?;
    Ok(())
}

fn tensor(
    set: &GgufSet,
    name: &str,
    dimensions: &[u64],
    tensor_type: GgmlTensorType,
) -> Result<(), Glm53ModelError> {
    let location = set
        .tensors
        .get(name)
        .ok_or_else(|| Glm53ModelError::MissingTensor(name.to_owned()))?;
    if location.dimensions != dimensions || location.tensor_type != tensor_type {
        return Err(Glm53ModelError::TensorContract {
            name: name.to_owned(),
            dimensions: location.dimensions.clone(),
            tensor_type: location.tensor_type,
            expected_dimensions: dimensions.to_vec(),
            expected_type: tensor_type,
        });
    }
    if tensor_type != GgmlTensorType::F32 && mmvq_geometry(tensor_type).is_none() {
        return Err(Glm53ModelError::UnsupportedQuantType {
            name: name.to_owned(),
            tensor_type,
        });
    }
    Ok(())
}

fn metadata<'a>(set: &'a GgufSet, key: &str) -> Result<&'a GgufMetadataValue, Glm53ModelError> {
    set.shards
        .first()
        .and_then(|shard| shard.metadata(key))
        .ok_or_else(|| Glm53ModelError::MissingMetadata(key.to_owned()))
}

fn locked_u64(set: &GgufSet, key: &'static str, expected: u64) -> Result<u64, Glm53ModelError> {
    let actual = metadata(set, key)?
        .as_u64()
        .ok_or_else(|| Glm53ModelError::MetadataType(key.to_owned()))?;
    expect_u64(key, actual, expected)?;
    Ok(actual)
}

fn locked_f32(set: &GgufSet, key: &'static str, expected: f32) -> Result<f32, Glm53ModelError> {
    let GgufMetadataValue::Float(value) = metadata(set, key)? else {
        return Err(Glm53ModelError::MetadataType(key.to_owned()));
    };
    let actual = *value as f32;
    if actual.to_bits() != expected.to_bits() {
        return Err(Glm53ModelError::UnexpectedFloat {
            key,
            actual,
            expected,
        });
    }
    Ok(actual)
}

fn locked_bool(set: &GgufSet, key: &'static str, expected: bool) -> Result<bool, Glm53ModelError> {
    let GgufMetadataValue::Bool(actual) = metadata(set, key)? else {
        return Err(Glm53ModelError::MetadataType(key.to_owned()));
    };
    if *actual != expected {
        return Err(Glm53ModelError::UnexpectedBool {
            key,
            actual: *actual,
            expected,
        });
    }
    Ok(*actual)
}

fn expect_u64(key: &'static str, actual: u64, expected: u64) -> Result<(), Glm53ModelError> {
    if actual != expected {
        Err(Glm53ModelError::UnexpectedInteger {
            key,
            actual,
            expected,
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Glm53ModelError {
    WrongArchitecture,
    NoShard,
    MissingMetadata(String),
    MetadataType(String),
    UnexpectedInteger {
        key: &'static str,
        actual: u64,
        expected: u64,
    },
    UnexpectedFloat {
        key: &'static str,
        actual: f32,
        expected: f32,
    },
    UnexpectedBool {
        key: &'static str,
        actual: bool,
        expected: bool,
    },
    LayerArrayLength {
        blocks: usize,
        kv_heads: usize,
        expert_clamps: usize,
        shared_clamps: usize,
    },
    LayerKvHeads {
        layer: u16,
        actual: i32,
        expected: i32,
    },
    LayerClamp {
        layer: u16,
        role: &'static str,
        actual: f32,
    },
    MissingTensor(String),
    TensorContract {
        name: String,
        dimensions: Vec<u64>,
        tensor_type: GgmlTensorType,
        expected_dimensions: Vec<u64>,
        expected_type: GgmlTensorType,
    },
    UnsupportedQuantType {
        name: String,
        tensor_type: GgmlTensorType,
    },
    MissingMtp,
    Topology(GlmTopologyError),
    Gguf(GgufError),
    Component(&'static str, String),
}

impl Display for Glm53ModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongArchitecture => formatter.write_str("GGUF architecture is not glm5next"),
            Self::NoShard => formatter.write_str("GLM GGUF has no shard"),
            Self::MissingMetadata(key) => write!(formatter, "missing GLM metadata {key}"),
            Self::MetadataType(key) => write!(formatter, "GLM metadata {key} has the wrong type"),
            Self::UnexpectedInteger { key, actual, expected } => write!(
                formatter,
                "GLM metadata {key} is {actual}, expected {expected}"
            ),
            Self::UnexpectedFloat { key, actual, expected } => write!(
                formatter,
                "GLM metadata {key} is {actual}, expected {expected}"
            ),
            Self::UnexpectedBool { key, actual, expected } => write!(
                formatter,
                "GLM metadata {key} is {actual}, expected {expected}"
            ),
            Self::LayerArrayLength {
                blocks,
                kv_heads,
                expert_clamps,
                shared_clamps,
            } => write!(
                formatter,
                "GLM layer arrays disagree: {blocks} blocks, {kv_heads} KV entries, {expert_clamps} expert clamps, {shared_clamps} shared clamps"
            ),
            Self::LayerKvHeads { layer, actual, expected } => write!(
                formatter,
                "GLM layer {layer} has {actual} KV heads, expected {expected}"
            ),
            Self::LayerClamp { layer, role, actual } => write!(
                formatter,
                "GLM layer {layer} {role} SwiGLU clamp is {actual}, expected 10"
            ),
            Self::MissingTensor(name) => write!(formatter, "missing GLM tensor {name}"),
            Self::TensorContract {
                name,
                dimensions,
                tensor_type,
                expected_dimensions,
                expected_type,
            } => write!(
                formatter,
                "GLM tensor {name} is {dimensions:?} {tensor_type:?}, expected {expected_dimensions:?} {expected_type:?}"
            ),
            Self::UnsupportedQuantType { name, tensor_type } => write!(
                formatter,
                "GLM tensor {name} uses unsupported runtime quant type {tensor_type:?}"
            ),
            Self::MissingMtp => formatter.write_str("GLM checkpoint has no MTP block"),
            Self::Topology(error) => write!(formatter, "invalid GLM topology: {error}"),
            Self::Gguf(error) => write!(formatter, "cannot decode GLM metadata: {error}"),
            Self::Component(name, error) => write!(formatter, "invalid GLM {name}: {error}"),
        }
    }
}

impl std::error::Error for Glm53ModelError {}

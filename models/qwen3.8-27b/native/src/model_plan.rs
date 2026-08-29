use crate::mapping::{MappedCheckpoint, MappingError};
use crate::scale_sidecar::{ScaleSidecar, SidecarError};
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::path::Path;

pub const MODEL_ABI_VERSION: u32 = 1;
pub const MODEL_LAYERS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelLayerWeights {
    pub input_norm_bf16: *const c_void,
    pub post_attention_norm_bf16: *const c_void,

    pub mlp_gate_weight_fp4: *const c_void,
    pub mlp_gate_scales_fp8_128x4: *const c_void,
    pub mlp_gate_alpha: *const f32,
    pub mlp_hidden_scale_inv: *const f32,
    pub mlp_up_weight_fp4: *const c_void,
    pub mlp_up_scales_fp8_128x4: *const c_void,
    pub mlp_up_alpha: *const f32,
    pub mlp_down_weight_fp4: *const c_void,
    pub mlp_down_scales_fp8_128x4: *const c_void,
    pub mlp_down_alpha: *const f32,
    pub mlp_activated_scale_inv: *const f32,

    pub gdn_qkv_weight_fp8: *const c_void,
    pub gdn_qkv_input_scale: *const f32,
    pub gdn_qkv_weight_scale: *const f32,
    pub gdn_z_weight_fp8: *const c_void,
    pub gdn_z_input_scale: *const f32,
    pub gdn_z_weight_scale: *const f32,
    pub gdn_a_weight_bf16: *const c_void,
    pub gdn_b_weight_bf16: *const c_void,
    pub gdn_conv_weight_bf16: *const c_void,
    pub gdn_norm_weight_bf16: *const c_void,
    pub gdn_a_log_bf16: *const c_void,
    pub gdn_dt_bias_bf16: *const c_void,
    pub gdn_out_weight_fp8: *const c_void,
    pub gdn_out_input_scale: *const f32,
    pub gdn_out_weight_scale: *const f32,

    pub attention_q_weight_fp8: *const c_void,
    pub attention_q_input_scale: *const f32,
    pub attention_q_weight_scale: *const f32,
    pub attention_k_weight_fp8: *const c_void,
    pub attention_k_input_scale: *const f32,
    pub attention_k_weight_scale: *const f32,
    pub attention_v_weight_fp8: *const c_void,
    pub attention_v_input_scale: *const f32,
    pub attention_v_weight_scale: *const f32,
    pub attention_o_weight_fp8: *const c_void,
    pub attention_o_input_scale: *const f32,
    pub attention_o_weight_scale: *const f32,
    pub attention_q_norm_bf16: *const c_void,
    pub attention_k_norm_bf16: *const c_void,
}

impl ModelLayerWeights {
    fn empty() -> Self {
        // SAFETY: this C aggregate contains only pointer fields, and null is
        // the required representation for the inactive model-specific branch.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
pub struct ModelWeights {
    pub struct_size: u32,
    pub abi_version: u32,
    pub embedding_bf16: *const c_void,
    pub final_norm_bf16: *const c_void,
    pub lm_head_bf16: *const c_void,
    pub layers: *const ModelLayerWeights,
    pub layer_count: u32,
    pub reserved: u32,
}

#[derive(Debug)]
pub enum ModelPlanError {
    Mapping(MappingError),
    Sidecar(SidecarError),
    Invalid(String),
}

impl Display for ModelPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mapping(error) => write!(formatter, "{error}"),
            Self::Sidecar(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ModelPlanError {}
impl From<MappingError> for ModelPlanError {
    fn from(error: MappingError) -> Self { Self::Mapping(error) }
}
impl From<SidecarError> for ModelPlanError {
    fn from(error: SidecarError) -> Self { Self::Sidecar(error) }
}

pub struct EagerWeightPlan {
    weights: ModelWeights,
    layers: Vec<ModelLayerWeights>,
    _checkpoint: MappedCheckpoint,
    _sidecar: ScaleSidecar,
}

impl EagerWeightPlan {
    pub fn open(checkpoint_root: &Path, sidecar_path: &Path) -> Result<Self, ModelPlanError> {
        let checkpoint = MappedCheckpoint::open(checkpoint_root)?;
        let sidecar = ScaleSidecar::open(sidecar_path)?;
        let mut layers = Vec::with_capacity(MODEL_LAYERS);
        for layer in 0..MODEL_LAYERS as u32 {
            let prefix = format!("model.language_model.layers.{layer}");
            let mut weights = ModelLayerWeights::empty();
            weights.input_norm_bf16 = tensor(&checkpoint, &format!("{prefix}.input_layernorm.weight"))?;
            weights.post_attention_norm_bf16 = tensor(&checkpoint, &format!("{prefix}.post_attention_layernorm.weight"))?;

            let mlp = format!("{prefix}.mlp");
            weights.mlp_gate_weight_fp4 = tensor(&checkpoint, &format!("{mlp}.gate_proj.weight"))?;
            weights.mlp_gate_scales_fp8_128x4 = sidecar.scale_device_address(layer, 0)? as *const c_void;
            weights.mlp_gate_alpha = sidecar.alpha_device_address(layer, 0)? as *const f32;
            weights.mlp_hidden_scale_inv = sidecar.input_scale_inv_device_address(layer, 0)? as *const f32;
            weights.mlp_up_weight_fp4 = tensor(&checkpoint, &format!("{mlp}.up_proj.weight"))?;
            weights.mlp_up_scales_fp8_128x4 = sidecar.scale_device_address(layer, 1)? as *const c_void;
            weights.mlp_up_alpha = sidecar.alpha_device_address(layer, 1)? as *const f32;
            weights.mlp_down_weight_fp4 = tensor(&checkpoint, &format!("{mlp}.down_proj.weight"))?;
            weights.mlp_down_scales_fp8_128x4 = sidecar.scale_device_address(layer, 2)? as *const c_void;
            weights.mlp_down_alpha = sidecar.alpha_device_address(layer, 2)? as *const f32;
            weights.mlp_activated_scale_inv = sidecar.input_scale_inv_device_address(layer, 2)? as *const f32;

            if (layer + 1) % 4 == 0 {
                let attention = format!("{prefix}.self_attn");
                fill_fp8_projection(
                    &checkpoint,
                    &format!("{attention}.q_proj"),
                    &mut weights.attention_q_weight_fp8,
                    &mut weights.attention_q_input_scale,
                    &mut weights.attention_q_weight_scale,
                )?;
                fill_fp8_projection(
                    &checkpoint,
                    &format!("{attention}.k_proj"),
                    &mut weights.attention_k_weight_fp8,
                    &mut weights.attention_k_input_scale,
                    &mut weights.attention_k_weight_scale,
                )?;
                fill_fp8_projection(
                    &checkpoint,
                    &format!("{attention}.v_proj"),
                    &mut weights.attention_v_weight_fp8,
                    &mut weights.attention_v_input_scale,
                    &mut weights.attention_v_weight_scale,
                )?;
                fill_fp8_projection(
                    &checkpoint,
                    &format!("{attention}.o_proj"),
                    &mut weights.attention_o_weight_fp8,
                    &mut weights.attention_o_input_scale,
                    &mut weights.attention_o_weight_scale,
                )?;
                weights.attention_q_norm_bf16 = tensor(&checkpoint, &format!("{attention}.q_norm.weight"))?;
                weights.attention_k_norm_bf16 = tensor(&checkpoint, &format!("{attention}.k_norm.weight"))?;
            } else {
                let gdn = format!("{prefix}.linear_attn");
                fill_fp8_projection(
                    &checkpoint,
                    &format!("{gdn}.in_proj_qkv"),
                    &mut weights.gdn_qkv_weight_fp8,
                    &mut weights.gdn_qkv_input_scale,
                    &mut weights.gdn_qkv_weight_scale,
                )?;
                fill_fp8_projection(
                    &checkpoint,
                    &format!("{gdn}.in_proj_z"),
                    &mut weights.gdn_z_weight_fp8,
                    &mut weights.gdn_z_input_scale,
                    &mut weights.gdn_z_weight_scale,
                )?;
                weights.gdn_a_weight_bf16 = tensor(&checkpoint, &format!("{gdn}.in_proj_a.weight"))?;
                weights.gdn_b_weight_bf16 = tensor(&checkpoint, &format!("{gdn}.in_proj_b.weight"))?;
                weights.gdn_conv_weight_bf16 = tensor(&checkpoint, &format!("{gdn}.conv1d.weight"))?;
                weights.gdn_norm_weight_bf16 = tensor(&checkpoint, &format!("{gdn}.norm.weight"))?;
                weights.gdn_a_log_bf16 = tensor(&checkpoint, &format!("{gdn}.A_log"))?;
                weights.gdn_dt_bias_bf16 = tensor(&checkpoint, &format!("{gdn}.dt_bias"))?;
                fill_fp8_projection(
                    &checkpoint,
                    &format!("{gdn}.out_proj"),
                    &mut weights.gdn_out_weight_fp8,
                    &mut weights.gdn_out_input_scale,
                    &mut weights.gdn_out_weight_scale,
                )?;
            }
            layers.push(weights);
        }
        if layers.len() != MODEL_LAYERS {
            return Err(ModelPlanError::Invalid("q27 layer plan is incomplete".into()));
        }
        let weights = ModelWeights {
            struct_size: size_of::<ModelWeights>() as u32,
            abi_version: MODEL_ABI_VERSION,
            embedding_bf16: tensor(&checkpoint, "model.language_model.embed_tokens.weight")?,
            final_norm_bf16: tensor(&checkpoint, "model.language_model.norm.weight")?,
            lm_head_bf16: tensor(&checkpoint, "lm_head.weight")?,
            layers: layers.as_ptr(),
            layer_count: MODEL_LAYERS as u32,
            reserved: 0,
        };
        Ok(Self { weights, layers, _checkpoint: checkpoint, _sidecar: sidecar })
    }

    pub fn weights(&self) -> &ModelWeights {
        debug_assert_eq!(self.weights.layers, self.layers.as_ptr());
        &self.weights
    }
}

fn tensor(mapped: &MappedCheckpoint, name: &str) -> Result<*const c_void, ModelPlanError> {
    let location = mapped.checkpoint().tensor(name)
        .map_err(|error| ModelPlanError::Invalid(error.to_string()))?;
    Ok(mapped.device_address(location)? as *const c_void)
}

fn fill_fp8_projection(
    mapped: &MappedCheckpoint,
    prefix: &str,
    weight: &mut *const c_void,
    input_scale: &mut *const f32,
    weight_scale: &mut *const f32,
) -> Result<(), ModelPlanError> {
    *weight = tensor(mapped, &format!("{prefix}.weight"))?;
    *input_scale = tensor(mapped, &format!("{prefix}.input_scale"))? as *const f32;
    *weight_scale = tensor(mapped, &format!("{prefix}.weight_scale"))? as *const f32;
    Ok(())
}

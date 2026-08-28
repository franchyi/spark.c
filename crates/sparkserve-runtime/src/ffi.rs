use std::ffi::{c_char, c_void};

use crate::gguf::GgmlTensorType;
use crate::kernel::{
    DataType, DenseNvfp4Spec, GroupedNvfp4Spec, KERNEL_ABI_VERSION, SegmentedNvfp4QuantizeSpec,
    SegmentedSiluNvfp4Spec, SiluNvfp4Spec,
};
use crate::routing::MoeRouteSpec;

pub const FABRIC_ABI_VERSION: u32 = 1;
pub const GGML_QUANT_ABI_VERSION: u32 = 1;
pub const GLM_DSA_ABI_VERSION: u32 = 1;
pub const GLM_KDA_ABI_VERSION: u32 = 1;
pub const GLM_MQA_ABI_VERSION: u32 = 1;
pub const GLM_MQA_GB10_SMS: u32 = 48;
pub const GLM_MQA_SCHEDULE_WORDS: u32 = 98;
pub const GLM_SPARSE_MLA_ABI_VERSION: u32 = 1;
pub const GLM_SPARSE_MLA_GB10_SMS: u32 = 48;
pub const GLM_SPARSE_MLA_PAGE_SIZE: u32 = 64;
pub const GLM_SPARSE_MLA_HEADS: u32 = 64;
pub const GLM_SPARSE_MLA_LATENT_DIM: u32 = 512;
pub const GLM_SPARSE_MLA_PADDED_Q_DIM: u32 = 576;
pub const GLM_SPARSE_MLA_TOKEN_BYTES: u32 = 656;
pub const GLM_SPARSE_MLA_HISTORY_TOPK: u32 = 2048;
pub const GLM_SPARSE_MLA_TAIL_TOPK: u32 = 128;
pub const GLM_SPARSE_MLA_HISTORY_SPLITS: u32 = 32;
pub const GLM_SPARSE_MLA_TAIL_SPLITS: u32 = 2;
pub const GLM_SPARSE_MLA_SELECTION_WIDTH: u32 = 2051;
pub const QWEN_EXPERT_PACK_ABI_VERSION: u32 = 1;
pub const QWEN_GDN_AUX_ABI_VERSION: u32 = 1;
pub const QWEN_QSA_BLOCK_ABI_VERSION: u32 = 1;
pub const QWEN_PLE_BLOCK_ABI_VERSION: u32 = 1;
pub const QWEN_DECODE_GLUE_ABI_VERSION: u32 = 1;
pub const QWEN_EXPERT_CAPACITY: u32 = 16;
pub const QWEN_W13_WEIGHT_BYTES: u64 = 1_638_400;
pub const QWEN_W2_WEIGHT_BYTES: u64 = 819_200;
pub const QWEN_W13_SCALE_BYTES: u64 = 204_800;
pub const QWEN_W2_SCALE_BYTES: u64 = 102_400;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Status {
    pub code: i32,
    pub message: *const c_char,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoherentRegionKind {
    Slab = 1,
    FileReadOnly = 2,
}

pub const COHERENT_REGION_PREFAULT: u32 = 1 << 0;
pub const COHERENT_REGION_HUGE_PAGE_HINT: u32 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoherentRegionConfig {
    pub struct_size: u32,
    pub abi_version: u32,
    pub kind: u32,
    pub flags: u32,
    pub payload_bytes: u64,
    pub file_offset: u64,
    pub required_alignment: u64,
    pub file_path: *const c_char,
}

impl CoherentRegionConfig {
    pub fn slab(payload_bytes: u64, required_alignment: u64, flags: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: FABRIC_ABI_VERSION,
            kind: CoherentRegionKind::Slab as u32,
            flags,
            payload_bytes,
            file_offset: 0,
            required_alignment,
            file_path: std::ptr::null(),
        }
    }

    pub fn file_read_only(
        payload_bytes: u64,
        file_offset: u64,
        required_alignment: u64,
        flags: u32,
        file_path: *const c_char,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: FABRIC_ABI_VERSION,
            kind: CoherentRegionKind::FileReadOnly as u32,
            flags,
            payload_bytes,
            file_offset,
            required_alignment,
            file_path,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoherentRegionView {
    pub struct_size: u32,
    pub abi_version: u32,
    pub kind: u32,
    pub flags: u32,
    pub host_pointer: *mut c_void,
    pub device_pointer: *mut c_void,
    pub mapped_bytes: u64,
    pub payload_bytes: u64,
    pub file_offset: u64,
    pub required_alignment: u64,
    pub page_bytes: u64,
    pub device_id: i32,
    pub reserved: u32,
}

impl CoherentRegionView {
    pub fn empty() -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: FABRIC_ABI_VERSION,
            kind: 0,
            flags: 0,
            host_pointer: std::ptr::null_mut(),
            device_pointer: std::ptr::null_mut(),
            mapped_bytes: 0,
            payload_bytes: 0,
            file_offset: 0,
            required_alignment: 0,
            page_bytes: 0,
            device_id: 0,
            reserved: 0,
        }
    }
}

#[repr(C)]
pub struct CoherentRegion {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CudaStream {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CudaEvent {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CudaBlas {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QwenExpertPackArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub fills: u32,
    pub capacity: u32,
    pub destination_slots: *const u32,
    pub gate_weights: *const *const u8,
    pub up_weights: *const *const u8,
    pub down_weights: *const *const u8,
    pub gate_weight_scales: *const *const u8,
    pub up_weight_scales: *const *const u8,
    pub down_weight_scales: *const *const u8,
    pub gate_input_scales: *const *const f32,
    pub gate_weight_scale_2: *const *const f32,
    pub down_input_scales: *const *const f32,
    pub down_weight_scale_2: *const *const f32,
    pub w13_weights: *mut u8,
    pub w2_weights: *mut u8,
    pub w13_scales: *mut u8,
    pub w2_scales: *mut u8,
    pub w13_input_global_scales: *mut f32,
    pub w13_alpha: *mut f32,
    pub w2_input_global_scales: *mut f32,
    pub w2_alpha: *mut f32,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QwenBf16ToF32Args {
    pub struct_size: u32,
    pub abi_version: u32,
    pub input_bf16: *const u16,
    pub output_f32: *mut f32,
    pub elements: u64,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QwenQsaProjectArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub tokens: u32,
    pub rotary_dim: u32,
    pub cos_sin_stride: u64,
    pub hidden_states: *const c_void,
    pub q_weight: *const c_void,
    pub k_weight: *const c_void,
    pub v_weight: *const c_void,
    pub index_qk_weight: *const c_void,
    pub q_norm_weight: *const c_void,
    pub k_norm_weight: *const c_void,
    pub cos_sin_cache: *const f32,
    pub positions: *const i64,
    pub projected_q: *mut c_void,
    pub projected_k: *mut c_void,
    pub query: *mut c_void,
    pub key: *mut c_void,
    pub value: *mut c_void,
    pub gate: *mut c_void,
    pub index_qk: *mut c_void,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QwenQsaFinishArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub tokens: u32,
    pub reserved: u32,
    pub attention_output: *const c_void,
    pub gate: *const c_void,
    pub out_weight: *const c_void,
    pub gated_output: *mut c_void,
    pub output: *mut c_void,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QwenPleBlockArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub tokens: u32,
    pub reserved: u32,
    pub hidden_states: *const c_void,
    pub embedding: *const c_void,
    pub key_weight: *const c_void,
    pub value_weight: *const c_void,
    pub norm_key_weight: *const c_void,
    pub norm_query_weight: *const c_void,
    pub norm_conv_weight: *const c_void,
    pub conv_weight: *const c_void,
    pub conv_state: *mut c_void,
    pub key_scratch: *mut c_void,
    pub value_scratch: *mut c_void,
    pub gated_scratch: *mut c_void,
    pub normed_scratch: *mut c_void,
    pub output: *mut c_void,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QwenDecodeGlueArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub input: *const c_void,
    pub output: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QwenLmHeadArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub vocabulary: u32,
    pub hidden_size: u32,
    pub hidden_states: *const c_void,
    pub weight: *const c_void,
    pub logits: *mut f32,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceCaps {
    pub struct_size: u32,
    pub abi_version: u32,
    pub sm: u32,
    pub supports_fp4_tensor_cores: u32,
    pub workspace_limit_bytes: u64,
}

impl DeviceCaps {
    pub fn gb10(workspace_limit_bytes: u64) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            sm: 121,
            supports_fp4_tensor_cores: 1,
            workspace_limit_bytes,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GgmlQuantDenseArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub quant_type: u32,
    pub input: *const f32,
    pub weights: *const c_void,
    pub output: *mut f32,
    pub q8_scratch: *mut c_void,
    pub q8_scratch_bytes: u64,
    pub vectors: u64,
    pub rows: u64,
    pub k: u64,
    pub cuda_stream: *mut c_void,
}

impl GgmlQuantDenseArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        quant_type: GgmlTensorType,
        input: *const f32,
        weights: *const c_void,
        output: *mut f32,
        q8_scratch: *mut c_void,
        q8_scratch_bytes: u64,
        vectors: u64,
        rows: u64,
        k: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GGML_QUANT_ABI_VERSION,
            quant_type: quant_type as u32,
            input,
            weights,
            output,
            q8_scratch,
            q8_scratch_bytes,
            vectors,
            rows,
            k,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GgmlQuantRoutedArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub quant_type: u32,
    pub input: *const f32,
    pub weights: *const c_void,
    pub expert_ids: *const i32,
    pub output: *mut f32,
    pub q8_scratch: *mut c_void,
    pub q8_scratch_bytes: u64,
    pub tokens: u64,
    pub top_k: u64,
    pub experts: u64,
    pub rows: u64,
    pub k: u64,
    pub weight_slot_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

impl GgmlQuantRoutedArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        quant_type: GgmlTensorType,
        input: *const f32,
        weights: *const c_void,
        expert_ids: *const i32,
        output: *mut f32,
        q8_scratch: *mut c_void,
        q8_scratch_bytes: u64,
        tokens: u64,
        top_k: u64,
        experts: u64,
        rows: u64,
        k: u64,
        weight_slot_stride_bytes: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GGML_QUANT_ABI_VERSION,
            quant_type: quant_type as u32,
            input,
            weights,
            expert_ids,
            output,
            q8_scratch,
            q8_scratch_bytes,
            tokens,
            top_k,
            experts,
            rows,
            k,
            weight_slot_stride_bytes,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmKdaArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub head_dim: u32,
    pub heads: u32,
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub log_decay: *const f32,
    pub beta: *const f32,
    pub state_input: *const f32,
    pub output: *mut f32,
    pub state_output: *mut f32,
    pub tokens: u64,
    pub sequences: u64,
    pub cuda_stream: *mut c_void,
}

impl GlmKdaArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        head_dim: u32,
        heads: u32,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        log_decay: *const f32,
        beta: *const f32,
        state_input: *const f32,
        output: *mut f32,
        state_output: *mut f32,
        tokens: u64,
        sequences: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_KDA_ABI_VERSION,
            head_dim,
            heads,
            q,
            k,
            v,
            log_decay,
            beta,
            state_input,
            output,
            state_output,
            tokens,
            sequences,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmKdaConvArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub channels: u32,
    pub kernel_width: u32,
    pub projected: *const f32,
    pub weight: *const f32,
    pub state_input: *const f32,
    pub output: *mut f32,
    pub state_output: *mut f32,
    pub tokens: u64,
    pub sequences: u64,
    pub cuda_stream: *mut c_void,
}

impl GlmKdaConvArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: u32,
        projected: *const f32,
        weight: *const f32,
        state_input: *const f32,
        output: *mut f32,
        state_output: *mut f32,
        tokens: u64,
        sequences: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_KDA_ABI_VERSION,
            channels,
            kernel_width: 4,
            projected,
            weight,
            state_input,
            output,
            state_output,
            tokens,
            sequences,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmKdaPrepareArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub head_dim: u32,
    pub heads: u32,
    pub q: *const f32,
    pub k: *const f32,
    pub dt: *const f32,
    pub beta_logits: *const f32,
    pub a: *const f32,
    pub dt_bias: *const f32,
    pub normalized_q: *mut f32,
    pub normalized_k: *mut f32,
    pub log_decay: *mut f32,
    pub beta: *mut f32,
    pub l2_epsilon: f32,
    pub reserved: u32,
    pub tokens: u64,
    pub sequences: u64,
    pub cuda_stream: *mut c_void,
}

impl GlmKdaPrepareArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        head_dim: u32,
        heads: u32,
        q: *const f32,
        k: *const f32,
        dt: *const f32,
        beta_logits: *const f32,
        a: *const f32,
        dt_bias: *const f32,
        normalized_q: *mut f32,
        normalized_k: *mut f32,
        log_decay: *mut f32,
        beta: *mut f32,
        l2_epsilon: f32,
        tokens: u64,
        sequences: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_KDA_ABI_VERSION,
            head_dim,
            heads,
            q,
            k,
            dt,
            beta_logits,
            a,
            dt_bias,
            normalized_q,
            normalized_k,
            log_decay,
            beta,
            l2_epsilon,
            reserved: 0,
            tokens,
            sequences,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmKdaGateArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub head_dim: u32,
    pub heads: u32,
    pub input: *const f32,
    pub gate: *const f32,
    pub norm_weight: *const f32,
    pub output: *mut f32,
    pub rms_epsilon: f32,
    pub reserved: u32,
    pub tokens: u64,
    pub sequences: u64,
    pub cuda_stream: *mut c_void,
}

impl GlmKdaGateArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        head_dim: u32,
        heads: u32,
        input: *const f32,
        gate: *const f32,
        norm_weight: *const f32,
        output: *mut f32,
        rms_epsilon: f32,
        tokens: u64,
        sequences: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_KDA_ABI_VERSION,
            head_dim,
            heads,
            input,
            gate,
            norm_weight,
            output,
            rms_epsilon,
            reserved: 0,
            tokens,
            sequences,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmKPoolCompressArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub rows: u32,
    pub pool_size: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub round_scale: u32,
    pub reserved: u32,
    pub slot_key_bf16: *const u16,
    pub slot_score_bf16: *const u16,
    pub ape: *const f32,
    pub locations: *const i64,
    pub key_cache_fp8: *mut u8,
    pub scale_cache: *mut f32,
    pub key_page_stride_bytes: u64,
    pub scale_page_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

impl GlmKPoolCompressArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        slot_key_bf16: *const u16,
        slot_score_bf16: *const u16,
        ape: *const f32,
        locations: *const i64,
        key_cache_fp8: *mut u8,
        scale_cache: *mut f32,
        key_page_stride_bytes: u64,
        scale_page_stride_bytes: u64,
        round_scale: bool,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_DSA_ABI_VERSION,
            rows,
            pool_size: 4,
            head_dim: 128,
            page_size: 64,
            round_scale: u32::from(round_scale),
            reserved: 0,
            slot_key_bf16,
            slot_score_bf16,
            ape,
            locations,
            key_cache_fp8,
            scale_cache,
            key_page_stride_bytes,
            scale_page_stride_bytes,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmKPoolDecodeArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub rows: u32,
    pub request_capacity: u32,
    pub tail_size: u32,
    pub head_dim: u32,
    pub pool_size: u32,
    pub page_size: u32,
    pub slots_per_page: u32,
    pub round_scale: u32,
    pub tail_key_bf16: *mut u16,
    pub tail_score_bf16: *mut u16,
    pub key_bf16: *const u16,
    pub score_bf16: *const u16,
    pub ape: *const f32,
    pub block_tables: *const i32,
    pub request_indices: *const i32,
    pub positions: *const i64,
    pub sequence_lengths: *const i32,
    pub output_cache_locations: *const i64,
    pub key_cache_fp8: *mut u8,
    pub scale_cache: *mut f32,
    pub block_table_stride: u64,
    pub key_page_stride_bytes: u64,
    pub scale_page_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmIndexerPrepArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub query_fp32: *const f32,
    pub key_fp32: *const f32,
    pub key_norm_weight: *const f32,
    pub key_norm_bias: *const f32,
    pub head_gate_fp32: *const f32,
    pub query_fp8: *mut u8,
    pub query_scale: *mut f32,
    pub key_bf16: *mut u16,
    pub logit_weights: *mut f32,
    pub tokens: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub layer_norm_epsilon: f32,
    pub round_scale: u32,
    pub reserved: u32,
    pub cuda_stream: *mut c_void,
}

impl GlmIndexerPrepArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_fp32: *const f32,
        key_fp32: *const f32,
        key_norm_weight: *const f32,
        key_norm_bias: *const f32,
        head_gate_fp32: *const f32,
        query_fp8: *mut u8,
        query_scale: *mut f32,
        key_bf16: *mut u16,
        logit_weights: *mut f32,
        tokens: u32,
        layer_norm_epsilon: f32,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_DSA_ABI_VERSION,
            query_fp32,
            key_fp32,
            key_norm_weight,
            key_norm_bias,
            head_gate_fp32,
            query_fp8,
            query_scale,
            key_bf16,
            logit_weights,
            tokens,
            heads: 32,
            head_dim: 128,
            layer_norm_epsilon,
            round_scale: 1,
            reserved: 0,
            cuda_stream,
        }
    }
}

impl GlmKPoolDecodeArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u32,
        request_capacity: u32,
        tail_key_bf16: *mut u16,
        tail_score_bf16: *mut u16,
        key_bf16: *const u16,
        score_bf16: *const u16,
        ape: *const f32,
        block_tables: *const i32,
        request_indices: *const i32,
        positions: *const i64,
        sequence_lengths: *const i32,
        output_cache_locations: *const i64,
        key_cache_fp8: *mut u8,
        scale_cache: *mut f32,
        block_table_stride: u64,
        key_page_stride_bytes: u64,
        scale_page_stride_bytes: u64,
        round_scale: bool,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_DSA_ABI_VERSION,
            rows,
            request_capacity,
            tail_size: 4,
            head_dim: 128,
            pool_size: 4,
            page_size: 64,
            slots_per_page: 64,
            round_scale: u32::from(round_scale),
            tail_key_bf16,
            tail_score_bf16,
            key_bf16,
            score_bf16,
            ape,
            block_tables,
            request_indices,
            positions,
            sequence_lengths,
            output_cache_locations,
            key_cache_fp8,
            scale_cache,
            block_table_stride,
            key_page_stride_bytes,
            scale_page_stride_bytes,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmPagedMqaArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub batch_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub num_pages: u32,
    pub num_sms: u32,
    pub max_context_len: u32,
    pub logits_stride: u32,
    pub block_table_stride: u32,
    pub reserved: u32,
    pub query_fp8: *const u8,
    pub key_cache_fp8: *const u8,
    pub scale_cache: *const f32,
    pub logit_weights: *const f32,
    pub context_lens: *const u32,
    pub logits: *mut f32,
    pub block_tables: *const u32,
    pub schedule_metadata: *mut u32,
    pub key_page_stride_bytes: u64,
    pub scale_page_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

impl GlmPagedMqaArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn gb10_decode(
        batch_size: u32,
        num_pages: u32,
        max_context_len: u32,
        logits_stride: u32,
        block_table_stride: u32,
        query_fp8: *const u8,
        key_cache_fp8: *const u8,
        scale_cache: *const f32,
        logit_weights: *const f32,
        context_lens: *const u32,
        logits: *mut f32,
        block_tables: *const u32,
        schedule_metadata: *mut u32,
        key_page_stride_bytes: u64,
        scale_page_stride_bytes: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_MQA_ABI_VERSION,
            batch_size,
            num_heads: 32,
            head_dim: 128,
            page_size: 64,
            num_pages,
            num_sms: GLM_MQA_GB10_SMS,
            max_context_len,
            logits_stride,
            block_table_stride,
            reserved: 0,
            query_fp8,
            key_cache_fp8,
            scale_cache,
            logit_weights,
            context_lens,
            logits,
            block_tables,
            schedule_metadata,
            key_page_stride_bytes,
            scale_page_stride_bytes,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmSparseMlaPackKvArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub tokens: u32,
    pub page_size: u32,
    pub latent_dim: u32,
    pub quant_group: u32,
    pub num_pages: u32,
    pub reserved: u32,
    pub input_bf16: *const u16,
    pub locations: *const i32,
    pub cache: *mut u8,
    pub page_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

impl GlmSparseMlaPackKvArgs {
    pub fn no_rope(
        tokens: u32,
        num_pages: u32,
        input_bf16: *const u16,
        locations: *const i32,
        cache: *mut u8,
        page_stride_bytes: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_SPARSE_MLA_ABI_VERSION,
            tokens,
            page_size: GLM_SPARSE_MLA_PAGE_SIZE,
            latent_dim: GLM_SPARSE_MLA_LATENT_DIM,
            quant_group: 128,
            num_pages,
            reserved: 0,
            input_bf16,
            locations,
            cache,
            page_stride_bytes,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmSparseMlaPadQueryArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub batch_size: u32,
    pub num_heads: u32,
    pub input_dim: u32,
    pub padded_dim: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub input_bf16: *const u16,
    pub output_bf16: *mut u16,
    pub cuda_stream: *mut c_void,
}

impl GlmSparseMlaPadQueryArgs {
    pub fn no_rope(
        batch_size: u32,
        input_bf16: *const u16,
        output_bf16: *mut u16,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_SPARSE_MLA_ABI_VERSION,
            batch_size,
            num_heads: GLM_SPARSE_MLA_HEADS,
            input_dim: GLM_SPARSE_MLA_LATENT_DIM,
            padded_dim: GLM_SPARSE_MLA_PADDED_Q_DIM,
            reserved0: 0,
            reserved1: 0,
            input_bf16,
            output_bf16,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlmSparseMlaDecodeArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub batch_size: u32,
    pub num_heads: u32,
    pub query_dim: u32,
    pub value_dim: u32,
    pub page_size: u32,
    pub history_topk: u32,
    pub tail_topk: u32,
    pub history_splits: u32,
    pub tail_splits: u32,
    pub num_pages: u32,
    pub num_sms: u32,
    pub selected_stride: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub query_bf16: *const u16,
    pub cache: *const u8,
    pub selected_indices: *const i32,
    pub query_positions: *const i64,
    pub sequence_lengths: *const i32,
    pub history_indices: *mut i32,
    pub tail_indices: *mut i32,
    pub history_lengths: *mut i32,
    pub tail_lengths: *mut i32,
    pub history_mid_out_bf16: *mut u16,
    pub history_mid_lse: *mut f32,
    pub output_bf16: *mut u16,
    pub output_lse: *mut f32,
    pub tail_mid_out_bf16: *mut u16,
    pub tail_mid_lse: *mut f32,
    pub tail_output_bf16: *mut u16,
    pub tail_output_lse: *mut f32,
    pub page_stride_bytes: u64,
    pub softmax_scale: f32,
    pub reserved2: u32,
    pub cuda_stream: *mut c_void,
}

impl GlmSparseMlaDecodeArgs {
    #[allow(clippy::too_many_arguments)]
    pub fn gb10_no_rope(
        batch_size: u32,
        num_pages: u32,
        query_bf16: *const u16,
        cache: *const u8,
        selected_indices: *const i32,
        query_positions: *const i64,
        sequence_lengths: *const i32,
        history_indices: *mut i32,
        tail_indices: *mut i32,
        history_lengths: *mut i32,
        tail_lengths: *mut i32,
        history_mid_out_bf16: *mut u16,
        history_mid_lse: *mut f32,
        output_bf16: *mut u16,
        output_lse: *mut f32,
        tail_mid_out_bf16: *mut u16,
        tail_mid_lse: *mut f32,
        tail_output_bf16: *mut u16,
        tail_output_lse: *mut f32,
        page_stride_bytes: u64,
        cuda_stream: *mut c_void,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: GLM_SPARSE_MLA_ABI_VERSION,
            batch_size,
            num_heads: GLM_SPARSE_MLA_HEADS,
            query_dim: GLM_SPARSE_MLA_PADDED_Q_DIM,
            value_dim: GLM_SPARSE_MLA_LATENT_DIM,
            page_size: GLM_SPARSE_MLA_PAGE_SIZE,
            history_topk: GLM_SPARSE_MLA_HISTORY_TOPK,
            tail_topk: GLM_SPARSE_MLA_TAIL_TOPK,
            history_splits: GLM_SPARSE_MLA_HISTORY_SPLITS,
            tail_splits: GLM_SPARSE_MLA_TAIL_SPLITS,
            num_pages,
            num_sms: GLM_SPARSE_MLA_GB10_SMS,
            selected_stride: GLM_SPARSE_MLA_SELECTION_WIDTH,
            reserved0: 0,
            reserved1: 0,
            query_bf16,
            cache,
            selected_indices,
            query_positions,
            sequence_lengths,
            history_indices,
            tail_indices,
            history_lengths,
            tail_lengths,
            history_mid_out_bf16,
            history_mid_lse,
            output_bf16,
            output_lse,
            tail_mid_out_bf16,
            tail_mid_lse,
            tail_output_bf16,
            tail_output_lse,
            page_stride_bytes,
            softmax_scale: 1.0 / 16.0,
            reserved2: 0,
            cuda_stream,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseNvfp4Plan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub padded_n: u64,
    pub padded_k: u64,
    pub scale_padded_n: u64,
    pub group_size: u32,
    pub input_scale_layout: u32,
    pub weight_scale_layout: u32,
    pub output_dtype: u32,
    pub requested_backend: u32,
    pub reserved: u32,
}

impl From<DenseNvfp4Spec> for DenseNvfp4Plan {
    fn from(spec: DenseNvfp4Spec) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            m: spec.m,
            n: spec.n,
            k: spec.k,
            padded_n: spec.padded_n,
            padded_k: spec.padded_k,
            scale_padded_n: spec.scale_padded_n,
            group_size: spec.group_size,
            input_scale_layout: spec.input_scale_layout as u32,
            weight_scale_layout: spec.weight_scale_layout as u32,
            output_dtype: spec.output_dtype as u32,
            requested_backend: spec.requested_backend as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4MatrixView {
    pub packed_data: *const c_void,
    pub block_scales: *const c_void,
    pub packed_row_stride_bytes: u64,
    pub scale_row_stride_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DenseNvfp4Args {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: DenseNvfp4Plan,
    pub input: Nvfp4MatrixView,
    pub weight: Nvfp4MatrixView,
    pub output: *mut c_void,
    pub output_row_stride_bytes: u64,
    pub alpha: f32,
    pub reserved: u32,
    pub workspace: *mut c_void,
    pub workspace_bytes: u64,
    pub cuda_stream: *mut c_void,
    pub alpha_device: *const f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedNvfp4Plan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_groups: u32,
    pub group_size: u32,
    pub total_rows: u64,
    pub input_scale_rows: u64,
    pub n: u64,
    pub k: u64,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub swap_ab: u32,
    pub input_scale_layout: u32,
    pub weight_scale_layout: u32,
    pub output_dtype: u32,
    pub requested_backend: u32,
}

impl From<GroupedNvfp4Spec> for GroupedNvfp4Plan {
    fn from(spec: GroupedNvfp4Spec) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_groups: spec.num_groups,
            group_size: spec.group_size,
            total_rows: spec.total_rows,
            input_scale_rows: spec.input_scale_rows,
            n: spec.n,
            k: spec.k,
            tile_m: spec.tile_m,
            tile_n: spec.tile_n,
            tile_k: spec.tile_k,
            swap_ab: u32::from(spec.swap_ab),
            input_scale_layout: spec.input_scale_layout as u32,
            weight_scale_layout: spec.weight_scale_layout as u32,
            output_dtype: spec.output_dtype as u32,
            requested_backend: spec.requested_backend as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GroupedNvfp4WeightView {
    pub packed_data: *const c_void,
    pub block_scales: *const c_void,
    pub packed_group_stride_bytes: u64,
    pub scale_group_stride_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GroupedNvfp4Args {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: GroupedNvfp4Plan,
    pub input: Nvfp4MatrixView,
    pub weights: GroupedNvfp4WeightView,
    pub m_indptr: *const i32,
    pub alpha_device: *const f32,
    pub output: *mut c_void,
    pub output_row_stride_bytes: u64,
    pub int_workspace: *mut c_void,
    pub int_workspace_bytes: u64,
    pub float_workspace: *mut c_void,
    pub float_workspace_bytes: u64,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SiluNvfp4Plan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_experts: u32,
    pub rows_per_expert: u32,
    pub hidden_size: u64,
    pub group_size: u32,
    pub input_dtype: u32,
    pub output_scale_layout: u32,
    pub requested_backend: u32,
    pub reserved: u32,
}

impl From<SiluNvfp4Spec> for SiluNvfp4Plan {
    fn from(spec: SiluNvfp4Spec) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_experts: spec.num_experts,
            rows_per_expert: spec.rows_per_expert,
            hidden_size: spec.hidden_size,
            group_size: spec.group_size,
            input_dtype: spec.input_dtype as u32,
            output_scale_layout: spec.output_scale_layout as u32,
            requested_backend: spec.requested_backend as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SiluNvfp4Args {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: SiluNvfp4Plan,
    pub input: *const c_void,
    pub input_global_scales: *const f32,
    pub active_rows: *const i32,
    pub packed_output: *mut c_void,
    pub output_scales: *mut c_void,
    pub input_expert_stride_bytes: u64,
    pub output_expert_stride_bytes: u64,
    pub scale_expert_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentedSiluNvfp4Plan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_experts: u32,
    pub group_size: u32,
    pub total_rows: u64,
    pub input_scale_rows: u64,
    pub hidden_size: u64,
    pub input_dtype: u32,
    pub output_scale_layout: u32,
    pub requested_backend: u32,
    pub reserved: u32,
}

impl From<SegmentedSiluNvfp4Spec> for SegmentedSiluNvfp4Plan {
    fn from(spec: SegmentedSiluNvfp4Spec) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_experts: spec.num_experts,
            group_size: spec.group_size,
            total_rows: spec.total_rows,
            input_scale_rows: spec.input_scale_rows,
            hidden_size: spec.hidden_size,
            input_dtype: spec.input_dtype as u32,
            output_scale_layout: spec.output_scale_layout as u32,
            requested_backend: spec.requested_backend as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SegmentedSiluNvfp4Args {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: SegmentedSiluNvfp4Plan,
    pub input: *const c_void,
    pub input_global_scales: *const f32,
    pub active_rows_host: *const i32,
    pub m_indptr_host: *const i32,
    pub scale_row_offsets_host: *const u64,
    pub packed_output: *mut c_void,
    pub output_scales: *mut c_void,
    pub input_row_stride_bytes: u64,
    pub output_row_stride_bytes: u64,
    pub scale_row_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentedNvfp4QuantizePlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_experts: u32,
    pub group_size: u32,
    pub total_rows: u64,
    pub input_scale_rows: u64,
    pub hidden_size: u64,
    pub input_dtype: u32,
    pub output_scale_layout: u32,
    pub requested_backend: u32,
    pub reserved: u32,
}

impl From<SegmentedNvfp4QuantizeSpec> for SegmentedNvfp4QuantizePlan {
    fn from(spec: SegmentedNvfp4QuantizeSpec) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_experts: spec.num_experts,
            group_size: spec.group_size,
            total_rows: spec.total_rows,
            input_scale_rows: spec.input_scale_rows,
            hidden_size: spec.hidden_size,
            input_dtype: spec.input_dtype as u32,
            output_scale_layout: spec.output_scale_layout as u32,
            requested_backend: spec.requested_backend as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SegmentedNvfp4QuantizeArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: SegmentedNvfp4QuantizePlan,
    pub input: *const c_void,
    pub input_global_scales: *const f32,
    pub active_rows_host: *const i32,
    pub m_indptr_host: *const i32,
    pub scale_row_offsets_host: *const u64,
    pub packed_output: *mut c_void,
    pub output_scales: *mut c_void,
    pub input_row_stride_bytes: u64,
    pub output_row_stride_bytes: u64,
    pub scale_row_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeRoutePlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_tokens: u32,
    pub top_k: u32,
    pub num_experts: u32,
    pub reserved: u32,
    pub hidden_size: u64,
    pub total_rows: u64,
}

impl From<MoeRouteSpec> for MoeRoutePlan {
    fn from(spec: MoeRouteSpec) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_tokens: spec.num_tokens,
            top_k: spec.top_k,
            num_experts: spec.num_experts,
            reserved: 0,
            hidden_size: spec.hidden_size,
            total_rows: spec.total_rows,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MoeRouteArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: MoeRoutePlan,
    pub token_input: *const c_void,
    pub route_to_packed_row: *const u32,
    pub packed_input: *mut c_void,
    pub route_weights: *const f32,
    pub packed_expert_output: *const c_void,
    pub token_output: *mut c_void,
    pub token_input_row_stride_bytes: u64,
    pub packed_row_stride_bytes: u64,
    pub expert_output_row_stride_bytes: u64,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeGatePlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub input_dtype: u32,
    pub weight_dtype: u32,
    pub logits_dtype: u32,
    pub requested_backend: u32,
    pub renormalize: u32,
    pub reserved: u32,
}

impl MoeGatePlan {
    pub fn qwen38_flash(num_tokens: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_tokens,
            hidden_size: 2560,
            num_experts: 512,
            top_k: 10,
            input_dtype: DataType::BFloat16 as u32,
            weight_dtype: DataType::BFloat16 as u32,
            logits_dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangCublasMoeGate as u32,
            renormalize: 1,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MoeGateArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: MoeGatePlan,
    pub hidden_states: *const c_void,
    pub router_weight: *const c_void,
    pub router_logits: *mut c_void,
    pub topk_weights: *mut f32,
    pub topk_ids: *mut i32,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedExpertPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub input_dtype: u32,
    pub weight_dtype: u32,
    pub output_dtype: u32,
    pub requested_backend: u32,
    pub output_mode: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

impl SharedExpertPlan {
    pub fn qwen38_flash(num_tokens: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_tokens,
            hidden_size: 2560,
            intermediate_size: 640,
            input_dtype: DataType::BFloat16 as u32,
            weight_dtype: DataType::BFloat16 as u32,
            output_dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangCublasSharedExpert as u32,
            output_mode: 1,
            reserved1: 0,
            reserved2: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SharedExpertArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: SharedExpertPlan,
    pub hidden_states: *const c_void,
    pub gate_up_weight: *const c_void,
    pub down_weight: *const c_void,
    pub shared_gate_weight: *const c_void,
    pub gate_up: *mut c_void,
    pub activated: *mut c_void,
    pub shared_gate: *mut c_void,
    pub output: *mut c_void,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeJoinPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub input_dtype: u32,
    pub output_dtype: u32,
    pub requested_backend: u32,
    pub reserved: u32,
}

impl MoeJoinPlan {
    pub fn qwen38_flash(num_tokens: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_tokens,
            hidden_size: 2560,
            input_dtype: DataType::BFloat16 as u32,
            output_dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangFusedMoeJoin as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MoeJoinArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: MoeJoinPlan,
    pub hidden_states: *const c_void,
    pub shared_gate_weight: *const c_void,
    pub shared_output: *const c_void,
    pub routed_output: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MhcPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_tokens: u32,
    pub hc_count: u32,
    pub hidden_size: u32,
    pub lowrank_size: u32,
    pub dtype: u32,
    pub requested_backend: u32,
    pub rms_norm_eps: f32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

impl MhcPlan {
    pub fn qwen38_flash(num_tokens: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_tokens,
            hc_count: 4,
            hidden_size: 2560,
            lowrank_size: 320,
            dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangCublasMhc as u32,
            rms_norm_eps: 1.0e-6,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MhcArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: MhcPlan,
    pub hyper_input: *const c_void,
    pub norm_weight: *const c_void,
    pub mix_down_weight: *const c_void,
    pub mix_up_weight: *const c_void,
    pub inject_weight: *const c_void,
    pub block_output: *const c_void,
    pub normed: *mut c_void,
    pub mix_down: *mut c_void,
    pub mix_activated: *mut c_void,
    pub mix_up: *mut c_void,
    pub mixed_output: *mut c_void,
    pub combined_output: *mut c_void,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleRowFragment {
    pub first_offset_bytes: u64,
    pub second_offset_bytes: u64,
    pub first_bytes: u32,
    pub second_bytes: u32,
}

impl PleRowFragment {
    pub fn from_fixed(
        row: crate::storage::FixedPleRow,
        row_bytes: usize,
    ) -> Result<Self, &'static str> {
        let second_bytes = row.second.map_or(0, |second| second.bytes);
        if row.first.bytes.checked_add(second_bytes) != Some(row_bytes) {
            return Err("PLE fragments do not cover exactly one row");
        }
        Ok(Self {
            first_offset_bytes: u64::try_from(row.first.buffer_offset)
                .map_err(|_| "PLE first offset exceeds u64")?,
            second_offset_bytes: row.second.map_or(Ok(0), |second| {
                u64::try_from(second.buffer_offset).map_err(|_| "PLE second offset exceeds u64")
            })?,
            first_bytes: u32::try_from(row.first.bytes)
                .map_err(|_| "PLE first fragment exceeds u32")?,
            second_bytes: u32::try_from(second_bytes)
                .map_err(|_| "PLE second fragment exceeds u32")?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleGatherPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub rows: u32,
    pub row_bytes: u32,
    pub input_dtype: u32,
    pub output_dtype: u32,
    pub requested_backend: u32,
    pub reserved: u32,
}

impl PleGatherPlan {
    pub fn qwen38_flash(rows: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            rows,
            row_bytes: 160,
            input_dtype: DataType::Fp8E4m3 as u32,
            output_dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangPleGather as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PleGatherArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: PleGatherPlan,
    pub coherent_base: *const c_void,
    pub fragments: *const PleRowFragment,
    pub output: *mut c_void,
    pub output_row_stride_bytes: u64,
    pub scale_bf16_bits: u16,
    pub reserved16: u16,
    pub reserved32: u32,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaTopkPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub rows: u32,
    pub columns: u32,
    pub topk: u32,
    pub input_dtype: u32,
    pub output_dtype: u32,
    pub requested_backend: u32,
    pub input_stride: u64,
}

impl QsaTopkPlan {
    pub fn qwen38_flash(rows: u32, columns: u32, input_stride: u64) -> Self {
        Self::pooled_history(rows, columns, input_stride)
    }

    pub fn pooled_history(rows: u32, columns: u32, input_stride: u64) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            rows,
            columns,
            topk: 512,
            input_dtype: DataType::Float32 as u32,
            output_dtype: DataType::Int32 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangQsaTopk as u32,
            input_stride,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QsaTopkArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: QsaTopkPlan,
    pub scores: *const f32,
    pub row_starts: *const i32,
    pub lengths: *const i32,
    pub indices: *mut i32,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaExpandPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub rows: u32,
    pub block_topk: u32,
    pub compress_ratio: u32,
    pub token_topk: u32,
    pub final_topk: u32,
    pub output_dtype: u32,
    pub requested_backend: u32,
    pub reserved: u32,
}

impl QsaExpandPlan {
    pub fn qwen38_flash(rows: u32) -> Self {
        Self::pooled_history(rows)
    }

    pub fn pooled_history(rows: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            rows,
            block_topk: 512,
            compress_ratio: 4,
            token_topk: 2048,
            final_topk: 2051,
            output_dtype: DataType::Int32 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangQsaExpand as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QsaExpandArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: QsaExpandPlan,
    pub block_indices: *const i32,
    pub query_positions: *const i64,
    pub sequence_lengths: *const i32,
    pub logical_indices: *mut i32,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaScorePlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub batch_size: u32,
    pub pages: u32,
    pub max_pages: u32,
    pub max_model_len: u32,
    pub query_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub query_dtype: u32,
    pub logits_dtype: u32,
    pub requested_backend: u32,
}

impl QsaScorePlan {
    pub fn qwen38_flash(batch_size: u32, pages: u32, max_pages: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            batch_size,
            pages,
            max_pages,
            max_model_len: max_pages.saturating_mul(16),
            query_heads: 8,
            head_dim: 128,
            page_size: 16,
            query_dtype: DataType::BFloat16 as u32,
            logits_dtype: DataType::Float32 as u32,
            requested_backend: crate::kernel::KernelBackend::TilelangQsaScore as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QsaScoreArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: QsaScorePlan,
    pub query: *const c_void,
    pub key_cache: *const c_void,
    pub page_table: *const i32,
    pub context_lengths: *const i32,
    pub logits: *mut f32,
    pub score_scale: f32,
    pub reserved: u32,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaIndexPrepPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub tokens: u32,
    pub groups: u32,
    pub state_slots: u32,
    pub compressed_slots: u32,
    pub num_q_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub compress_ratio: u32,
    pub num_position_axes: u32,
    pub dtype: u32,
    pub requested_backend: u32,
    pub q_heads_padded: u32,
}

impl QsaIndexPrepPlan {
    pub fn qwen38_flash(
        tokens: u32,
        groups: u32,
        state_slots: u32,
        compressed_slots: u32,
        num_position_axes: u32,
    ) -> Self {
        Self::qwen38_flash_with_rotary(
            tokens,
            groups,
            state_slots,
            compressed_slots,
            num_position_axes,
            128,
        )
    }

    pub fn qwen38_flash_with_rotary(
        tokens: u32,
        groups: u32,
        state_slots: u32,
        compressed_slots: u32,
        num_position_axes: u32,
        rotary_dim: u32,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            tokens,
            groups,
            state_slots,
            compressed_slots,
            num_q_heads: 4,
            head_dim: 128,
            rotary_dim,
            compress_ratio: 4,
            num_position_axes,
            dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangQsaIndexPrep as u32,
            q_heads_padded: 8,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QsaIndexPrepArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: QsaIndexPrepPlan,
    pub qk: *const c_void,
    pub q_output: *mut c_void,
    pub q_norm_weight: *const c_void,
    pub k_norm_weight: *const c_void,
    pub cos_sin_cache: *const f32,
    pub cos_sin_rows: u64,
    pub axis_map: *const i32,
    pub positions: *const i64,
    pub positions_stride: u64,
    pub cache_locs: *const i64,
    pub key_state: *mut c_void,
    pub rope_positions: *mut i64,
    pub group_locs: *const i32,
    pub write_locs: *const i32,
    pub compressed_keys: *mut c_void,
    pub eps: f32,
    pub reserved: u32,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaKvPackPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub batch_size: u32,
    pub slot_capacity: u32,
    pub request_capacity: u32,
    pub request_stride: u32,
    pub topk: u32,
    pub packed_row_stride: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: u32,
    pub requested_backend: u32,
}

impl QsaKvPackPlan {
    pub fn qwen38_flash(
        batch_size: u32,
        slot_capacity: u32,
        request_capacity: u32,
        request_stride: u32,
    ) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            batch_size,
            slot_capacity,
            request_capacity,
            request_stride,
            topk: 2051,
            packed_row_stride: 2112,
            num_kv_heads: 2,
            head_dim: 256,
            dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangQsaKvPack as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QsaKvPackArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: QsaKvPackPlan,
    pub key_state: *const c_void,
    pub value_state: *const c_void,
    pub req_to_token: *const i32,
    pub request_indices: *const i32,
    pub logical_indices: *const i32,
    pub sequence_lengths: *const i32,
    pub valid_counts: *mut i32,
    pub packed_key: *mut c_void,
    pub packed_value: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QsaDecodePlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub batch_size: u32,
    pub multiprocessor_count: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub pages_per_row: u32,
    pub packed_row_stride: u32,
    pub dtype: u32,
    pub requested_backend: u32,
    pub enable_pdl: u32,
    pub reserved: u32,
}

impl QsaDecodePlan {
    pub fn qwen38_flash(batch_size: u32, multiprocessor_count: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            batch_size,
            multiprocessor_count,
            num_q_heads: 24,
            num_kv_heads: 2,
            head_dim: 256,
            page_size: 64,
            pages_per_row: 33,
            packed_row_stride: 2112,
            dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::FlashInferXqaDecode as u32,
            enable_pdl: 1,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QsaDecodeArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: QsaDecodePlan,
    pub query: *const c_void,
    pub packed_key: *const c_void,
    pub packed_value: *const c_void,
    pub block_tables: *const i32,
    pub sequence_lengths: *const i32,
    pub output: *mut c_void,
    pub workspace: *mut c_void,
    pub workspace_bytes: u64,
    pub bmm1_scale: f32,
    pub bmm2_scale: f32,
    pub cuda_stream: *mut c_void,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnBackend {
    Auto = 0,
    LocalCuda = 1,
    FlashInfer = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnDecodePlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub batch_size: u32,
    pub num_qk_heads: u32,
    pub num_value_heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub state_slots: u32,
    pub state_dtype: u32,
    pub requested_backend: u32,
}

impl GdnDecodePlan {
    pub fn qwen38_flash_decode(batch_size: u32, state_slots: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            batch_size,
            num_qk_heads: 16,
            num_value_heads: 48,
            key_dim: 128,
            value_dim: 128,
            state_slots,
            state_dtype: DataType::BFloat16 as u32,
            requested_backend: GdnBackend::Auto as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GdnDecodeArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: GdnDecodePlan,
    pub q: *const c_void,
    pub k: *const c_void,
    pub v: *const c_void,
    pub a: *const c_void,
    pub b: *const c_void,
    pub a_log: *const f32,
    pub dt_bias: *const f32,
    pub state_pool: *mut c_void,
    pub state_indices: *const i32,
    pub output: *mut c_void,
    pub scale: f32,
    pub sequence_length: u32,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnBlockPlan {
    pub struct_size: u32,
    pub abi_version: u32,
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub num_qk_heads: u32,
    pub num_value_heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    pub dtype: u32,
    pub requested_backend: u32,
    pub rms_norm_eps: f32,
    pub reserved: u32,
}

impl GdnBlockPlan {
    pub fn qwen38_flash_decode(num_tokens: u32) -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            num_tokens,
            hidden_size: 2560,
            num_qk_heads: 16,
            num_value_heads: 48,
            head_dim: 128,
            conv_kernel: 4,
            dtype: DataType::BFloat16 as u32,
            requested_backend: crate::kernel::KernelBackend::SglangCublasGdnBlock as u32,
            rms_norm_eps: 1.0e-6,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GdnBlockArgs {
    pub struct_size: u32,
    pub abi_version: u32,
    pub plan: GdnBlockPlan,
    pub hidden_states: *const c_void,
    pub in_proj_qkv_weight: *const c_void,
    pub in_proj_z_weight: *const c_void,
    pub in_proj_b_weight: *const c_void,
    pub in_proj_a_weight: *const c_void,
    pub conv_weight: *const c_void,
    pub gated_norm_weight: *const c_void,
    pub out_proj_weight: *const c_void,
    pub conv_state_pool: *mut c_void,
    pub state_indices: *const i32,
    pub projected_qkv: *mut c_void,
    pub projected_z: *mut c_void,
    pub projected_b: *mut c_void,
    pub projected_a: *mut c_void,
    pub convolved_qkv: *mut c_void,
    pub gdn_core_output: *const c_void,
    pub gated_norm_output: *mut c_void,
    pub attention_output: *mut c_void,
    pub cublas_handle: *mut c_void,
    pub cuda_stream: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KernelInfo {
    pub struct_size: u32,
    pub abi_version: u32,
    pub backend: u32,
    pub available: u32,
    pub workspace_bytes: u64,
    pub name: *const c_char,
    pub source_revision: *const c_char,
}

impl KernelInfo {
    pub fn empty() -> Self {
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            backend: 0,
            available: 0,
            workspace_bytes: 0,
            name: std::ptr::null(),
            source_revision: std::ptr::null(),
        }
    }
}

unsafe extern "C" {
    pub fn sparkserve_coherent_region_validate(config: *const CoherentRegionConfig) -> Status;
    pub fn sparkserve_coherent_region_create(
        config: *const CoherentRegionConfig,
        region: *mut *mut CoherentRegion,
    ) -> Status;
    pub fn sparkserve_coherent_region_view(
        region: *const CoherentRegion,
        view: *mut CoherentRegionView,
    ) -> Status;
    pub fn sparkserve_coherent_region_destroy(region: *mut CoherentRegion) -> Status;
    pub fn sparkserve_cuda_stream_create(stream: *mut *mut CudaStream) -> Status;
    pub fn sparkserve_cuda_stream_raw(
        stream: *const CudaStream,
        raw_stream: *mut *mut c_void,
    ) -> Status;
    pub fn sparkserve_cuda_stream_memset_async(
        stream: *mut CudaStream,
        device_pointer: *mut c_void,
        value: u32,
        bytes: u64,
    ) -> Status;
    pub fn sparkserve_cuda_stream_memcpy_async(
        stream: *mut CudaStream,
        destination_device_pointer: *mut c_void,
        source_device_pointer: *const c_void,
        bytes: u64,
    ) -> Status;
    pub fn sparkserve_cuda_stream_wait_event(
        stream: *mut CudaStream,
        event: *const CudaEvent,
    ) -> Status;
    pub fn sparkserve_cuda_stream_synchronize(stream: *mut CudaStream) -> Status;
    pub fn sparkserve_cuda_stream_destroy(stream: *mut CudaStream) -> Status;
    pub fn sparkserve_cuda_event_create(event: *mut *mut CudaEvent) -> Status;
    pub fn sparkserve_cuda_event_record(event: *mut CudaEvent, stream: *mut CudaStream) -> Status;
    pub fn sparkserve_cuda_event_query(event: *const CudaEvent, complete: *mut u32) -> Status;
    pub fn sparkserve_cuda_event_synchronize(event: *mut CudaEvent) -> Status;
    pub fn sparkserve_cuda_event_destroy(event: *mut CudaEvent) -> Status;
    pub fn sparkserve_cuda_blas_create(blas: *mut *mut CudaBlas) -> Status;
    pub fn sparkserve_cuda_blas_raw(blas: *const CudaBlas, raw_blas: *mut *mut c_void) -> Status;
    pub fn sparkserve_cuda_blas_destroy(blas: *mut CudaBlas) -> Status;
    pub fn sparkserve_qwen_expert_pack_validate(args: *const QwenExpertPackArgs) -> Status;
    pub fn sparkserve_qwen_expert_pack_launch(args: *const QwenExpertPackArgs) -> Status;
    pub fn sparkserve_qwen_bf16_to_f32_launch(args: *const QwenBf16ToF32Args) -> Status;
    pub fn sparkserve_qwen_qsa_project_launch(args: *const QwenQsaProjectArgs) -> Status;
    pub fn sparkserve_qwen_qsa_finish_launch(args: *const QwenQsaFinishArgs) -> Status;
    pub fn sparkserve_qwen_ple_block_launch(args: *const QwenPleBlockArgs) -> Status;
    pub fn sparkserve_qwen_repeat_embedding_launch(args: *const QwenDecodeGlueArgs) -> Status;
    pub fn sparkserve_qwen_add_hyper_launch(args: *const QwenDecodeGlueArgs) -> Status;
    pub fn sparkserve_qwen_qsa_single_value_launch(args: *const QwenDecodeGlueArgs) -> Status;
    pub fn sparkserve_qwen_lm_head_launch(args: *const QwenLmHeadArgs) -> Status;
    pub fn sparkserve_qwen_runtime_mhc_mix(
        caps: *const DeviceCaps,
        args: *const MhcArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_mhc_combine(
        caps: *const DeviceCaps,
        args: *const MhcArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_gdn_prepare(
        caps: *const DeviceCaps,
        args: *const GdnBlockArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_gdn_decode(
        caps: *const DeviceCaps,
        args: *const GdnDecodeArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_gdn_finish(
        caps: *const DeviceCaps,
        args: *const GdnBlockArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_bf16_to_f32(args: *const QwenBf16ToF32Args) -> Status;
    pub fn sparkserve_qwen_runtime_grouped_nvfp4(
        caps: *const DeviceCaps,
        args: *const GroupedNvfp4Args,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_segmented_quantize(
        caps: *const DeviceCaps,
        args: *const SegmentedNvfp4QuantizeArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_segmented_silu(
        caps: *const DeviceCaps,
        args: *const SegmentedSiluNvfp4Args,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_moe_gate(
        caps: *const DeviceCaps,
        args: *const MoeGateArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_moe_dispatch(
        caps: *const DeviceCaps,
        args: *const MoeRouteArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_moe_finalize(
        caps: *const DeviceCaps,
        args: *const MoeRouteArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_shared_expert(
        caps: *const DeviceCaps,
        args: *const SharedExpertArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_moe_join(
        caps: *const DeviceCaps,
        args: *const MoeJoinArgs,
    ) -> Status;
    pub fn sparkserve_qwen_runtime_ple_gather(
        caps: *const DeviceCaps,
        args: *const PleGatherArgs,
    ) -> Status;
    pub fn sparkserve_kernel_abi_version() -> u32;
    pub fn sparkserve_dense_nvfp4_validate(plan: *const DenseNvfp4Plan) -> Status;
    pub fn sparkserve_dense_nvfp4_query(
        caps: *const DeviceCaps,
        plan: *const DenseNvfp4Plan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_dense_nvfp4_launch(
        caps: *const DeviceCaps,
        args: *const DenseNvfp4Args,
    ) -> Status;
    pub fn sparkserve_grouped_nvfp4_validate(plan: *const GroupedNvfp4Plan) -> Status;
    pub fn sparkserve_grouped_nvfp4_query(
        caps: *const DeviceCaps,
        plan: *const GroupedNvfp4Plan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_grouped_nvfp4_launch(
        caps: *const DeviceCaps,
        args: *const GroupedNvfp4Args,
    ) -> Status;
    pub fn sparkserve_silu_nvfp4_validate(plan: *const SiluNvfp4Plan) -> Status;
    pub fn sparkserve_silu_nvfp4_query(
        caps: *const DeviceCaps,
        plan: *const SiluNvfp4Plan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_silu_nvfp4_launch(
        caps: *const DeviceCaps,
        args: *const SiluNvfp4Args,
    ) -> Status;
    pub fn sparkserve_segmented_silu_nvfp4_validate(plan: *const SegmentedSiluNvfp4Plan) -> Status;
    pub fn sparkserve_segmented_silu_nvfp4_query(
        caps: *const DeviceCaps,
        plan: *const SegmentedSiluNvfp4Plan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_segmented_silu_nvfp4_launch(
        caps: *const DeviceCaps,
        args: *const SegmentedSiluNvfp4Args,
    ) -> Status;
    pub fn sparkserve_segmented_nvfp4_quantize_validate(
        plan: *const SegmentedNvfp4QuantizePlan,
    ) -> Status;
    pub fn sparkserve_segmented_nvfp4_quantize_query(
        caps: *const DeviceCaps,
        plan: *const SegmentedNvfp4QuantizePlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_segmented_nvfp4_quantize_launch(
        caps: *const DeviceCaps,
        args: *const SegmentedNvfp4QuantizeArgs,
    ) -> Status;
    pub fn sparkserve_moe_route_validate(plan: *const MoeRoutePlan) -> Status;
    pub fn sparkserve_moe_route_query(
        caps: *const DeviceCaps,
        plan: *const MoeRoutePlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_moe_route_dispatch(
        caps: *const DeviceCaps,
        args: *const MoeRouteArgs,
    ) -> Status;
    pub fn sparkserve_moe_route_finalize(
        caps: *const DeviceCaps,
        args: *const MoeRouteArgs,
    ) -> Status;
    pub fn sparkserve_moe_gate_validate(plan: *const MoeGatePlan) -> Status;
    pub fn sparkserve_moe_gate_query(
        caps: *const DeviceCaps,
        plan: *const MoeGatePlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_moe_gate_launch(caps: *const DeviceCaps, args: *const MoeGateArgs) -> Status;
    pub fn sparkserve_shared_expert_validate(plan: *const SharedExpertPlan) -> Status;
    pub fn sparkserve_shared_expert_query(
        caps: *const DeviceCaps,
        plan: *const SharedExpertPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_shared_expert_launch(
        caps: *const DeviceCaps,
        args: *const SharedExpertArgs,
    ) -> Status;
    pub fn sparkserve_moe_join_validate(plan: *const MoeJoinPlan) -> Status;
    pub fn sparkserve_moe_join_query(
        caps: *const DeviceCaps,
        plan: *const MoeJoinPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_moe_join_launch(caps: *const DeviceCaps, args: *const MoeJoinArgs) -> Status;
    pub fn sparkserve_mhc_validate(plan: *const MhcPlan) -> Status;
    pub fn sparkserve_mhc_query(
        caps: *const DeviceCaps,
        plan: *const MhcPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_mhc_mix_launch(caps: *const DeviceCaps, args: *const MhcArgs) -> Status;
    pub fn sparkserve_mhc_combine_launch(caps: *const DeviceCaps, args: *const MhcArgs) -> Status;
    pub fn sparkserve_ple_gather_validate(plan: *const PleGatherPlan) -> Status;
    pub fn sparkserve_ple_gather_query(
        caps: *const DeviceCaps,
        plan: *const PleGatherPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_ple_gather_launch(
        caps: *const DeviceCaps,
        args: *const PleGatherArgs,
    ) -> Status;
    pub fn sparkserve_qsa_topk_validate(plan: *const QsaTopkPlan) -> Status;
    pub fn sparkserve_qsa_topk_query(
        caps: *const DeviceCaps,
        plan: *const QsaTopkPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_qsa_topk_launch(caps: *const DeviceCaps, args: *const QsaTopkArgs) -> Status;
    pub fn sparkserve_qsa_expand_validate(plan: *const QsaExpandPlan) -> Status;
    pub fn sparkserve_qsa_expand_query(
        caps: *const DeviceCaps,
        plan: *const QsaExpandPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_qsa_expand_launch(
        caps: *const DeviceCaps,
        args: *const QsaExpandArgs,
    ) -> Status;
    pub fn sparkserve_qsa_score_validate(plan: *const QsaScorePlan) -> Status;
    pub fn sparkserve_qsa_score_query(
        caps: *const DeviceCaps,
        plan: *const QsaScorePlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_qsa_score_launch(
        caps: *const DeviceCaps,
        args: *const QsaScoreArgs,
    ) -> Status;
    pub fn sparkserve_qsa_index_prep_validate(plan: *const QsaIndexPrepPlan) -> Status;
    pub fn sparkserve_qsa_index_prep_query(
        caps: *const DeviceCaps,
        plan: *const QsaIndexPrepPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_qsa_index_prep_launch(
        caps: *const DeviceCaps,
        args: *const QsaIndexPrepArgs,
    ) -> Status;
    pub fn sparkserve_qsa_kv_pack_validate(plan: *const QsaKvPackPlan) -> Status;
    pub fn sparkserve_qsa_kv_pack_query(
        caps: *const DeviceCaps,
        plan: *const QsaKvPackPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_qsa_kv_pack_launch(
        caps: *const DeviceCaps,
        args: *const QsaKvPackArgs,
    ) -> Status;
    pub fn sparkserve_qsa_decode_validate(plan: *const QsaDecodePlan) -> Status;
    pub fn sparkserve_qsa_decode_query(
        caps: *const DeviceCaps,
        plan: *const QsaDecodePlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_qsa_decode_launch(
        caps: *const DeviceCaps,
        args: *const QsaDecodeArgs,
    ) -> Status;
    pub fn sparkserve_gdn_decode_validate(plan: *const GdnDecodePlan) -> Status;
    pub fn sparkserve_gdn_decode_query(
        caps: *const DeviceCaps,
        plan: *const GdnDecodePlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_gdn_decode_launch(
        caps: *const DeviceCaps,
        args: *const GdnDecodeArgs,
    ) -> Status;
    pub fn sparkserve_gdn_block_validate(plan: *const GdnBlockPlan) -> Status;
    pub fn sparkserve_gdn_block_query(
        caps: *const DeviceCaps,
        plan: *const GdnBlockPlan,
        info: *mut KernelInfo,
    ) -> Status;
    pub fn sparkserve_gdn_block_prepare_launch(
        caps: *const DeviceCaps,
        args: *const GdnBlockArgs,
    ) -> Status;
    pub fn sparkserve_gdn_block_finish_launch(
        caps: *const DeviceCaps,
        args: *const GdnBlockArgs,
    ) -> Status;
    pub fn sparkserve_ggml_quant_q8_scratch_bytes(vectors: u64, k: u64, bytes: *mut u64) -> Status;
    pub fn sparkserve_ggml_quant_dense_launch(args: *const GgmlQuantDenseArgs) -> Status;
    pub fn sparkserve_ggml_quant_routed_launch(args: *const GgmlQuantRoutedArgs) -> Status;
    pub fn sparkserve_glm_kda_validate(args: *const GlmKdaArgs) -> Status;
    pub fn sparkserve_glm_kda_launch(args: *const GlmKdaArgs) -> Status;
    pub fn sparkserve_glm_kda_conv_validate(args: *const GlmKdaConvArgs) -> Status;
    pub fn sparkserve_glm_kda_conv_launch(args: *const GlmKdaConvArgs) -> Status;
    pub fn sparkserve_glm_kda_prepare_validate(args: *const GlmKdaPrepareArgs) -> Status;
    pub fn sparkserve_glm_kda_prepare_launch(args: *const GlmKdaPrepareArgs) -> Status;
    pub fn sparkserve_glm_kda_gate_validate(args: *const GlmKdaGateArgs) -> Status;
    pub fn sparkserve_glm_kda_gate_launch(args: *const GlmKdaGateArgs) -> Status;
    pub fn sparkserve_glm_kpool_compress_validate(args: *const GlmKPoolCompressArgs) -> Status;
    pub fn sparkserve_glm_kpool_compress_launch(args: *const GlmKPoolCompressArgs) -> Status;
    pub fn sparkserve_glm_kpool_decode_validate(args: *const GlmKPoolDecodeArgs) -> Status;
    pub fn sparkserve_glm_kpool_decode_launch(args: *const GlmKPoolDecodeArgs) -> Status;
    pub fn sparkserve_glm_indexer_prep_validate(args: *const GlmIndexerPrepArgs) -> Status;
    pub fn sparkserve_glm_indexer_prep_launch(args: *const GlmIndexerPrepArgs) -> Status;
    pub fn sparkserve_glm_paged_mqa_validate(args: *const GlmPagedMqaArgs) -> Status;
    pub fn sparkserve_glm_paged_mqa_launch(args: *const GlmPagedMqaArgs) -> Status;
    pub fn sparkserve_glm_sparse_mla_pack_kv_validate(
        args: *const GlmSparseMlaPackKvArgs,
    ) -> Status;
    pub fn sparkserve_glm_sparse_mla_pack_kv_launch(
        args: *const GlmSparseMlaPackKvArgs,
    ) -> Status;
    pub fn sparkserve_glm_sparse_mla_pad_query_validate(
        args: *const GlmSparseMlaPadQueryArgs,
    ) -> Status;
    pub fn sparkserve_glm_sparse_mla_pad_query_launch(
        args: *const GlmSparseMlaPadQueryArgs,
    ) -> Status;
    pub fn sparkserve_glm_sparse_mla_decode_validate(
        args: *const GlmSparseMlaDecodeArgs,
    ) -> Status;
    pub fn sparkserve_glm_sparse_mla_decode_launch(
        args: *const GlmSparseMlaDecodeArgs,
    ) -> Status;
}

fn size_u32<T>() -> u32 {
    u32::try_from(std::mem::size_of::<T>()).expect("kernel ABI structs fit in u32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{DataType, KernelBackend, ScaleLayout};

    #[test]
    fn ffi_layout_matches_version_one_c_header() {
        assert_eq!(std::mem::size_of::<CoherentRegionConfig>(), 48);
        assert_eq!(std::mem::size_of::<CoherentRegionView>(), 80);
        assert_eq!(std::mem::size_of::<DeviceCaps>(), 24);
        assert_eq!(std::mem::size_of::<GgmlQuantDenseArgs>(), 88);
        assert_eq!(std::mem::size_of::<GgmlQuantRoutedArgs>(), 120);
        assert_eq!(std::mem::size_of::<GlmKdaArgs>(), 104);
        assert_eq!(std::mem::size_of::<GlmKdaConvArgs>(), 80);
        assert_eq!(std::mem::size_of::<GlmKdaPrepareArgs>(), 128);
        assert_eq!(std::mem::size_of::<GlmKdaGateArgs>(), 80);
        assert_eq!(std::mem::size_of::<GlmKPoolCompressArgs>(), 104);
        assert_eq!(std::mem::size_of::<GlmKPoolDecodeArgs>(), 168);
        assert_eq!(std::mem::size_of::<GlmIndexerPrepArgs>(), 112);
        assert_eq!(std::mem::size_of::<GlmPagedMqaArgs>(), 136);
        assert_eq!(std::mem::size_of::<GlmSparseMlaPackKvArgs>(), 72);
        assert_eq!(std::mem::size_of::<GlmSparseMlaPadQueryArgs>(), 56);
        assert_eq!(std::mem::size_of::<GlmSparseMlaDecodeArgs>(), 224);
        assert_eq!(std::mem::size_of::<DenseNvfp4Plan>(), 80);
        assert_eq!(std::mem::size_of::<Nvfp4MatrixView>(), 32);
        assert_eq!(std::mem::size_of::<DenseNvfp4Args>(), 208);
        assert_eq!(std::mem::size_of::<GroupedNvfp4Plan>(), 80);
        assert_eq!(std::mem::size_of::<GroupedNvfp4WeightView>(), 32);
        assert_eq!(std::mem::size_of::<GroupedNvfp4Args>(), 224);
        assert_eq!(std::mem::size_of::<SiluNvfp4Plan>(), 48);
        assert_eq!(std::mem::size_of::<SiluNvfp4Args>(), 128);
        assert_eq!(std::mem::size_of::<SegmentedSiluNvfp4Plan>(), 56);
        assert_eq!(std::mem::size_of::<SegmentedSiluNvfp4Args>(), 152);
        assert_eq!(std::mem::size_of::<SegmentedNvfp4QuantizePlan>(), 56);
        assert_eq!(std::mem::size_of::<SegmentedNvfp4QuantizeArgs>(), 152);
        assert_eq!(std::mem::size_of::<MoeRoutePlan>(), 40);
        assert_eq!(std::mem::size_of::<MoeRouteArgs>(), 128);
        assert_eq!(std::mem::size_of::<MoeGatePlan>(), 48);
        assert_eq!(std::mem::size_of::<MoeGateArgs>(), 112);
        assert_eq!(std::mem::size_of::<SharedExpertPlan>(), 48);
        assert_eq!(std::mem::size_of::<SharedExpertArgs>(), 136);
        assert_eq!(std::mem::size_of::<MoeJoinPlan>(), 32);
        assert_eq!(std::mem::size_of::<MoeJoinArgs>(), 80);
        assert_eq!(std::mem::size_of::<MhcPlan>(), 48);
        assert_eq!(std::mem::size_of::<MhcArgs>(), 168);
        assert_eq!(std::mem::size_of::<PleRowFragment>(), 24);
        assert_eq!(std::mem::size_of::<PleGatherPlan>(), 32);
        assert_eq!(std::mem::size_of::<PleGatherArgs>(), 88);
        assert_eq!(std::mem::size_of::<QsaTopkPlan>(), 40);
        assert_eq!(std::mem::size_of::<QsaTopkArgs>(), 88);
        assert_eq!(std::mem::size_of::<QsaExpandPlan>(), 40);
        assert_eq!(std::mem::size_of::<QsaExpandArgs>(), 88);
        assert_eq!(std::mem::size_of::<QsaScorePlan>(), 48);
        assert_eq!(std::mem::size_of::<QsaScoreArgs>(), 112);
        assert_eq!(std::mem::size_of::<QsaIndexPrepPlan>(), 56);
        assert_eq!(std::mem::size_of::<QsaIndexPrepArgs>(), 200);
        assert_eq!(std::mem::size_of::<QsaKvPackPlan>(), 48);
        assert_eq!(std::mem::size_of::<QsaKvPackArgs>(), 136);
        assert_eq!(std::mem::size_of::<QsaDecodePlan>(), 56);
        assert_eq!(std::mem::size_of::<QsaDecodeArgs>(), 144);
        assert_eq!(std::mem::size_of::<GdnDecodePlan>(), 40);
        assert_eq!(std::mem::size_of::<GdnDecodeArgs>(), 144);
        assert_eq!(std::mem::size_of::<GdnBlockPlan>(), 48);
        assert_eq!(std::mem::size_of::<GdnBlockArgs>(), 216);
        assert_eq!(std::mem::size_of::<QwenExpertPackArgs>(), 176);
        assert_eq!(std::mem::size_of::<QwenBf16ToF32Args>(), 40);
        assert_eq!(std::mem::size_of::<QwenQsaProjectArgs>(), 168);
        assert_eq!(std::mem::size_of::<QwenQsaFinishArgs>(), 72);
        assert_eq!(std::mem::size_of::<QwenPleBlockArgs>(), 144);
        assert_eq!(std::mem::size_of::<QwenDecodeGlueArgs>(), 32);
        assert_eq!(std::mem::size_of::<QwenLmHeadArgs>(), 56);
        assert_eq!(std::mem::size_of::<KernelInfo>(), 40);
    }

    #[test]
    fn coherent_file_config_keeps_original_file_range() {
        let path = std::ptr::NonNull::<c_char>::dangling().as_ptr();
        let config = CoherentRegionConfig::file_read_only(
            64 * 1024,
            4096,
            256,
            COHERENT_REGION_PREFAULT,
            path,
        );
        assert_eq!(config.kind, CoherentRegionKind::FileReadOnly as u32);
        assert_eq!(config.payload_bytes, 64 * 1024);
        assert_eq!(config.file_offset, 4096);
        assert_eq!(config.required_alignment, 256);
        assert_eq!(config.file_path, path);
    }

    #[test]
    fn native_spec_converts_without_reinterpreting_enums() {
        let native = DenseNvfp4Spec::native(1, 4096, 4096).expect("valid shape");
        let plan = DenseNvfp4Plan::from(native);
        assert_eq!(plan.abi_version, 1);
        assert_eq!(plan.group_size, 16);
        assert_eq!(plan.input_scale_layout, ScaleLayout::Cutlass128x4 as u32);
        assert_eq!(plan.output_dtype, DataType::BFloat16 as u32);
        assert_eq!(plan.requested_backend, KernelBackend::Auto as u32);
    }

    #[test]
    fn qwen_flash_gdn_plan_matches_checkpoint_topology() {
        let plan = GdnDecodePlan::qwen38_flash_decode(1, 20);
        assert_eq!(plan.num_qk_heads, 16);
        assert_eq!(plan.num_value_heads, 48);
        assert_eq!(plan.key_dim, 128);
        assert_eq!(plan.value_dim, 128);
        assert_eq!(plan.state_dtype, DataType::BFloat16 as u32);
        assert_eq!(plan.requested_backend, GdnBackend::Auto as u32);
    }

    #[test]
    fn qwen_flash_gdn_block_plan_freezes_projection_geometry() {
        let plan = GdnBlockPlan::qwen38_flash_decode(1);
        assert_eq!(plan.hidden_size, 2560);
        assert_eq!(plan.num_qk_heads, 16);
        assert_eq!(plan.num_value_heads, 48);
        assert_eq!(plan.head_dim, 128);
        assert_eq!(plan.conv_kernel, 4);
        assert_eq!(plan.dtype, DataType::BFloat16 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangCublasGdnBlock as u32
        );
    }

    #[test]
    fn qwen_moe_gate_plan_freezes_router_geometry() {
        let plan = MoeGatePlan::qwen38_flash(8);
        assert_eq!(plan.hidden_size, 2560);
        assert_eq!(plan.num_experts, 512);
        assert_eq!(plan.top_k, 10);
        assert_eq!(plan.logits_dtype, DataType::BFloat16 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangCublasMoeGate as u32
        );
        assert_eq!(plan.renormalize, 1);
    }

    #[test]
    fn qwen_shared_expert_plan_freezes_resident_bf16_geometry() {
        let plan = SharedExpertPlan::qwen38_flash(8);
        assert_eq!(plan.hidden_size, 2560);
        assert_eq!(plan.intermediate_size, 640);
        assert_eq!(plan.output_dtype, DataType::BFloat16 as u32);
        assert_eq!(plan.output_mode, 1);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangCublasSharedExpert as u32
        );
    }

    #[test]
    fn qwen_moe_join_plan_freezes_deployed_fused_epilogue() {
        let plan = MoeJoinPlan::qwen38_flash(8);
        assert_eq!(plan.hidden_size, 2560);
        assert_eq!(plan.input_dtype, DataType::BFloat16 as u32);
        assert_eq!(plan.output_dtype, DataType::BFloat16 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangFusedMoeJoin as u32
        );
    }

    #[test]
    fn qwen_mhc_plan_freezes_checkpoint_geometry() {
        let plan = MhcPlan::qwen38_flash(1);
        assert_eq!(plan.hc_count, 4);
        assert_eq!(plan.hidden_size, 2560);
        assert_eq!(plan.lowrank_size, 320);
        assert_eq!(plan.rms_norm_eps, 1.0e-6);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangCublasMhc as u32
        );
    }

    #[test]
    fn qwen_ple_plan_and_fragment_preserve_fixed_slab_offsets() {
        let plan = PleGatherPlan::qwen38_flash(16);
        assert_eq!(plan.rows, 16);
        assert_eq!(plan.row_bytes, 160);
        assert_eq!(plan.input_dtype, DataType::Fp8E4m3 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangPleGather as u32
        );
        let row = crate::storage::FixedPleRow {
            first: crate::storage::FixedPageSlice {
                buffer_offset: 4016,
                bytes: 80,
            },
            second: Some(crate::storage::FixedPageSlice {
                buffer_offset: 8192,
                bytes: 80,
            }),
        };
        assert_eq!(
            PleRowFragment::from_fixed(row, 160).expect("fragment"),
            PleRowFragment {
                first_offset_bytes: 4016,
                second_offset_bytes: 8192,
                first_bytes: 80,
                second_bytes: 80,
            }
        );
        assert!(PleRowFragment::from_fixed(row, 159).is_err());
    }

    #[test]
    fn qwen_qsa_topk_plan_freezes_the_donor_shape() {
        let plan = QsaTopkPlan::qwen38_flash(16, 65_536, 65_536);
        assert_eq!(plan.topk, 512);
        assert_eq!(plan.input_dtype, DataType::Float32 as u32);
        assert_eq!(plan.output_dtype, DataType::Int32 as u32);
        assert_eq!(plan.requested_backend, KernelBackend::SglangQsaTopk as u32);
    }

    #[test]
    fn qwen_qsa_expand_plan_freezes_block_and_tail_geometry() {
        let plan = QsaExpandPlan::qwen38_flash(16);
        assert_eq!(plan.block_topk, 512);
        assert_eq!(plan.compress_ratio, 4);
        assert_eq!(plan.token_topk, 2048);
        assert_eq!(plan.final_topk, 2051);
        assert_eq!(plan.output_dtype, DataType::Int32 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangQsaExpand as u32
        );
    }

    #[test]
    fn qwen_qsa_score_plan_freezes_tilelang_decode_geometry() {
        let plan = QsaScorePlan::qwen38_flash(3, 41, 17);
        assert_eq!(plan.max_model_len, 272);
        assert_eq!(plan.query_heads, 8);
        assert_eq!(plan.head_dim, 128);
        assert_eq!(plan.page_size, 16);
        assert_eq!(plan.query_dtype, DataType::BFloat16 as u32);
        assert_eq!(plan.logits_dtype, DataType::Float32 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::TilelangQsaScore as u32
        );
    }

    #[test]
    fn qwen_qsa_index_prep_plan_freezes_state_geometry() {
        let plan = QsaIndexPrepPlan::qwen38_flash(16, 4, 32_768, 8_192, 1);
        assert_eq!(plan.num_q_heads, 4);
        assert_eq!(plan.q_heads_padded, 8);
        assert_eq!(plan.head_dim, 128);
        assert_eq!(plan.rotary_dim, 128);
        assert_eq!(plan.compress_ratio, 4);
        assert_eq!(plan.dtype, DataType::BFloat16 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangQsaIndexPrep as u32
        );
    }

    #[test]
    fn qwen_qsa_kv_pack_plan_freezes_xqa_page_geometry() {
        let plan = QsaKvPackPlan::qwen38_flash(8, 262_144, 64, 262_144);
        assert_eq!(plan.topk, 2051);
        assert_eq!(plan.packed_row_stride, 2112);
        assert_eq!(plan.num_kv_heads, 2);
        assert_eq!(plan.head_dim, 256);
        assert_eq!(plan.dtype, DataType::BFloat16 as u32);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::SglangQsaKvPack as u32
        );
    }

    #[test]
    fn qwen_qsa_decode_plan_selects_the_working_sm121_xqa_kernel() {
        let plan = QsaDecodePlan::qwen38_flash(8, 48);
        assert_eq!(plan.num_q_heads, 24);
        assert_eq!(plan.num_kv_heads, 2);
        assert_eq!(plan.head_dim, 256);
        assert_eq!(plan.page_size, 64);
        assert_eq!(plan.pages_per_row, 33);
        assert_eq!(plan.packed_row_stride, 2112);
        assert_eq!(plan.enable_pdl, 1);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::FlashInferXqaDecode as u32
        );
    }

    #[test]
    fn grouped_spec_converts_to_flashinfer_donor_contract() {
        use crate::kernel::{GroupedExpertLayout, GroupedNvfp4Spec};

        let layout = GroupedExpertLayout::from_expert_rows(&[4, 0]).expect("layout");
        let native = GroupedNvfp4Spec::qwen_expert_projection(&layout, 640, 2560).expect("spec");
        let plan = GroupedNvfp4Plan::from(native);
        assert_eq!(plan.num_groups, 2);
        assert_eq!(plan.total_rows, 4);
        assert_eq!(plan.input_scale_rows, 256);
        assert_eq!(plan.tile_k, 256);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::FlashInferGroupMmFp4 as u32
        );
    }

    #[test]
    fn fused_spec_converts_to_cute_aot_contract() {
        use crate::kernel::SiluNvfp4Spec;

        let native = SiluNvfp4Spec::qwen_flash(10, 4).expect("fused spec");
        let plan = SiluNvfp4Plan::from(native);
        assert_eq!(plan.hidden_size, 640);
        assert_eq!(plan.num_experts, 10);
        assert_eq!(plan.rows_per_expert, 4);
        assert_eq!(
            plan.requested_backend,
            KernelBackend::FlashInferCuteSiluNvfp4 as u32
        );
    }

    #[test]
    fn segmented_spec_preserves_grouped_offsets() {
        use crate::kernel::{GroupedExpertLayout, SegmentedSiluNvfp4Spec};

        let layout = GroupedExpertLayout::from_expert_rows(&[4, 0, 2]).expect("layout");
        let native =
            SegmentedSiluNvfp4Spec::from_grouped_layout(&layout, 640).expect("segmented spec");
        let plan = SegmentedSiluNvfp4Plan::from(native);
        assert_eq!(plan.num_experts, 3);
        assert_eq!(plan.total_rows, 8);
        assert_eq!(plan.input_scale_rows, 384);
    }

    #[test]
    fn route_and_quantize_specs_convert_without_scheduler_metadata_loss() {
        use crate::kernel::{GroupedExpertLayout, SegmentedNvfp4QuantizeSpec};
        use crate::routing::RoutePlan;

        let routes = RoutePlan::build(1, 2, 4, &[3, 1]).expect("routes");
        let route_plan = MoeRoutePlan::from(routes.kernel_spec(2560).expect("route spec"));
        assert_eq!(route_plan.num_tokens, 1);
        assert_eq!(route_plan.top_k, 2);
        assert_eq!(route_plan.total_rows, 8);

        let layout = GroupedExpertLayout::from_expert_rows(&[0, 1, 0, 1]).expect("layout");
        let quantize =
            SegmentedNvfp4QuantizeSpec::from_grouped_layout(&layout, 2560).expect("quantizer spec");
        let quantize_plan = SegmentedNvfp4QuantizePlan::from(quantize);
        assert_eq!(quantize_plan.num_experts, 4);
        assert_eq!(quantize_plan.total_rows, 8);
        assert_eq!(quantize_plan.hidden_size, 2560);
    }
}

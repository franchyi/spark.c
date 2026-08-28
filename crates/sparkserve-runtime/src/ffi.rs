use std::ffi::{c_char, c_void};

use crate::kernel::{
    DataType, DenseNvfp4Spec, GroupedNvfp4Spec, KERNEL_ABI_VERSION, SegmentedNvfp4QuantizeSpec,
    SegmentedSiluNvfp4Spec, SiluNvfp4Spec,
};
use crate::routing::MoeRouteSpec;

pub const FABRIC_ABI_VERSION: u32 = 1;

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
        Self {
            struct_size: size_u32::<Self>(),
            abi_version: KERNEL_ABI_VERSION,
            tokens,
            groups,
            state_slots,
            compressed_slots,
            num_q_heads: 4,
            head_dim: 128,
            rotary_dim: 128,
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
    pub reserved: u32,
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
        assert_eq!(std::mem::size_of::<KernelInfo>(), 40);
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

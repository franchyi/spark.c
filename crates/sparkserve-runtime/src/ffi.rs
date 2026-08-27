use std::ffi::{c_char, c_void};

use crate::kernel::{DataType, DenseNvfp4Spec, GroupedNvfp4Spec, KERNEL_ABI_VERSION};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Status {
    pub code: i32,
    pub message: *const c_char,
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
        assert_eq!(std::mem::size_of::<DeviceCaps>(), 24);
        assert_eq!(std::mem::size_of::<DenseNvfp4Plan>(), 80);
        assert_eq!(std::mem::size_of::<Nvfp4MatrixView>(), 32);
        assert_eq!(std::mem::size_of::<DenseNvfp4Args>(), 208);
        assert_eq!(std::mem::size_of::<GroupedNvfp4Plan>(), 80);
        assert_eq!(std::mem::size_of::<GroupedNvfp4WeightView>(), 32);
        assert_eq!(std::mem::size_of::<GroupedNvfp4Args>(), 224);
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
}

use std::collections::BTreeMap;
use std::ffi::{CStr, c_void};
use std::path::Path;

use spark_flash_next::checkpoint::load_flash_next_checkpoint;
use spark_flash_next::coherent::CoherentRegionOwner;
use spark_flash_next::cuda::{CudaBlasOwner, CudaStreamOwner};
use spark_flash_next::fabric::{ExpertKey, ExpertLoad, ExpertSlotAddress};
use spark_flash_next::ffi::{
    DeviceCaps, GroupedNvfp4Args, GroupedNvfp4Plan, GroupedNvfp4WeightView,
    MoeGateArgs, MoeGatePlan, MoeJoinArgs, MoeJoinPlan, MoeRouteArgs, MoeRoutePlan,
    Nvfp4MatrixView, SegmentedNvfp4QuantizeArgs, SegmentedNvfp4QuantizePlan,
    SegmentedSiluNvfp4Args, SegmentedSiluNvfp4Plan, SharedExpertArgs,
    SharedExpertPlan, Status, flash_grouped_nvfp4_launch,
    flash_moe_gate_launch, flash_moe_join_launch,
    flash_moe_route_dispatch, flash_moe_route_finalize,
    flash_segmented_nvfp4_quantize_launch,
    flash_segmented_silu_nvfp4_launch, flash_shared_expert_launch,
};
use spark_flash_next::kernel::{
    GroupedNvfp4Spec, KERNEL_ABI_VERSION, SegmentedNvfp4QuantizeSpec,
    SegmentedSiluNvfp4Spec,
};
use spark_flash_next::qwen_expert_cache::QwenExpertHotCache;
use spark_flash_next::qwen_weights::{FlashNextWeightMaps, QwenTensorView};
use spark_flash_next::routing::RoutePlan;

const HIDDEN: u64 = 2560;
const INTERMEDIATE: u64 = 640;
const EXPERTS: u32 = 10;
const TOP_K: u32 = 10;
const WORKSPACE: u64 = 32 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_moe_smoke <model-root> <fixture>"));
    let fixture = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_moe_smoke <model-root> <fixture>"));
    let fixture = Path::new(&fixture);
    let checkpoint = load_flash_next_checkpoint(Path::new(&model))?;
    let mut weights = FlashNextWeightMaps::new(&checkpoint, 0);
    let mut stream = CudaStreamOwner::create()?;
    let blas = CudaBlasOwner::create()?;
    let caps = DeviceCaps::gb10(2 * WORKSPACE);

    let mut hidden = slab(HIDDEN * 2)?;
    write(&mut hidden, &std::fs::read(fixture.join("hidden_bf16.bin"))?)?;
    let router_logits = slab(512 * 2)?;
    let route_weights = slab(u64::from(TOP_K) * 4)?;
    let route_ids = slab(u64::from(TOP_K) * 4)?;
    let router = checked_tensor(
        &mut weights,
        "model.language_model.layers.0.mlp.gate.weight",
        "BF16",
        &[512, 2560],
        512 * HIDDEN * 2,
    )?;
    let router_aligned = aligned_copy(&mut stream, router.device_address, router.data_bytes)?;
    let gate = MoeGateArgs {
        struct_size: size::<MoeGateArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: MoeGatePlan::qwen38_flash(1),
        hidden_states: ptr(hidden.device_address()),
        router_weight: ptr(router_aligned.device_address()),
        router_logits: ptr_mut(router_logits.device_address()),
        topk_weights: ptr_mut(route_weights.device_address()),
        topk_ids: ptr_mut(route_ids.device_address()),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    native("Qwen real router", unsafe { flash_moe_gate_launch(&caps, &gate) })?;
    stream.synchronize()?;
    let logical_ids = unsafe { route_ids.host_payload()? }
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let oracle_ids = std::fs::read(fixture.join("route_experts_i32.bin"))?
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    if logical_ids != oracle_ids {
        return Err(format!("real router ids {logical_ids:?} differ from oracle {oracle_ids:?}").into());
    }

    let mut sorted_ids = logical_ids.clone();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    if sorted_ids.len() != EXPERTS as usize {
        return Err("Qwen decode top-k does not contain ten unique experts".into());
    }
    let slot_by_expert = sorted_ids
        .iter()
        .enumerate()
        .map(|(slot, expert)| (*expert, slot as u32))
        .collect::<BTreeMap<_, _>>();
    let physical_ids = logical_ids
        .iter()
        .map(|expert| slot_by_expert[expert])
        .collect::<Vec<_>>();
    let route = RoutePlan::build(1, TOP_K, EXPERTS, &physical_ids)?;
    let loads = sorted_ids
        .iter()
        .enumerate()
        .map(|(slot, expert)| ExpertLoad {
            key: ExpertKey {
                layer: 0,
                expert: u16::try_from(*expert).expect("Qwen expert id"),
            },
            address: ExpertSlotAddress {
                slot: slot as u32,
                byte_offset: slot as u64 * 2_764_800,
            },
            evicts: None,
        })
        .collect::<Vec<_>>();
    let mut hot = QwenExpertHotCache::create(0)?;
    hot.pack_misses(&mut weights, &loads, &mut stream)?;

    let mut route_map = slab(u64::from(TOP_K) * 4)?;
    write_u32(&mut route_map, &route.route_to_packed_row)?;
    let mut m_indptr = slab((EXPERTS as u64 + 1) * 4)?;
    write_i32(&mut m_indptr, &route.grouped.m_indptr)?;
    let rows = route.grouped.total_rows;
    let scale_rows = route.grouped.input_scale_rows;
    let packed_input = slab(rows * HIDDEN * 2)?;
    let input_fp4 = slab(rows * HIDDEN / 2)?;
    let input_scales = slab(scale_rows * HIDDEN / 16)?;
    let gate_up = slab(rows * 2 * INTERMEDIATE * 2)?;
    let down_input = slab(rows * INTERMEDIATE / 2)?;
    let down_scales = slab(scale_rows * INTERMEDIATE / 16)?;
    let expert_output = slab(rows * HIDDEN * 2)?;
    let routed_output = slab(HIDDEN * 2)?;
    let int_workspace = slab(WORKSPACE)?;
    let float_workspace = slab(WORKSPACE)?;

    let route_spec = route.kernel_spec(HIDDEN)?;
    let route_args = MoeRouteArgs {
        struct_size: size::<MoeRouteArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: MoeRoutePlan::from(route_spec),
        token_input: ptr(hidden.device_address()),
        route_to_packed_row: ptr(route_map.device_address()),
        packed_input: ptr_mut(packed_input.device_address()),
        route_weights: ptr(route_weights.device_address()),
        packed_expert_output: ptr(expert_output.device_address()),
        token_output: ptr_mut(routed_output.device_address()),
        token_input_row_stride_bytes: HIDDEN * 2,
        packed_row_stride_bytes: HIDDEN * 2,
        expert_output_row_stride_bytes: HIDDEN * 2,
        cuda_stream: stream.raw(),
    };
    native("Qwen route dispatch", unsafe {
        flash_moe_route_dispatch(&caps, &route_args)
    })?;

    let active_rows = route
        .expert_rows
        .iter()
        .map(|rows| i32::try_from(*rows).expect("active rows"))
        .collect::<Vec<_>>();
    let hot_views = hot.views();
    let quantize_spec = SegmentedNvfp4QuantizeSpec::from_grouped_layout(
        &route.grouped,
        HIDDEN,
    )?;
    let quantize = SegmentedNvfp4QuantizeArgs {
        struct_size: size::<SegmentedNvfp4QuantizeArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: SegmentedNvfp4QuantizePlan::from(quantize_spec),
        input: ptr(packed_input.device_address()),
        input_global_scales: ptr(hot_views.w13_input_global_scales),
        active_rows_host: active_rows.as_ptr(),
        m_indptr_host: route.grouped.m_indptr.as_ptr(),
        scale_row_offsets_host: route.grouped.scale_row_offsets.as_ptr(),
        packed_output: ptr_mut(input_fp4.device_address()),
        output_scales: ptr_mut(input_scales.device_address()),
        input_row_stride_bytes: HIDDEN * 2,
        output_row_stride_bytes: HIDDEN / 2,
        scale_row_stride_bytes: HIDDEN / 16,
        cuda_stream: stream.raw(),
    };
    native("Qwen routed input quantize", unsafe {
        flash_segmented_nvfp4_quantize_launch(&caps, &quantize)
    })?;

    let w13_spec = GroupedNvfp4Spec::qwen_expert_projection(
        &route.grouped,
        2 * INTERMEDIATE,
        HIDDEN,
    )?;
    let w13 = grouped_args(
        GroupedNvfp4Plan::from(w13_spec),
        input_fp4.device_address(),
        input_scales.device_address(),
        hot_views.w13_weights,
        hot_views.w13_scales,
        m_indptr.device_address(),
        hot_views.w13_alpha,
        gate_up.device_address(),
        int_workspace.device_address(),
        float_workspace.device_address(),
        2 * INTERMEDIATE,
        HIDDEN,
        stream.raw(),
    );
    native("Qwen grouped gate/up", unsafe {
        flash_grouped_nvfp4_launch(&caps, &w13)
    })?;

    let silu_spec = SegmentedSiluNvfp4Spec::from_grouped_layout(
        &route.grouped,
        INTERMEDIATE,
    )?;
    let silu = SegmentedSiluNvfp4Args {
        struct_size: size::<SegmentedSiluNvfp4Args>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: SegmentedSiluNvfp4Plan::from(silu_spec),
        input: ptr(gate_up.device_address()),
        input_global_scales: ptr(hot_views.w2_input_global_scales),
        active_rows_host: active_rows.as_ptr(),
        m_indptr_host: route.grouped.m_indptr.as_ptr(),
        scale_row_offsets_host: route.grouped.scale_row_offsets.as_ptr(),
        packed_output: ptr_mut(down_input.device_address()),
        output_scales: ptr_mut(down_scales.device_address()),
        input_row_stride_bytes: 2 * INTERMEDIATE * 2,
        output_row_stride_bytes: INTERMEDIATE / 2,
        scale_row_stride_bytes: INTERMEDIATE / 16,
        cuda_stream: stream.raw(),
    };
    native("Qwen fused SiLU quantize", unsafe {
        flash_segmented_silu_nvfp4_launch(&caps, &silu)
    })?;

    let w2_spec = GroupedNvfp4Spec::qwen_expert_projection(
        &route.grouped,
        HIDDEN,
        INTERMEDIATE,
    )?;
    let w2 = grouped_args(
        GroupedNvfp4Plan::from(w2_spec),
        down_input.device_address(),
        down_scales.device_address(),
        hot_views.w2_weights,
        hot_views.w2_scales,
        m_indptr.device_address(),
        hot_views.w2_alpha,
        expert_output.device_address(),
        int_workspace.device_address(),
        float_workspace.device_address(),
        HIDDEN,
        INTERMEDIATE,
        stream.raw(),
    );
    native("Qwen grouped down", unsafe {
        flash_grouped_nvfp4_launch(&caps, &w2)
    })?;
    native("Qwen route finalize", unsafe {
        flash_moe_route_finalize(&caps, &route_args)
    })?;

    let shared_gate = checked_tensor(
        &mut weights,
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "BF16",
        &[640, 2560],
        INTERMEDIATE * HIDDEN * 2,
    )?;
    let shared_up = checked_tensor(
        &mut weights,
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "BF16",
        &[640, 2560],
        INTERMEDIATE * HIDDEN * 2,
    )?;
    if shared_up.device_address != shared_gate.device_address + shared_gate.data_bytes {
        return Err("Qwen shared gate/up tensors are not contiguous".into());
    }
    let shared_gate_up_weight = aligned_copy(
        &mut stream,
        shared_gate.device_address,
        shared_gate.data_bytes + shared_up.data_bytes,
    )?;
    let shared_down = checked_tensor(
        &mut weights,
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
        "BF16",
        &[2560, 640],
        HIDDEN * INTERMEDIATE * 2,
    )?;
    let shared_gate_weight = checked_tensor(
        &mut weights,
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        "BF16",
        &[1, 2560],
        HIDDEN * 2,
    )?;
    let shared_down_aligned = aligned_copy(
        &mut stream,
        shared_down.device_address,
        shared_down.data_bytes,
    )?;
    let shared_gate_weight_aligned = aligned_copy(
        &mut stream,
        shared_gate_weight.device_address,
        shared_gate_weight.data_bytes,
    )?;
    let shared_gate_up = slab(2 * INTERMEDIATE * 2)?;
    let shared_activated = slab(INTERMEDIATE * 2)?;
    let shared_output = slab(HIDDEN * 2)?;
    let shared = SharedExpertArgs {
        struct_size: size::<SharedExpertArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: SharedExpertPlan::qwen38_flash(1),
        hidden_states: ptr(hidden.device_address()),
        gate_up_weight: ptr(shared_gate_up_weight.device_address()),
        down_weight: ptr(shared_down_aligned.device_address()),
        shared_gate_weight: ptr(shared_gate_weight_aligned.device_address()),
        gate_up: ptr_mut(shared_gate_up.device_address()),
        activated: ptr_mut(shared_activated.device_address()),
        shared_gate: std::ptr::null_mut(),
        output: ptr_mut(shared_output.device_address()),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    native("Qwen shared expert", unsafe {
        flash_shared_expert_launch(&caps, &shared)
    })?;
    let join = MoeJoinArgs {
        struct_size: size::<MoeJoinArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: MoeJoinPlan::qwen38_flash(1),
        hidden_states: ptr(hidden.device_address()),
        shared_gate_weight: ptr(shared_gate_weight_aligned.device_address()),
        shared_output: ptr(shared_output.device_address()),
        routed_output: ptr_mut(routed_output.device_address()),
        cuda_stream: stream.raw(),
    };
    native("Qwen routed/shared join", unsafe {
        flash_moe_join_launch(&caps, &join)
    })?;
    stream.synchronize()?;
    let expected = std::fs::read(fixture.join("joined_output_bf16.bin"))?;
    if unsafe { routed_output.host_payload()? } != expected.as_slice() {
        return Err("real checkpoint Qwen MoE output differs from SGLang oracle".into());
    }
    println!(
        "Qwen real checkpoint router -> mmap experts -> NVFP4 MoE -> shared join: exact"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn grouped_args(
    plan: GroupedNvfp4Plan,
    input: u64,
    input_scales: u64,
    weights: u64,
    weight_scales: u64,
    m_indptr: u64,
    alpha: u64,
    output: u64,
    int_workspace: u64,
    float_workspace: u64,
    n: u64,
    k: u64,
    stream: *mut c_void,
) -> GroupedNvfp4Args {
    GroupedNvfp4Args {
        struct_size: size::<GroupedNvfp4Args>(),
        abi_version: KERNEL_ABI_VERSION,
        plan,
        input: Nvfp4MatrixView {
            packed_data: ptr(input),
            block_scales: ptr(input_scales),
            packed_row_stride_bytes: k / 2,
            scale_row_stride_bytes: k / 16,
        },
        weights: GroupedNvfp4WeightView {
            packed_data: ptr(weights),
            block_scales: ptr(weight_scales),
            packed_group_stride_bytes: n * k / 2,
            scale_group_stride_bytes: n * k / 16,
        },
        m_indptr: ptr(m_indptr),
        alpha_device: ptr(alpha),
        output: ptr_mut(output),
        output_row_stride_bytes: n * 2,
        int_workspace: ptr_mut(int_workspace),
        int_workspace_bytes: WORKSPACE,
        float_workspace: ptr_mut(float_workspace),
        float_workspace_bytes: WORKSPACE,
        cuda_stream: stream,
    }
}

fn checked_tensor(
    weights: &mut FlashNextWeightMaps,
    name: &str,
    dtype: &str,
    shape: &[u64],
    bytes: u64,
) -> Result<QwenTensorView, Box<dyn std::error::Error>> {
    // This checkpoint's safetensors data section starts at an odd file byte.
    // The first execution probe deliberately preserves the direct mmap view;
    // production repacking can align the resident BF16 section once on NVMe.
    let tensor = weights.tensor(name, 1)?;
    if tensor.dtype != dtype || tensor.shape != shape || tensor.data_bytes != bytes {
        return Err(format!("unexpected Qwen tensor geometry for {name}").into());
    }
    Ok(tensor)
}

fn aligned_copy(
    stream: &mut CudaStreamOwner,
    source: u64,
    bytes: u64,
) -> Result<CoherentRegionOwner, Box<dyn std::error::Error>> {
    let destination = slab(bytes)?;
    unsafe {
        stream.memcpy_async(
            destination.device_address(),
            source,
            usize::try_from(bytes)?,
        )?;
    }
    Ok(destination)
}

fn slab(bytes: u64) -> Result<CoherentRegionOwner, Box<dyn std::error::Error>> {
    Ok(CoherentRegionOwner::slab(bytes, 256, 0)?)
}

fn write(region: &mut CoherentRegionOwner, source: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let destination = unsafe { region.host_payload_mut()? };
    if destination.len() != source.len() {
        return Err("coherent slab size mismatch".into());
    }
    destination.copy_from_slice(source);
    Ok(())
}

fn write_u32(region: &mut CoherentRegionOwner, values: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    write_words(region, values.iter().map(|value| value.to_ne_bytes()))
}

fn write_i32(region: &mut CoherentRegionOwner, values: &[i32]) -> Result<(), Box<dyn std::error::Error>> {
    write_words(region, values.iter().map(|value| value.to_ne_bytes()))
}

fn write_words(
    region: &mut CoherentRegionOwner,
    values: impl Iterator<Item = [u8; 4]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let destination = unsafe { region.host_payload_mut()? };
    let mut offset = 0;
    for word in values {
        destination[offset..offset + 4].copy_from_slice(&word);
        offset += 4;
    }
    if offset != destination.len() {
        return Err("coherent word slab size mismatch".into());
    }
    Ok(())
}

fn native(stage: &str, status: Status) -> Result<(), Box<dyn std::error::Error>> {
    if status.code == 0 {
        return Ok(());
    }
    let message = if status.message.is_null() {
        "native error".to_owned()
    } else {
        unsafe { CStr::from_ptr(status.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(format!("{stage}: {message} (status {})", status.code).into())
}

fn size<T>() -> u32 {
    u32::try_from(std::mem::size_of::<T>()).expect("ABI size fits u32")
}

fn ptr<T>(address: u64) -> *const T {
    address as usize as *const T
}

fn ptr_mut<T>(address: u64) -> *mut T {
    address as usize as *mut T
}

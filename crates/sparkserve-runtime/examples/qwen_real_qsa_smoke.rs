use std::ffi::CStr;
use std::path::Path;

use sparkserve_runtime::checkpoint::load_flash_next_checkpoint;
use sparkserve_runtime::coherent::CoherentRegionOwner;
use sparkserve_runtime::cuda::{CudaBlasOwner, CudaStreamOwner};
use sparkserve_runtime::ffi::{
    DeviceCaps, QWEN_QSA_BLOCK_ABI_VERSION, QsaIndexPrepArgs, QsaIndexPrepPlan,
    QwenQsaFinishArgs, QwenQsaProjectArgs, Status, sparkserve_qsa_index_prep_launch,
    sparkserve_qwen_qsa_finish_launch, sparkserve_qwen_qsa_project_launch,
};
use sparkserve_runtime::kernel::KERNEL_ABI_VERSION;
use sparkserve_runtime::qwen_weights::{FlashNextWeightMaps, QwenTensorView};

const LAYER: u32 = 3;
const HIDDEN: u64 = 2560;
const QUERY_HEADS: u64 = 24;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 256;
const QUERY_WIDTH: u64 = QUERY_HEADS * HEAD_DIM;
const PROJECTED_QUERY_WIDTH: u64 = 2 * QUERY_WIDTH;
const KV_WIDTH: u64 = KV_HEADS * HEAD_DIM;
const INDEX_HEADS: u64 = 4;
const INDEX_HEAD_DIM: u64 = 128;
const INDEX_WIDTH: u64 = (INDEX_HEADS + 1) * INDEX_HEAD_DIM;
const ROTARY_DIM: u32 = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_qsa_smoke <model-root> <fixture>"));
    let fixture = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_qsa_smoke <model-root> <fixture>"));
    let fixture = Path::new(&fixture);

    let checkpoint = load_flash_next_checkpoint(Path::new(&model))?;
    let mut weights = FlashNextWeightMaps::new(&checkpoint, 0);
    let mut stream = CudaStreamOwner::create()?;
    let blas = CudaBlasOwner::create()?;
    let caps = DeviceCaps::gb10(0);
    let prefix = format!("model.language_model.layers.{LAYER}.self_attn");

    let q_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.q_proj.weight"),
        &[PROJECTED_QUERY_WIDTH, HIDDEN],
        PROJECTED_QUERY_WIDTH * HIDDEN * 2,
    )?;
    let k_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.k_proj.weight"),
        &[KV_WIDTH, HIDDEN],
        KV_WIDTH * HIDDEN * 2,
    )?;
    let v_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.v_proj.weight"),
        &[KV_WIDTH, HIDDEN],
        KV_WIDTH * HIDDEN * 2,
    )?;
    let out_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.o_proj.weight"),
        &[HIDDEN, QUERY_WIDTH],
        HIDDEN * QUERY_WIDTH * 2,
    )?;
    let q_norm = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.q_norm.weight"),
        &[HEAD_DIM],
        HEAD_DIM * 2,
    )?;
    let k_norm = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.k_norm.weight"),
        &[HEAD_DIM],
        HEAD_DIM * 2,
    )?;
    let index_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.indexer.index_qk_proj.weight"),
        &[INDEX_WIDTH, HIDDEN],
        INDEX_WIDTH * HIDDEN * 2,
    )?;
    let index_q_norm = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.indexer.q_layernorm.weight"),
        &[INDEX_HEAD_DIM],
        INDEX_HEAD_DIM * 2,
    )?;

    let hidden = fixture_slab(fixture, "hidden_bf16.bin")?;
    let cos_sin = fixture_slab(fixture, "cos_sin_f32.bin")?;
    let positions = fixture_slab(fixture, "positions_i64.bin")?;
    let projected_q = slab(PROJECTED_QUERY_WIDTH * 2)?;
    let projected_k = slab(KV_WIDTH * 2)?;
    let query = slab(QUERY_WIDTH * 2)?;
    let key = slab(KV_WIDTH * 2)?;
    let value = slab(KV_WIDTH * 2)?;
    let gate = slab(QUERY_WIDTH * 2)?;
    let index_qk = slab(INDEX_WIDTH * 2)?;

    let project = QwenQsaProjectArgs {
        struct_size: size::<QwenQsaProjectArgs>(),
        abi_version: QWEN_QSA_BLOCK_ABI_VERSION,
        tokens: 1,
        rotary_dim: ROTARY_DIM,
        cos_sin_stride: u64::from(ROTARY_DIM),
        hidden_states: ptr(hidden.device_address()),
        q_weight: ptr(q_weight.device_address()),
        k_weight: ptr(k_weight.device_address()),
        v_weight: ptr(v_weight.device_address()),
        index_qk_weight: ptr(index_weight.device_address()),
        q_norm_weight: ptr(q_norm.device_address()),
        k_norm_weight: ptr(k_norm.device_address()),
        cos_sin_cache: ptr(cos_sin.device_address()),
        positions: ptr(positions.device_address()),
        projected_q: ptr_mut(projected_q.device_address()),
        projected_k: ptr_mut(projected_k.device_address()),
        query: ptr_mut(query.device_address()),
        key: ptr_mut(key.device_address()),
        value: ptr_mut(value.device_address()),
        gate: ptr_mut(gate.device_address()),
        index_qk: ptr_mut(index_qk.device_address()),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    native("Qwen real QSA projections", unsafe {
        sparkserve_qwen_qsa_project_launch(&project)
    })?;

    let axis_map = fixture_slab(fixture, "axis_map_i32.bin")?;
    let cache_locs = fixture_slab(fixture, "cache_locs_i64.bin")?;
    let index_query = slab(8 * INDEX_HEAD_DIM * 2)?;
    let index_key_state = slab(INDEX_HEAD_DIM * 2)?;
    let rope_positions = slab(3 * 8)?;
    let index_prep = QsaIndexPrepArgs {
        struct_size: size::<QsaIndexPrepArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: QsaIndexPrepPlan::qwen38_flash_with_rotary(1, 0, 1, 1, 1, ROTARY_DIM),
        qk: ptr(index_qk.device_address()),
        q_output: ptr_mut(index_query.device_address()),
        q_norm_weight: ptr(index_q_norm.device_address()),
        k_norm_weight: std::ptr::null(),
        cos_sin_cache: ptr(cos_sin.device_address()),
        cos_sin_rows: u64::try_from(cos_sin.payload_bytes()?)? / (u64::from(ROTARY_DIM) * 4),
        axis_map: ptr(axis_map.device_address()),
        positions: ptr(positions.device_address()),
        positions_stride: 1,
        cache_locs: ptr(cache_locs.device_address()),
        key_state: ptr_mut(index_key_state.device_address()),
        rope_positions: ptr_mut(rope_positions.device_address()),
        group_locs: std::ptr::null(),
        write_locs: std::ptr::null(),
        compressed_keys: std::ptr::null_mut(),
        eps: 1.0e-6,
        reserved: 0,
        cuda_stream: stream.raw(),
    };
    native("Qwen real QSA index preparation", unsafe {
        sparkserve_qsa_index_prep_launch(&caps, &index_prep)
    })?;

    let attention = fixture_slab(fixture, "attention_output_bf16.bin")?;
    let gated = slab(QUERY_WIDTH * 2)?;
    let output = slab(HIDDEN * 2)?;
    let finish = QwenQsaFinishArgs {
        struct_size: size::<QwenQsaFinishArgs>(),
        abi_version: QWEN_QSA_BLOCK_ABI_VERSION,
        tokens: 1,
        reserved: 0,
        attention_output: ptr(attention.device_address()),
        gate: ptr(gate.device_address()),
        out_weight: ptr(out_weight.device_address()),
        gated_output: ptr_mut(gated.device_address()),
        output: ptr_mut(output.device_address()),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    native("Qwen real QSA gate and output", unsafe {
        sparkserve_qwen_qsa_finish_launch(&finish)
    })?;
    stream.synchronize()?;

    for (region, file, stage, tolerance) in [
        (&projected_q, "projected_q_bf16.bin", "Q/gate projection", 0.0078125),
        (&projected_k, "projected_k_bf16.bin", "K projection", 0.0078125),
        (&query, "query_bf16.bin", "Q norm/RoPE", 0.0078125),
        (&key, "key_bf16.bin", "K norm/RoPE", 0.0078125),
        (&value, "value_bf16.bin", "V projection", 0.0078125),
        (&gate, "gate_bf16.bin", "gate deinterleave", 0.0078125),
        (&index_qk, "index_qk_bf16.bin", "index projection", 0.0078125),
        (&index_query, "index_query_bf16.bin", "index Q norm/RoPE", 0.03125),
        (&index_key_state, "index_key_state_bf16.bin", "index K state", 0.0078125),
        (&gated, "gated_output_bf16.bin", "attention gate", 0.0078125),
        (&output, "output_bf16.bin", "output projection", 0.0078125),
    ] {
        expect_bf16_fixture(region, fixture, file, stage, tolerance)?;
    }
    expect_fixture(
        &rope_positions,
        fixture,
        "rope_positions_i64.bin",
        "index position state",
    )?;
    println!("Qwen real checkpoint QSA projections -> index prep -> gate -> output: within donor parity bounds");
    Ok(())
}

fn checkpoint_weight(
    weights: &mut FlashNextWeightMaps,
    stream: &mut CudaStreamOwner,
    name: &str,
    shape: &[u64],
    bytes: u64,
) -> Result<CoherentRegionOwner, Box<dyn std::error::Error>> {
    let tensor = checked_tensor(weights, name, shape, bytes)?;
    let destination = slab(bytes)?;
    unsafe {
        stream.memcpy_async(
            destination.device_address(),
            tensor.device_address,
            usize::try_from(bytes)?,
        )?;
    }
    Ok(destination)
}

fn checked_tensor(
    weights: &mut FlashNextWeightMaps,
    name: &str,
    shape: &[u64],
    bytes: u64,
) -> Result<QwenTensorView, Box<dyn std::error::Error>> {
    let tensor = weights.tensor(name, 1)?;
    if tensor.dtype != "BF16" || tensor.shape != shape || tensor.data_bytes != bytes {
        return Err(format!("unexpected Qwen tensor geometry for {name}").into());
    }
    Ok(tensor)
}

fn fixture_slab(
    fixture: &Path,
    file: &str,
) -> Result<CoherentRegionOwner, Box<dyn std::error::Error>> {
    let source = std::fs::read(fixture.join(file))?;
    let mut region = slab(u64::try_from(source.len())?)?;
    unsafe { region.host_payload_mut()? }.copy_from_slice(&source);
    Ok(region)
}

fn expect_fixture(
    region: &CoherentRegionOwner,
    fixture: &Path,
    file: &str,
    stage: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = std::fs::read(fixture.join(file))?;
    let actual = unsafe { region.host_payload()? };
    if actual != expected.as_slice() {
        let mismatches = actual
            .iter()
            .zip(&expected)
            .filter(|(actual, expected)| actual != expected)
            .count();
        return Err(format!("{stage} differs from oracle in {mismatches} bytes").into());
    }
    Ok(())
}

fn expect_bf16_fixture(
    region: &CoherentRegionOwner,
    fixture: &Path,
    file: &str,
    stage: &str,
    tolerance: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = std::fs::read(fixture.join(file))?;
    let actual = unsafe { region.host_payload()? };
    if actual.len() != expected.len() || !actual.len().is_multiple_of(2) {
        return Err(format!("{stage} fixture size is invalid").into());
    }
    let mut mismatches = 0_usize;
    let mut max_error = 0.0_f32;
    for (actual, expected) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
        let actual_bits = u32::from(u16::from_ne_bytes(actual.try_into()?)) << 16;
        let expected_bits = u32::from(u16::from_ne_bytes(expected.try_into()?)) << 16;
        if actual_bits != expected_bits {
            mismatches += 1;
        }
        let error = (f32::from_bits(actual_bits) - f32::from_bits(expected_bits)).abs();
        if !error.is_finite() {
            return Err(format!("{stage} produced a non-finite difference").into());
        }
        max_error = max_error.max(error);
    }
    println!("{stage}: {mismatches} BF16 differences, max abs {max_error}");
    if max_error > tolerance {
        return Err(format!("{stage} max error {max_error} exceeds {tolerance}").into());
    }
    Ok(())
}

fn slab(bytes: u64) -> Result<CoherentRegionOwner, Box<dyn std::error::Error>> {
    Ok(CoherentRegionOwner::slab(bytes, 256, 0)?)
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

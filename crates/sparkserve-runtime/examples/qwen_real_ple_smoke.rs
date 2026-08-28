use std::ffi::{CStr, c_void};
use std::path::Path;

use sparkserve_runtime::checkpoint::load_flash_next_checkpoint;
use sparkserve_runtime::coherent::CoherentRegionOwner;
use sparkserve_runtime::cuda::{CudaBlasOwner, CudaStreamOwner};
use sparkserve_runtime::ffi::{
    DeviceCaps, PleGatherArgs, PleGatherPlan, PleRowFragment, QWEN_PLE_BLOCK_ABI_VERSION,
    QwenPleBlockArgs, Status, sparkserve_ple_gather_launch, sparkserve_qwen_ple_block_launch,
};
use sparkserve_runtime::kernel::KERNEL_ABI_VERSION;
use sparkserve_runtime::qwen_ple::decode_row_ids;
use sparkserve_runtime::qwen_weights::{FlashNextWeightMaps, QwenTensorView};
use sparkserve_runtime::storage::{FixedPleCache, PleIndex};

const LAYER: u32 = 1;
const HIDDEN: u64 = 2560;
const STREAMS: u64 = 4;
const HYPER: u64 = HIDDEN * STREAMS;
const PLE_CACHE_BYTES: u64 = 4 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_ple_smoke <model-root> <fixture>"));
    let fixture = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_ple_smoke <model-root> <fixture>"));
    let fixture = Path::new(&fixture);

    let model_path = Path::new(&model);
    let checkpoint = load_flash_next_checkpoint(model_path)?;
    let mut weights = FlashNextWeightMaps::new(&checkpoint, 0);
    let mut stream = CudaStreamOwner::create()?;
    let blas = CudaBlasOwner::create()?;
    let caps = DeviceCaps::gb10(0);
    let prefix = format!("model.language_model.layers.{LAYER}.ple");
    let key_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.key_proj.weight"),
        &[HYPER, HIDDEN],
        HYPER * HIDDEN * 2,
    )?;
    let value_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.value_proj.weight"),
        &[HIDDEN, HIDDEN],
        HIDDEN * HIDDEN * 2,
    )?;
    let norm_key = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.norm_key.weight"),
        &[HYPER],
        HYPER * 2,
    )?;
    let norm_query = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.norm_query.weight"),
        &[HYPER],
        HYPER * 2,
    )?;
    let norm_conv = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.norm_conv.weight"),
        &[HYPER],
        HYPER * 2,
    )?;
    let conv_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.conv1d.weight"),
        &[HYPER, 1, 4],
        HYPER * 4 * 2,
    )?;

    let hidden = fixture_slab(fixture, "hidden_bf16.bin")?;
    let multipliers: [i64; 3] = read_i64(fixture, "layer_multipliers_i64.bin")?
        .try_into()
        .map_err(|_| "PLE multiplier fixture has the wrong length")?;
    let head_sizes: [i64; 16] = read_i64(fixture, "head_vocab_sizes_i64.bin")?
        .try_into()
        .map_err(|_| "PLE head-size fixture has the wrong length")?;
    let head_offsets: [i64; 16] = read_i64(fixture, "head_offsets_i64.bin")?
        .try_into()
        .map_err(|_| "PLE head-offset fixture has the wrong length")?;
    let history = read_i64(fixture, "history_i64.bin")?;
    let eos = read_i64(fixture, "eos_i64.bin")?;
    let row_ids = decode_row_ids(&history, eos[0], multipliers, head_sizes, head_offsets)?;
    let expected_row_ids = read_i64(fixture, "row_ids_i64.bin")?;
    if row_ids.map(|row| row as i64) != expected_row_ids.as_slice() {
        return Err("native PLE hash differs from SGLang semantics".into());
    }

    let ple_index = PleIndex::decode(&std::fs::read(
        model_path.join(".sparkserve/ple.ssple"),
    )?)?;
    let cache_region = slab(PLE_CACHE_BYTES)?;
    let mut cache = unsafe {
        FixedPleCache::open_coherent(&ple_index, model_path, cache_region.view(), 32)?
    };
    let batch = cache.fetch_rows(&ple_index, &row_ids)?;
    let mut fragments = [PleRowFragment {
        first_offset_bytes: 0,
        second_offset_bytes: 0,
        first_bytes: 0,
        second_bytes: 0,
    }; 16];
    batch.write_kernel_fragments(&mut fragments)?;
    let mut fragment_region = slab(u64::try_from(std::mem::size_of_val(&fragments))?)?;
    let fragment_bytes = unsafe {
        std::slice::from_raw_parts(
            fragments.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&fragments),
        )
    };
    unsafe { fragment_region.host_payload_mut()? }.copy_from_slice(fragment_bytes);
    let embedding = slab(HIDDEN * 2)?;
    let gather = PleGatherArgs {
        struct_size: size::<PleGatherArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: PleGatherPlan::qwen38_flash(16),
        coherent_base: batch
            .device_base()
            .ok_or("PLE fixed cache is not CUDA-visible")?
            .as_ptr(),
        fragments: fragment_region.device_address() as usize as *const PleRowFragment,
        output: ptr_mut(embedding.device_address()),
        output_row_stride_bytes: 160 * 2,
        scale_bf16_bits: ple_index.scale_bf16_bits,
        reserved16: 0,
        reserved32: 0,
        cuda_stream: stream.raw(),
    };
    native("Qwen real PLE NVMe gather", unsafe {
        sparkserve_ple_gather_launch(&caps, &gather)
    })?;
    let conv_state = fixture_slab(fixture, "conv_state_bf16.bin")?;
    let key = slab(HYPER * 2)?;
    let value = slab(HIDDEN * 2)?;
    let gated = slab(HYPER * 2)?;
    let normed = slab(HYPER * 2)?;
    let output = slab(HYPER * 2)?;
    let args = QwenPleBlockArgs {
        struct_size: size::<QwenPleBlockArgs>(),
        abi_version: QWEN_PLE_BLOCK_ABI_VERSION,
        tokens: 1,
        reserved: 0,
        hidden_states: ptr(hidden.device_address()),
        embedding: ptr(embedding.device_address()),
        key_weight: ptr(key_weight.device_address()),
        value_weight: ptr(value_weight.device_address()),
        norm_key_weight: ptr(norm_key.device_address()),
        norm_query_weight: ptr(norm_query.device_address()),
        norm_conv_weight: ptr(norm_conv.device_address()),
        conv_weight: ptr(conv_weight.device_address()),
        conv_state: ptr_mut(conv_state.device_address()),
        key_scratch: ptr_mut(key.device_address()),
        value_scratch: ptr_mut(value.device_address()),
        gated_scratch: ptr_mut(gated.device_address()),
        normed_scratch: ptr_mut(normed.device_address()),
        output: ptr_mut(output.device_address()),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    native("Qwen real PLE block", unsafe {
        sparkserve_qwen_ple_block_launch(&args)
    })?;
    stream.synchronize()?;

    expect_bf16_fixture(
        &embedding,
        fixture,
        "embedding_bf16.bin",
        "PLE FP8 NVMe gather",
        0.0,
    )?;
    for (region, file, stage, tolerance) in [
        (&key, "key_normed_bf16.bin", "PLE key projection/norm", 0.0078125),
        (&value, "value_bf16.bin", "PLE value projection", 0.001),
        (&gated, "gated_bf16.bin", "PLE gate", 0.001),
        (&normed, "normed_bf16.bin", "PLE conv norm", 0.001),
        (&output, "output_bf16.bin", "PLE output", 0.001),
        (&conv_state, "next_state_bf16.bin", "PLE recurrent state", 0.001),
    ] {
        expect_bf16_fixture(region, fixture, file, stage, tolerance)?;
    }
    println!("Qwen real PLE hash -> NVMe FP8 gather -> projection -> gate -> short conv: within donor parity bounds");
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

fn read_i64(fixture: &Path, file: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(fixture.join(file))?;
    if !bytes.len().is_multiple_of(8) {
        return Err(format!("fixture {file} is not an i64 array").into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|value| i64::from_ne_bytes(value.try_into().expect("eight bytes")))
        .collect())
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
    u32::try_from(std::mem::size_of::<T>()).expect("ABI structure size fits u32")
}

fn ptr(address: u64) -> *const c_void {
    address as usize as *const c_void
}

fn ptr_mut(address: u64) -> *mut c_void {
    address as usize as *mut c_void
}

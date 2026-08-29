use std::ffi::CStr;
use std::path::Path;

use spark_flash_next::checkpoint::load_flash_next_checkpoint;
use spark_flash_next::coherent::CoherentRegionOwner;
use spark_flash_next::cuda::{CudaBlasOwner, CudaStreamOwner};
use spark_flash_next::ffi::{
    DeviceCaps, GdnBlockArgs, GdnBlockPlan, GdnDecodeArgs, GdnDecodePlan, MhcArgs,
    MhcPlan, QWEN_GDN_AUX_ABI_VERSION, QwenBf16ToF32Args, Status,
    flash_gdn_block_finish_launch, flash_gdn_block_prepare_launch,
    flash_gdn_decode_launch, flash_mhc_combine_launch,
    flash_mhc_mix_launch, flash_qwen_bf16_to_f32_launch,
};
use spark_flash_next::kernel::KERNEL_ABI_VERSION;
use spark_flash_next::qwen_weights::{FlashNextWeightMaps, QwenTensorView};

const HC: u64 = 4;
const HIDDEN: u64 = 2560;
const LOWRANK: u64 = 320;
const QK_HEADS: u64 = 16;
const VALUE_HEADS: u64 = 48;
const HEAD_DIM: u64 = 128;
const QK_WIDTH: u64 = QK_HEADS * HEAD_DIM;
const VALUE_WIDTH: u64 = VALUE_HEADS * HEAD_DIM;
const CONV_WIDTH: u64 = 2 * QK_WIDTH + VALUE_WIDTH;
const CONV_KERNEL: u64 = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_gdn_smoke <model-root> <fixture>"));
    let fixture = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_real_gdn_smoke <model-root> <fixture>"));
    let fixture = Path::new(&fixture);

    let checkpoint = load_flash_next_checkpoint(Path::new(&model))?;
    let mut weights = FlashNextWeightMaps::new(&checkpoint, 0);
    let mut stream = CudaStreamOwner::create()?;
    let blas = CudaBlasOwner::create()?;
    let caps = DeviceCaps::gb10(0);

    let prefix = "model.language_model.layers.0";
    let mhc_norm_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.attn_hyper_connection.hc_norm.weight"),
        &[HC * HIDDEN],
        HC * HIDDEN * 2,
    )?;
    let mhc_down_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.attn_hyper_connection.input_mix_weight_down.weight"),
        &[LOWRANK, HC * HIDDEN],
        LOWRANK * HC * HIDDEN * 2,
    )?;
    let mhc_up_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.attn_hyper_connection.input_mix_weight_up.weight"),
        &[HC * HIDDEN, LOWRANK],
        HC * HIDDEN * LOWRANK * 2,
    )?;
    let mhc_inject_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.attn_hyper_connection.block_inject_weight.weight"),
        &[HC, HC * HIDDEN],
        HC * HC * HIDDEN * 2,
    )?;

    let qkv_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
        &[CONV_WIDTH, HIDDEN],
        CONV_WIDTH * HIDDEN * 2,
    )?;
    let z_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.in_proj_z.weight"),
        &[VALUE_WIDTH, HIDDEN],
        VALUE_WIDTH * HIDDEN * 2,
    )?;
    let b_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.in_proj_b.weight"),
        &[VALUE_HEADS, HIDDEN],
        VALUE_HEADS * HIDDEN * 2,
    )?;
    let a_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.in_proj_a.weight"),
        &[VALUE_HEADS, HIDDEN],
        VALUE_HEADS * HIDDEN * 2,
    )?;
    let conv_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.conv1d.weight"),
        &[CONV_WIDTH, 1, CONV_KERNEL],
        CONV_WIDTH * CONV_KERNEL * 2,
    )?;
    let gated_norm_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.norm.weight"),
        &[HEAD_DIM],
        HEAD_DIM * 2,
    )?;
    let out_weight = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.out_proj.weight"),
        &[HIDDEN, VALUE_WIDTH],
        HIDDEN * VALUE_WIDTH * 2,
    )?;
    let a_log_bf16 = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.A_log"),
        &[VALUE_HEADS],
        VALUE_HEADS * 2,
    )?;
    let dt_bias_bf16 = checkpoint_weight(
        &mut weights,
        &mut stream,
        &format!("{prefix}.linear_attn.dt_bias"),
        &[VALUE_HEADS],
        VALUE_HEADS * 2,
    )?;
    let a_log = slab(VALUE_HEADS * 4)?;
    let dt_bias = slab(VALUE_HEADS * 4)?;
    convert_bf16(
        &stream,
        a_log_bf16.device_address(),
        a_log.device_address(),
        VALUE_HEADS,
        "A_log",
    )?;
    convert_bf16(
        &stream,
        dt_bias_bf16.device_address(),
        dt_bias.device_address(),
        VALUE_HEADS,
        "dt_bias",
    )?;

    let mut hyper_input = slab(HC * HIDDEN * 2)?;
    write_fixture(&mut hyper_input, fixture, "mhc_hyper_input_bf16.bin")?;
    let mut state_indices = slab(4)?;
    write_fixture(&mut state_indices, fixture, "state_indices_i32.bin")?;
    let mut conv_state = slab(CONV_WIDTH * (CONV_KERNEL - 1) * 2)?;
    write_fixture(&mut conv_state, fixture, "conv_state_before_bf16.bin")?;
    let mut temporal_state = slab(VALUE_HEADS * HEAD_DIM * HEAD_DIM * 2)?;
    write_fixture(
        &mut temporal_state,
        fixture,
        "temporal_state_before_bf16.bin",
    )?;

    let mhc_normed = slab(HC * HIDDEN * 2)?;
    let mhc_down = slab(LOWRANK * 2)?;
    let mhc_activated = slab(LOWRANK * 2)?;
    let mhc_up = slab(HC * HIDDEN * 2)?;
    let mixed = slab(HIDDEN * 2)?;
    let combined = slab(HC * HIDDEN * 2)?;
    let projected_qkv = slab(CONV_WIDTH * 2)?;
    let projected_z = slab(VALUE_WIDTH * 2)?;
    let projected_b = slab(VALUE_HEADS * 2)?;
    let projected_a = slab(VALUE_HEADS * 2)?;
    let convolved = slab(CONV_WIDTH * 2)?;
    let core = slab(VALUE_WIDTH * 2)?;
    let gated = slab(VALUE_WIDTH * 2)?;
    let attention = slab(HIDDEN * 2)?;

    let mhc = MhcArgs {
        struct_size: size::<MhcArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: MhcPlan::qwen38_flash(1),
        hyper_input: ptr(hyper_input.device_address()),
        norm_weight: ptr(mhc_norm_weight.device_address()),
        mix_down_weight: ptr(mhc_down_weight.device_address()),
        mix_up_weight: ptr(mhc_up_weight.device_address()),
        inject_weight: ptr(mhc_inject_weight.device_address()),
        block_output: ptr(attention.device_address()),
        normed: ptr_mut(mhc_normed.device_address()),
        mix_down: ptr_mut(mhc_down.device_address()),
        mix_activated: ptr_mut(mhc_activated.device_address()),
        mix_up: ptr_mut(mhc_up.device_address()),
        mixed_output: ptr_mut(mixed.device_address()),
        combined_output: ptr_mut(combined.device_address()),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    let block = GdnBlockArgs {
        struct_size: size::<GdnBlockArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: GdnBlockPlan::qwen38_flash_decode(1),
        hidden_states: ptr(mixed.device_address()),
        in_proj_qkv_weight: ptr(qkv_weight.device_address()),
        in_proj_z_weight: ptr(z_weight.device_address()),
        in_proj_b_weight: ptr(b_weight.device_address()),
        in_proj_a_weight: ptr(a_weight.device_address()),
        conv_weight: ptr(conv_weight.device_address()),
        gated_norm_weight: ptr(gated_norm_weight.device_address()),
        out_proj_weight: ptr(out_weight.device_address()),
        conv_state_pool: ptr_mut(conv_state.device_address()),
        state_indices: ptr(state_indices.device_address()),
        projected_qkv: ptr_mut(projected_qkv.device_address()),
        projected_z: ptr_mut(projected_z.device_address()),
        projected_b: ptr_mut(projected_b.device_address()),
        projected_a: ptr_mut(projected_a.device_address()),
        convolved_qkv: ptr_mut(convolved.device_address()),
        gdn_core_output: ptr(core.device_address()),
        gated_norm_output: ptr_mut(gated.device_address()),
        attention_output: ptr_mut(attention.device_address()),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    let recurrence = GdnDecodeArgs {
        struct_size: size::<GdnDecodeArgs>(),
        abi_version: KERNEL_ABI_VERSION,
        plan: GdnDecodePlan::qwen38_flash_decode(1, 1),
        q: ptr(convolved.device_address()),
        k: ptr(convolved.device_address() + QK_WIDTH * 2),
        v: ptr(convolved.device_address() + 2 * QK_WIDTH * 2),
        a: ptr(projected_a.device_address()),
        b: ptr(projected_b.device_address()),
        a_log: ptr(a_log.device_address()),
        dt_bias: ptr(dt_bias.device_address()),
        state_pool: ptr_mut(temporal_state.device_address()),
        state_indices: ptr(state_indices.device_address()),
        output: ptr_mut(core.device_address()),
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        sequence_length: 1,
        cuda_stream: stream.raw(),
    };

    native("Qwen mHC mix", unsafe {
        flash_mhc_mix_launch(&caps, &mhc)
    })?;
    native("Qwen GDN projections and convolution", unsafe {
        flash_gdn_block_prepare_launch(&caps, &block)
    })?;
    native("Qwen GDN recurrence", unsafe {
        flash_gdn_decode_launch(&caps, &recurrence)
    })?;
    native("Qwen GDN gate and output", unsafe {
        flash_gdn_block_finish_launch(&caps, &block)
    })?;
    native("Qwen mHC combine", unsafe {
        flash_mhc_combine_launch(&caps, &mhc)
    })?;
    stream.synchronize()?;

    expect_fixture(&attention, fixture, "attention_output_bf16.bin", "attention output")?;
    expect_fixture(&conv_state, fixture, "conv_state_after_bf16.bin", "convolution state")?;
    expect_fixture(
        &temporal_state,
        fixture,
        "temporal_state_after_bf16.bin",
        "recurrent state",
    )?;
    expect_fixture(&combined, fixture, "mhc_combined_bf16.bin", "mHC combined output")?;

    println!("Qwen real checkpoint mHC -> GDN -> recurrent state -> mHC combine: exact");
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
    aligned_copy(stream, tensor.device_address, tensor.data_bytes)
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

fn convert_bf16(
    stream: &CudaStreamOwner,
    input: u64,
    output: u64,
    elements: u64,
    stage: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let args = QwenBf16ToF32Args {
        struct_size: size::<QwenBf16ToF32Args>(),
        abi_version: QWEN_GDN_AUX_ABI_VERSION,
        input_bf16: ptr(input),
        output_f32: ptr_mut(output),
        elements,
        cuda_stream: stream.raw(),
    };
    native(stage, unsafe { flash_qwen_bf16_to_f32_launch(&args) })
}

fn slab(bytes: u64) -> Result<CoherentRegionOwner, Box<dyn std::error::Error>> {
    Ok(CoherentRegionOwner::slab(bytes, 256, 0)?)
}

fn write_fixture(
    region: &mut CoherentRegionOwner,
    fixture: &Path,
    file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read(fixture.join(file))?;
    let destination = unsafe { region.host_payload_mut()? };
    if destination.len() != source.len() {
        return Err(format!("fixture {file} has the wrong size").into());
    }
    destination.copy_from_slice(&source);
    Ok(())
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

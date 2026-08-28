use sparkserve_runtime::coherent::CoherentRegionOwner;
use sparkserve_runtime::cuda::{CudaBlasOwner, CudaEventOwner, CudaStreamOwner, status_result};
use sparkserve_runtime::ffi::{
    COHERENT_REGION_PREFAULT, DeviceCaps, MoeGateArgs, MoeGatePlan, sparkserve_moe_gate_launch,
};
use sparkserve_runtime::kernel::KERNEL_ABI_VERSION;
use std::ffi::c_void;
use std::io;
use std::path::{Path, PathBuf};

const TOKENS: usize = 8;
const HIDDEN: usize = 2560;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const ALIGNMENT: usize = 256;

#[derive(Default)]
struct LayoutBuilder {
    next: usize,
}

impl LayoutBuilder {
    fn field(&mut self, bytes: usize) -> io::Result<usize> {
        self.next = self
            .next
            .checked_add(ALIGNMENT - 1)
            .map(|sum| sum / ALIGNMENT * ALIGNMENT)
            .ok_or_else(|| io::Error::other("MoE gate arena alignment overflow"))?;
        let offset = self.next;
        self.next = offset
            .checked_add(bytes)
            .ok_or_else(|| io::Error::other("MoE gate arena overflow"))?;
        Ok(offset)
    }

    fn finish(self) -> io::Result<usize> {
        self.next
            .checked_add(ALIGNMENT - 1)
            .map(|sum| sum / ALIGNMENT * ALIGNMENT)
            .ok_or_else(|| io::Error::other("MoE gate arena size overflow"))
    }
}

struct Layout {
    hidden: usize,
    weight: usize,
    logits: usize,
    topk_weights: usize,
    topk_ids: usize,
    bytes: usize,
}

impl Layout {
    fn qwen() -> io::Result<Self> {
        let mut builder = LayoutBuilder::default();
        let hidden = builder.field(TOKENS * HIDDEN * 2)?;
        let weight = builder.field(EXPERTS * HIDDEN * 2)?;
        let logits = builder.field(TOKENS * EXPERTS * 2)?;
        let topk_weights = builder.field(TOKENS * TOP_K * 4)?;
        let topk_ids = builder.field(TOKENS * TOP_K * 4)?;
        let bytes = builder.finish()?;
        Ok(Self {
            hidden,
            weight,
            logits,
            topk_weights,
            topk_ids,
            bytes,
        })
    }
}

fn read_exact(root: &Path, name: &str, bytes: usize) -> io::Result<Vec<u8>> {
    let path = root.join(name);
    let payload = std::fs::read(&path)?;
    if payload.len() != bytes {
        return Err(io::Error::other(format!(
            "{} has {} bytes; expected {bytes}",
            path.display(),
            payload.len()
        )));
    }
    Ok(payload)
}

fn copy_at(output: &mut [u8], offset: usize, input: &[u8]) {
    output[offset..offset + input.len()].copy_from_slice(input);
}

fn pointer(base: u64, offset: usize) -> io::Result<*mut c_void> {
    let address = base
        .checked_add(u64::try_from(offset).map_err(io::Error::other)?)
        .ok_or_else(|| io::Error::other("MoE gate device address overflow"))?;
    Ok(usize::try_from(address).map_err(io::Error::other)? as *mut c_void)
}

fn bf16_to_float(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn main() -> io::Result<()> {
    let fixture = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("usage: moe_gate_smoke FIXTURE"))?;
    let layout = Layout::qwen()?;
    let hidden = read_exact(&fixture, "hidden_bf16.bin", TOKENS * HIDDEN * 2)?;
    let weight = read_exact(&fixture, "router_weight_bf16.bin", EXPERTS * HIDDEN * 2)?;
    let expected_logits = read_exact(&fixture, "logits_bf16.bin", TOKENS * EXPERTS * 2)?;
    let expected_weights = read_exact(&fixture, "topk_weights_f32.bin", TOKENS * TOP_K * 4)?;
    let expected_ids = read_exact(&fixture, "topk_ids_i32.bin", TOKENS * TOP_K * 4)?;

    let mut region = CoherentRegionOwner::slab(
        u64::try_from(layout.bytes).map_err(io::Error::other)?,
        4096,
        COHERENT_REGION_PREFAULT,
    )
    .map_err(io::Error::other)?;
    // SAFETY: no CUDA work references this newly allocated mapping.
    let payload = unsafe { region.host_payload_mut() }.map_err(io::Error::other)?;
    copy_at(payload, layout.hidden, &hidden);
    copy_at(payload, layout.weight, &weight);

    let blas = CudaBlasOwner::create().map_err(io::Error::other)?;
    let mut stream = CudaStreamOwner::create().map_err(io::Error::other)?;
    let mut event = CudaEventOwner::create().map_err(io::Error::other)?;
    let base = region.device_address();
    let plan = MoeGatePlan::qwen38_flash(u32::try_from(TOKENS).map_err(io::Error::other)?);
    let args = MoeGateArgs {
        struct_size: u32::try_from(std::mem::size_of::<MoeGateArgs>()).map_err(io::Error::other)?,
        abi_version: KERNEL_ABI_VERSION,
        plan,
        hidden_states: pointer(base, layout.hidden)?,
        router_weight: pointer(base, layout.weight)?,
        router_logits: pointer(base, layout.logits)?,
        topk_weights: pointer(base, layout.topk_weights)?.cast(),
        topk_ids: pointer(base, layout.topk_ids)?.cast(),
        cublas_handle: blas.raw(),
        cuda_stream: stream.raw(),
    };
    let caps = DeviceCaps::gb10(0);
    // SAFETY: all pointers address disjoint live ranges in the coherent arena;
    // the stream, cuBLAS handle, and region outlive event completion.
    status_result(unsafe { sparkserve_moe_gate_launch(&caps, &args) }).map_err(io::Error::other)?;
    event.record(&mut stream).map_err(io::Error::other)?;
    event.synchronize().map_err(io::Error::other)?;

    let payload = region.host_payload().map_err(io::Error::other)?;
    let actual_logits = &payload[layout.logits..layout.logits + expected_logits.len()];
    let actual_weights =
        &payload[layout.topk_weights..layout.topk_weights + expected_weights.len()];
    let actual_ids = &payload[layout.topk_ids..layout.topk_ids + expected_ids.len()];

    let max_logit_error = actual_logits
        .chunks_exact(2)
        .zip(expected_logits.chunks_exact(2))
        .map(|(actual, expected)| {
            let actual = bf16_to_float(u16::from_ne_bytes(actual.try_into().unwrap()));
            let expected = bf16_to_float(u16::from_ne_bytes(expected.try_into().unwrap()));
            (actual - expected).abs()
        })
        .fold(0.0_f32, f32::max);
    let max_weight_error = actual_weights
        .chunks_exact(4)
        .zip(expected_weights.chunks_exact(4))
        .map(|(actual, expected)| {
            let actual = f32::from_ne_bytes(actual.try_into().unwrap());
            let expected = f32::from_ne_bytes(expected.try_into().unwrap());
            (actual - expected).abs()
        })
        .fold(0.0_f32, f32::max);
    let id_mismatches = actual_ids
        .chunks_exact(4)
        .zip(expected_ids.chunks_exact(4))
        .filter(|(actual, expected)| actual != expected)
        .count();
    println!("Rust coherent Qwen router max BF16 error: {max_logit_error}");
    println!("Rust coherent SGLang top-k id mismatches: {id_mismatches}");
    println!("Rust coherent SGLang top-k max weight error: {max_weight_error}");
    if max_logit_error > 0.125 || max_weight_error > 1.0e-5 || id_mismatches != 0 {
        return Err(io::Error::other("Qwen MoE gate parity failed"));
    }
    Ok(())
}

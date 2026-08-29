mod checkpoint;
mod mapping;

use mapping::MappedCheckpoint;
use std::ffi::{CStr, c_char, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::env;
use std::path::Path;
use std::time::Instant;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

#[repr(C)]
struct KernelStatus {
    code: i32,
    message: *const c_char,
}

#[repr(C)]
struct Fp8Args {
    struct_size: u32,
    abi_version: u32,
    n: u32,
    k: u32,
    input_bf16: *const c_void,
    weight_fp8_e4m3: *const c_void,
    input_scale: *const f32,
    weight_scale: *const f32,
    quantized_input_fp8_e4m3: *mut c_void,
    output_bf16: *mut c_void,
    cuda_stream: *mut c_void,
}

unsafe extern "C" {
    fn q27_fp8_project(args: *const Fp8Args) -> KernelStatus;
    fn cudaMalloc(output: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaMemcpy(output: *mut c_void, input: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaFree(pointer: *mut c_void) -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
}

fn cuda(error: i32, operation: &str) {
    if error == 0 { return; }
    // SAFETY: CUDA returns a process-lifetime error string.
    let detail = unsafe { CStr::from_ptr(cudaGetErrorString(error)) }.to_string_lossy();
    panic!("{operation}: {detail}");
}

struct DeviceBuffer {
    pointer: *mut c_void,
}

impl DeviceBuffer {
    fn allocate(bytes: usize) -> Self {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: the output slot is valid and ownership transfers to this guard.
        cuda(unsafe { cudaMalloc(&mut pointer, bytes) }, "cudaMalloc");
        Self { pointer }
    }

    fn copy_from(bytes: &[u8]) -> Self {
        let buffer = Self::allocate(bytes.len());
        // SAFETY: both ranges are valid for bytes.len().
        cuda(unsafe { cudaMemcpy(buffer.pointer, bytes.as_ptr().cast(), bytes.len(), CUDA_MEMCPY_HOST_TO_DEVICE) }, "cudaMemcpy H2D");
        buffer
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: this guard owns the allocation exactly once.
        let _ = unsafe { cudaFree(self.pointer) };
    }
}

fn u32_le(input: &mut File) -> u32 {
    let mut bytes = [0_u8; 4];
    input.read_exact(&mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}

fn u64_le(input: &mut File) -> u64 {
    let mut bytes = [0_u8; 8];
    input.read_exact(&mut bytes).unwrap();
    u64::from_le_bytes(bytes)
}

fn f32_le(input: &mut File) -> f32 { f32::from_bits(u32_le(input)) }

fn read_bytes(input: &mut File, bytes: u64) -> Vec<u8> {
    let mut output = vec![0_u8; usize::try_from(bytes).unwrap()];
    input.read_exact(&mut output).unwrap();
    output
}

fn probe_mapped_fp8(mapped: &MappedCheckpoint, fixture_path: &Path) {
    let checkpoint = mapped.checkpoint();
    let base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
    let weight = checkpoint.tensor(&format!("{base}.weight")).unwrap();
    let input_scale = checkpoint.tensor(&format!("{base}.input_scale")).unwrap();
    let weight_scale = checkpoint.tensor(&format!("{base}.weight_scale")).unwrap();
    let mut fixture = File::open(fixture_path).unwrap();
    let mut magic = [0_u8; 8];
    fixture.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"Q27FP8V1");
    assert_eq!(u32_le(&mut fixture), 1);
    assert_eq!(u32_le(&mut fixture), 0);
    let n = u64_le(&mut fixture);
    let k = u64_le(&mut fixture);
    let _fixture_input_scale = f32_le(&mut fixture);
    let _fixture_weight_scale = f32_le(&mut fixture);
    let input_bytes = u64_le(&mut fixture);
    let quantized_bytes = u64_le(&mut fixture);
    let weight_bytes = u64_le(&mut fixture);
    let output_bytes = u64_le(&mut fixture);
    assert_eq!((n, k, weight_bytes), (10240, 5120, weight.data_bytes));
    let input = read_bytes(&mut fixture, input_bytes);
    let expected_quantized = read_bytes(&mut fixture, quantized_bytes);
    fixture.seek(SeekFrom::Current(i64::try_from(weight_bytes).unwrap())).unwrap();
    let expected_output = read_bytes(&mut fixture, output_bytes);
    let input_d = DeviceBuffer::copy_from(&input);
    let quantized_d = DeviceBuffer::allocate(expected_quantized.len());
    let output_d = DeviceBuffer::allocate(expected_output.len());
    let args = Fp8Args {
        struct_size: size_of::<Fp8Args>() as u32,
        abi_version: 1,
        n: n as u32,
        k: k as u32,
        input_bf16: input_d.pointer,
        weight_fp8_e4m3: mapped.device_address(weight).unwrap() as *const c_void,
        input_scale: mapped.device_address(input_scale).unwrap() as *const f32,
        weight_scale: mapped.device_address(weight_scale).unwrap() as *const f32,
        quantized_input_fp8_e4m3: quantized_d.pointer,
        output_bf16: output_d.pointer,
        cuda_stream: std::ptr::null_mut(),
    };
    let launch = || {
        // SAFETY: every pointer names a live device-visible range with the exact
        // shape required by the q27 kernel ABI.
        let result = unsafe { q27_fp8_project(&args) };
        if result.code != 0 {
            let message = if result.message.is_null() { "unknown".into() } else {
                // SAFETY: the kernel status owns a static/thread-local C string.
                unsafe { CStr::from_ptr(result.message) }.to_string_lossy()
            };
            panic!("mapped FP8 launch failed: {message}");
        }
    };
    launch();
    cuda(unsafe { cudaDeviceSynchronize() }, "mapped FP8 sync");
    let mut actual_quantized = vec![0_u8; expected_quantized.len()];
    let mut actual_output = vec![0_u8; expected_output.len()];
    cuda(unsafe { cudaMemcpy(actual_quantized.as_mut_ptr().cast(), quantized_d.pointer, actual_quantized.len(), CUDA_MEMCPY_DEVICE_TO_HOST) }, "read mapped quantized input");
    cuda(unsafe { cudaMemcpy(actual_output.as_mut_ptr().cast(), output_d.pointer, actual_output.len(), CUDA_MEMCPY_DEVICE_TO_HOST) }, "read mapped FP8 output");
    assert_eq!(actual_quantized, expected_quantized);
    assert_eq!(actual_output, expected_output);
    for _ in 0..10 { launch(); }
    cuda(unsafe { cudaDeviceSynchronize() }, "mapped FP8 warmup");
    let begin = Instant::now();
    for _ in 0..100 { launch(); }
    cuda(unsafe { cudaDeviceSynchronize() }, "mapped FP8 benchmark");
    let mean_us = begin.elapsed().as_secs_f64() * 1.0e6 / 100.0;
    println!("mapped_fp8=exact");
    println!("mapped_fp8_mean_us={mean_us:.3}");
    println!("mapped_fp8_weight_gb_s={:.2}", weight_bytes as f64 / mean_us / 1000.0);
}

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(root) = arguments.next() else {
        eprintln!("usage: {} CHECKPOINT [FP8_FIXTURE]", Path::new(&program).display());
        std::process::exit(2);
    };
    let fixture = arguments.next();
    if arguments.next().is_some() {
        eprintln!("q27-map-inspect accepts a checkpoint and optional FP8 fixture");
        std::process::exit(2);
    }
    let mapped = MappedCheckpoint::open(Path::new(&root)).unwrap_or_else(|error| {
        eprintln!("q27 checkpoint mapping failed: {error}");
        std::process::exit(1);
    });
    let checkpoint = mapped.checkpoint();
    let embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
        .expect("strict checkpoint omitted embedding");
    let layer0 = checkpoint.tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        .expect("strict checkpoint omitted layer-0 projection");
    let lm_head = checkpoint.tensor("lm_head.weight")
        .expect("strict checkpoint omitted LM head");
    println!("q27_mapping=valid");
    println!("shards={}", checkpoint.plan().files);
    println!("mapped_gib={:.3}", mapped.mapped_bytes() as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("embedding_device=0x{:x}", mapped.device_address(embedding).unwrap());
    println!("layer0_qkv_device=0x{:x}", mapped.device_address(layer0).unwrap());
    println!("lm_head_device=0x{:x}", mapped.device_address(lm_head).unwrap());
    if let Some(fixture) = fixture {
        probe_mapped_fp8(&mapped, Path::new(&fixture));
    }
}

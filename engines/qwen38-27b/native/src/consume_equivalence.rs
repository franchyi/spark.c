//! Spark-only end-to-end equivalence check for prompt-token consumption.

mod checkpoint;
mod mapping;
mod model_plan;
mod scale_sidecar;

use model_plan::{EagerWeightPlan, MODEL_ABI_VERSION, ModelWeights};
use std::env;
use std::ffi::{CStr, c_char};
use std::path::Path;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

const VOCABULARY: usize = 248_320;

#[repr(C)]
struct ModelStatus {
    code: i32,
    message: *const c_char,
}

#[repr(C)]
struct ModelOptions {
    struct_size: u32,
    abi_version: u32,
    context_capacity: u32,
    device_id: u32,
}

#[repr(C)]
struct ModelStats {
    struct_size: u32,
    abi_version: u32,
    resident_weight_bytes: u64,
    state_bytes: u64,
    scratch_bytes: u64,
    context_capacity: u32,
    position: u32,
    last_decode_us: u64,
}

#[repr(C)]
struct NativeModel {
    _private: [u8; 0],
}

#[link(name = "q27-model")]
unsafe extern "C" {
    fn q27_model_create(
        weights: *const ModelWeights,
        options: *const ModelOptions,
        output: *mut *mut NativeModel,
    ) -> ModelStatus;
    fn q27_model_reset(model: *mut NativeModel) -> ModelStatus;
    fn q27_model_consume_token(model: *mut NativeModel, token: u32) -> ModelStatus;
    fn q27_model_decode_greedy(
        model: *mut NativeModel,
        token: u32,
        output_token: *mut u32,
    ) -> ModelStatus;
    fn q27_model_copy_logits(
        model: *const NativeModel,
        host_logits: *mut f32,
        elements: u32,
    ) -> ModelStatus;
    fn q27_model_get_stats(model: *const NativeModel, output: *mut ModelStats) -> ModelStatus;
    fn q27_model_destroy(model: *mut NativeModel) -> ModelStatus;
}

fn status(result: ModelStatus) -> Result<(), String> {
    if result.code == 0 {
        return Ok(());
    }
    let detail = if result.message.is_null() {
        format!("native q27 status {}", result.code)
    } else {
        // SAFETY: copy the native status text before making another call.
        unsafe { CStr::from_ptr(result.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(detail)
}

struct ModelGuard(NonNull<NativeModel>);

impl Drop for ModelGuard {
    fn drop(&mut self) {
        // SAFETY: this guard owns the native model exactly once.
        let _ = unsafe { q27_model_destroy(self.0.as_ptr()) };
    }
}

fn reset(model: &ModelGuard) -> Result<(), String> {
    // SAFETY: the fixture owns and serializes the native model.
    status(unsafe { q27_model_reset(model.0.as_ptr()) })
}

fn stats(model: &ModelGuard) -> Result<ModelStats, String> {
    let mut output = ModelStats {
        struct_size: size_of::<ModelStats>() as u32,
        abi_version: MODEL_ABI_VERSION,
        resident_weight_bytes: 0,
        state_bytes: 0,
        scratch_bytes: 0,
        context_capacity: 0,
        position: 0,
        last_decode_us: 0,
    };
    // SAFETY: the output struct matches the versioned native ABI.
    status(unsafe { q27_model_get_stats(model.0.as_ptr(), &mut output) })?;
    Ok(output)
}

fn greedy_path(
    model: &ModelGuard,
    tokens: &[u32],
) -> Result<(u32, Vec<f32>, Duration), String> {
    reset(model)?;
    let started = Instant::now();
    let mut output = 0_u32;
    for &token in tokens {
        // SAFETY: the fixture owns and serializes the native model.
        status(unsafe {
            q27_model_decode_greedy(model.0.as_ptr(), token, &mut output)
        })?;
    }
    let elapsed = started.elapsed();
    let mut logits = vec![0.0_f32; VOCABULARY];
    // SAFETY: the output slice is live and exactly vocabulary-sized.
    status(unsafe {
        q27_model_copy_logits(model.0.as_ptr(), logits.as_mut_ptr(), VOCABULARY as u32)
    })?;
    Ok((output, logits, elapsed))
}

fn consume_path(
    model: &ModelGuard,
    tokens: &[u32],
) -> Result<(u32, Vec<f32>, Duration), String> {
    reset(model)?;
    let started = Instant::now();
    for &token in &tokens[..tokens.len() - 1] {
        // SAFETY: the fixture owns and serializes the native model.
        status(unsafe { q27_model_consume_token(model.0.as_ptr(), token) })?;
    }
    let mut output = 0_u32;
    // SAFETY: the fixture owns and serializes the native model.
    status(unsafe {
        q27_model_decode_greedy(
            model.0.as_ptr(),
            tokens[tokens.len() - 1],
            &mut output,
        )
    })?;
    let elapsed = started.elapsed();
    let mut logits = vec![0.0_f32; VOCABULARY];
    // SAFETY: the output slice is live and exactly vocabulary-sized.
    status(unsafe {
        q27_model_copy_logits(model.0.as_ptr(), logits.as_mut_ptr(), VOCABULARY as u32)
    })?;
    Ok((output, logits, elapsed))
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let usage = || {
        format!(
            "usage: {} CHECKPOINT SCALE_SIDECAR TOKEN TOKEN [TOKEN ...]",
            Path::new(&program).display()
        )
    };
    let checkpoint = arguments.next().ok_or_else(&usage)?;
    let sidecar = arguments.next().ok_or_else(&usage)?;
    let tokens = arguments
        .map(|token| {
            token
                .to_string_lossy()
                .parse::<u32>()
                .map_err(|_| usage())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.len() < 2 || tokens.iter().any(|&token| token >= VOCABULARY as u32) {
        return Err(usage());
    }

    let plan = EagerWeightPlan::open(Path::new(&checkpoint), Path::new(&sidecar))
        .map_err(|error| format!("q27 weight plan failed: {error}"))?;
    let options = ModelOptions {
        struct_size: size_of::<ModelOptions>() as u32,
        abi_version: MODEL_ABI_VERSION,
        context_capacity: tokens.len() as u32,
        device_id: 0,
    };
    let mut raw = std::ptr::null_mut();
    // SAFETY: the plan remains alive until after the native model is destroyed.
    status(unsafe { q27_model_create(plan.weights(), &options, &mut raw) })?;
    let model = ModelGuard(
        NonNull::new(raw).ok_or_else(|| "q27 model returned null".to_owned())?,
    );

    // Warm every resident path once, then prove that consume invalidates stale
    // diagnostic logits until a greedy tail has produced a new vector.
    let mut warm_output = 0_u32;
    // SAFETY: the fixture owns and serializes the native model.
    status(unsafe {
        q27_model_decode_greedy(model.0.as_ptr(), tokens[0], &mut warm_output)
    })?;
    reset(&model)?;
    // SAFETY: the fixture owns and serializes the native model.
    status(unsafe { q27_model_consume_token(model.0.as_ptr(), tokens[0]) })?;
    let consumed_stats = stats(&model)?;
    if consumed_stats.position != 1 || consumed_stats.last_decode_us != 0 {
        return Err(format!(
            "consume stats mismatch: position={} last_decode_us={}",
            consumed_stats.position, consumed_stats.last_decode_us
        ));
    }
    let mut stale = vec![0.0_f32; VOCABULARY];
    // SAFETY: this deliberately tests the rejected stale-logit call.
    if status(unsafe {
        q27_model_copy_logits(model.0.as_ptr(), stale.as_mut_ptr(), VOCABULARY as u32)
    })
    .is_ok()
    {
        return Err("q27 consume left stale diagnostic logits readable".to_owned());
    }

    let (greedy_token, greedy_logits, greedy_elapsed) = greedy_path(&model, &tokens)?;
    let (consume_token, consume_logits, consume_elapsed) = consume_path(&model, &tokens)?;
    let final_stats = stats(&model)?;
    if final_stats.position != tokens.len() as u32 || final_stats.last_decode_us == 0 {
        return Err(format!(
            "final stats mismatch: position={} last_decode_us={}",
            final_stats.position, final_stats.last_decode_us
        ));
    }
    if greedy_token != consume_token {
        return Err(format!(
            "consume token mismatch: greedy={greedy_token} consume={consume_token}"
        ));
    }
    let mismatches = greedy_logits
        .iter()
        .zip(&consume_logits)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    if mismatches != 0 {
        return Err(format!("consume logits differ at {mismatches}/{VOCABULARY} values"));
    }

    let greedy_ms = greedy_elapsed.as_secs_f64() * 1000.0;
    let consume_ms = consume_elapsed.as_secs_f64() * 1000.0;
    println!("q27_consume_equivalence=PASS");
    println!(
        "input_tokens={}",
        tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
    );
    println!("output_token={consume_token}");
    println!("logit_mismatches=0/{VOCABULARY}");
    println!("greedy_prefix_ms={greedy_ms:.3}");
    println!("consume_prefix_ms={consume_ms:.3}");
    println!(
        "saved_percent={:.3}",
        100.0 * (greedy_ms - consume_ms) / greedy_ms
    );
    drop(model);
    drop(plan);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("q27 consume equivalence failed: {error}");
        std::process::exit(1);
    }
}

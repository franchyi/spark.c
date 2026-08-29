mod checkpoint;
mod mapping;
mod model_plan;
mod scale_sidecar;

use model_plan::{EagerWeightPlan, MODEL_ABI_VERSION, ModelWeights};
use std::env;
use std::ffi::{CStr, c_char};
use std::path::Path;
use std::ptr::NonNull;
use std::time::Instant;

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
#[derive(Clone, Copy)]
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
    fn q27_model_decode_greedy(
        model: *mut NativeModel,
        token: u32,
        output_token: *mut u32,
    ) -> ModelStatus;
    fn q27_model_prefill_greedy(
        model: *mut NativeModel,
        host_tokens: *const u32,
        count: u32,
        output_token: *mut u32,
    ) -> ModelStatus;
    fn q27_model_get_stats(model: *const NativeModel, output: *mut ModelStats) -> ModelStatus;
    fn q27_model_copy_logits(
        model: *const NativeModel,
        host_logits: *mut f32,
        elements: u32,
    ) -> ModelStatus;
    fn q27_model_destroy(model: *mut NativeModel) -> ModelStatus;
}

fn status(result: ModelStatus) -> Result<(), String> {
    if result.code == 0 {
        return Ok(());
    }
    let detail = if result.message.is_null() {
        format!("native q27 status {}", result.code)
    } else {
        // SAFETY: native statuses point to static or thread-local storage and
        // are copied before another native call is made.
        unsafe { CStr::from_ptr(result.message) }.to_string_lossy().into_owned()
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

fn usage(program: &Path) -> ! {
    eprintln!(
        "usage: {} CHECKPOINT SCALE_SIDECAR TOKEN[,TOKEN...] [CONTEXT_CAPACITY] [STEPS]",
        program.display()
    );
    std::process::exit(2);
}

fn main() {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(checkpoint) = arguments.next() else { usage(Path::new(&program)) };
    let Some(sidecar) = arguments.next() else { usage(Path::new(&program)) };
    let Some(token) = arguments.next() else { usage(Path::new(&program)) };
    let capacity_argument = arguments.next();
    let steps = arguments.next().unwrap_or_else(|| "1".into());
    if arguments.next().is_some() { usage(Path::new(&program)); }
    let token_text = token.to_string_lossy();
    let batched_prefill = token_text.contains(',');
    let token_ids = token_text
        .split(',')
        .map(|value| value.parse::<u32>().unwrap_or_else(|_| usage(Path::new(&program))))
        .collect::<Vec<_>>();
    if token_ids.is_empty() { usage(Path::new(&program)); }
    let default_capacity = if batched_prefill {
        u32::try_from(token_ids.len()).unwrap_or_else(|_| usage(Path::new(&program)))
    } else {
        1
    };
    let capacity: u32 = capacity_argument
        .map(|value| value.to_string_lossy().parse().unwrap_or_else(|_| usage(Path::new(&program))))
        .unwrap_or(default_capacity);
    let steps: u32 = steps.to_string_lossy().parse().unwrap_or_else(|_| usage(Path::new(&program)));
    if steps == 0 || steps > capacity ||
        (batched_prefill && (steps != 1 || token_ids.len() > capacity as usize)) {
        usage(Path::new(&program));
    }
    let token = token_ids[0];

    let load_begin = Instant::now();
    let plan = EagerWeightPlan::open(Path::new(&checkpoint), Path::new(&sidecar)).unwrap_or_else(|error| {
        eprintln!("q27 eager weight plan failed: {error}");
        std::process::exit(1);
    });
    let options = ModelOptions {
        struct_size: size_of::<ModelOptions>() as u32,
        abi_version: MODEL_ABI_VERSION,
        context_capacity: capacity,
        device_id: 0,
    };
    let mut raw = std::ptr::null_mut();
    // SAFETY: the strict plan and its mapped payloads remain alive until after
    // the native model is destroyed.
    status(unsafe { q27_model_create(plan.weights(), &options, &mut raw) }).unwrap_or_else(|error| {
        eprintln!("q27 eager model creation failed: {error}");
        std::process::exit(1);
    });
    let model = ModelGuard(NonNull::new(raw).expect("q27 model returned null"));
    let load_seconds = load_begin.elapsed().as_secs_f64();

    if batched_prefill {
        let mut output_token = 0_u32;
        let count = u32::try_from(token_ids.len()).unwrap_or_else(|_| usage(Path::new(&program)));
        let prefill_begin = Instant::now();
        // SAFETY: the token vector remains live through this synchronous
        // native call and the model has exactly one Rust owner.
        status(unsafe {
            q27_model_prefill_greedy(
                model.0.as_ptr(),
                token_ids.as_ptr(),
                count,
                &mut output_token,
            )
        })
        .unwrap_or_else(|error| {
            eprintln!("q27 eager batched prefill failed: {error}");
            std::process::exit(1);
        });
        let prefill_seconds = prefill_begin.elapsed().as_secs_f64();
        println!("q27_eager=native");
        println!("prompt_tokens={count}");
        println!("output_token={output_token}");
        println!("load_seconds={load_seconds:.3}");
        println!("prefill_seconds={prefill_seconds:.6}");
        println!(
            "effective_prefill_tok_s={:.3}",
            f64::from(count) / prefill_seconds
        );
        return;
    }

    let mut output_token = 0;
    status(unsafe { q27_model_decode_greedy(model.0.as_ptr(), token, &mut output_token) })
        .unwrap_or_else(|error| {
            eprintln!("q27 eager decode failed: {error}");
            std::process::exit(1);
        });
    let mut first_stats = ModelStats {
        struct_size: size_of::<ModelStats>() as u32,
        abi_version: MODEL_ABI_VERSION,
        resident_weight_bytes: 0,
        state_bytes: 0,
        scratch_bytes: 0,
        context_capacity: 0,
        position: 0,
        last_decode_us: 0,
    };
    status(unsafe { q27_model_get_stats(model.0.as_ptr(), &mut first_stats) }).unwrap();
    let mut logits = vec![0.0_f32; 248_320];
    status(unsafe {
        q27_model_copy_logits(model.0.as_ptr(), logits.as_mut_ptr(), logits.len() as u32)
    }).unwrap();
    if let Some(path) = env::var_os("Q27_LOGITS_PATH") {
        // SAFETY: `logits` is a live contiguous f32 allocation and the byte
        // view is used only for this synchronous, opt-in diagnostic write.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                logits.as_ptr().cast::<u8>(),
                logits.len() * size_of::<f32>(),
            )
        };
        std::fs::write(&path, bytes).unwrap_or_else(|error| {
            eprintln!("q27 logits diagnostic write failed: {error}");
            std::process::exit(1);
        });
    }
    let mut token_ids: Vec<usize> = (0..logits.len()).collect();
    token_ids.select_nth_unstable_by(5, |left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    token_ids[..5].sort_unstable_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    let mut final_token = output_token;
    let mut final_stats = first_stats;
    let mut warm_decode_us = 0_u64;
    let mut warm_decode_times = Vec::with_capacity(steps.saturating_sub(1) as usize);
    let mut token_sequence = Vec::with_capacity(steps as usize);
    token_sequence.push(output_token);
    for _ in 1..steps {
        let input = final_token;
        status(unsafe { q27_model_decode_greedy(model.0.as_ptr(), input, &mut final_token) })
            .unwrap_or_else(|error| {
                eprintln!("q27 eager warm decode failed: {error}");
                std::process::exit(1);
            });
        status(unsafe { q27_model_get_stats(model.0.as_ptr(), &mut final_stats) }).unwrap();
        warm_decode_us += final_stats.last_decode_us;
        warm_decode_times.push(final_stats.last_decode_us);
        token_sequence.push(final_token);
    }
    println!("q27_eager=native");
    println!("input_token={token}");
    println!("output_token={output_token}");
    println!(
        "token_sequence={}",
        token_sequence.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
    );
    for (rank, token_id) in token_ids[..5].iter().enumerate() {
        println!("top{}_token={} top{}_logit={:.9}", rank + 1, token_id, rank + 1, logits[*token_id]);
    }
    println!("load_seconds={load_seconds:.3}");
    println!("decode_seconds={:.6}", first_stats.last_decode_us as f64 / 1.0e6);
    println!("resident_weight_gib={:.3}", first_stats.resident_weight_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("state_mib={:.3}", first_stats.state_bytes as f64 / 1024.0 / 1024.0);
    println!("scratch_mib={:.3}", first_stats.scratch_bytes as f64 / 1024.0 / 1024.0);
    println!("position={}", final_stats.position);
    if steps > 1 {
        println!("steps={steps}");
        println!("final_output_token={final_token}");
        println!("warm_decode_mean_seconds={:.6}", warm_decode_us as f64 / (steps - 1) as f64 / 1.0e6);
        println!("warm_decode_min_seconds={:.6}", *warm_decode_times.iter().min().unwrap() as f64 / 1.0e6);
        println!("warm_decode_max_seconds={:.6}", *warm_decode_times.iter().max().unwrap() as f64 / 1.0e6);
    }
}

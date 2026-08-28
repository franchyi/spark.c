//! Persistent batch-one Qwen decode smoke over the native engine owner.

use std::path::Path;

use sparkserve_runtime::tokenizer::NativeQwenTokenizer;

#[path = "qwen_first_token.rs"]
mod qwen_native;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_decode <model-root> [input-token-id] [steps]"));
    let mut input_token = arguments
        .next()
        .map_or(Ok(9707_u32), |value| value.parse::<u32>())?;
    let steps = arguments
        .next()
        .map_or(Ok(2_u32), |value| value.parse::<u32>())?;
    if arguments.next().is_some() || steps == 0 {
        return Err("invalid Qwen decode arguments".into());
    }

    let model_root = Path::new(&model);
    let tokenizer = NativeQwenTokenizer::from_model_root(model_root)?;
    let mut engine = qwen_native::QwenNativeEngine::create(model_root)?;
    engine.reset_sequence()?;
    let started = std::time::Instant::now();
    for position in 0..steps {
        let step = engine.forward_token(input_token, false)?;
        let text = tokenizer.decode(&[step.token], false)?;
        println!(
            "Qwen persistent position {position}: input {input_token}, output {} {text:?}, {:.3} s, experts {}/{} hit/miss, {} evictions",
            step.token,
            step.elapsed_seconds,
            step.expert_hits,
            step.expert_misses,
            step.expert_evictions,
        );
        input_token = step.token;
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "Qwen persistent decode: {steps} steps in {elapsed:.3} s ({:.3} token/s)",
        f64::from(steps) / elapsed,
    );
    Ok(())
}

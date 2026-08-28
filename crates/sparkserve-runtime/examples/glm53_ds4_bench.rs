use std::env;
use std::fs;
use std::time::Instant;

use sparkserve_runtime::ds4_glm53::{Engine, EngineConfig};

const CONTEXT_TOKENS: usize = 2048;
const GENERATION_TOKENS: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: glm53_ds4_bench MODEL PROMPT_FILE")?;
    let prompt_path = args
        .next()
        .ok_or("usage: glm53_ds4_bench MODEL PROMPT_FILE")?;
    let prompt = fs::read_to_string(prompt_path)?;

    let engine = Engine::open(EngineConfig::resident(&model, 2304))?;
    let mut prompt_tokens = engine.encode_text(&prompt)?;
    if prompt_tokens.len() < CONTEXT_TOKENS {
        return Err(format!(
            "benchmark input has only {} tokens, need {CONTEXT_TOKENS}",
            prompt_tokens.len()
        )
        .into());
    }
    prompt_tokens.truncate(CONTEXT_TOKENS);

    let mut session = engine.session()?;
    let prefill_start = Instant::now();
    session.sync(&prompt_tokens)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let mut rng = 1u64;
    let generation_start = Instant::now();
    let mut first_ms = 0.0;
    let mut steady_seconds = 0.0;
    let mut generated = Vec::with_capacity(GENERATION_TOKENS);
    for index in 0..GENERATION_TOKENS {
        let token = session.sample(0.0, 0, 1.0, 0.0, &mut rng)?;
        generated.push(token);
        let token_start = Instant::now();
        session.eval(token)?;
        let token_seconds = token_start.elapsed().as_secs_f64();
        if index == 0 {
            first_ms = token_seconds * 1000.0;
        } else {
            steady_seconds += token_seconds;
        }
    }
    let generation_seconds = generation_start.elapsed().as_secs_f64();

    println!(
        "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps"
    );
    println!(
        "{CONTEXT_TOKENS},{CONTEXT_TOKENS},{:.2},{GENERATION_TOKENS},{:.2},{:.3},{},{:.2}",
        CONTEXT_TOKENS as f64 / prefill_seconds,
        GENERATION_TOKENS as f64 / generation_seconds,
        first_ms,
        GENERATION_TOKENS - 1,
        (GENERATION_TOKENS - 1) as f64 / steady_seconds.max(f64::EPSILON),
    );
    eprintln!("sparkserve: token_ids={generated:?}");
    Ok(())
}

use std::env;
use std::io::{self, Write};
use std::time::Instant;

use sparkserve_runtime::ds4_glm53::{ChatMessage, Engine, EngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: glm53_ds4_first_token MODEL [PROMPT] [TOKENS]")?;
    let prompt = args
        .next()
        .unwrap_or_else(|| "Write one short sentence about DGX Spark.".to_owned());
    let predict = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);

    let load_start = Instant::now();
    let engine = Engine::open(EngineConfig::resident(&model, 4096))?;
    eprintln!(
        "sparkserve: loaded {} in {:.3}s",
        engine.model_name(),
        load_start.elapsed().as_secs_f64()
    );

    let prompt_tokens = engine.encode_messages(
        &[
            ChatMessage::new("system", "You are a helpful assistant"),
            ChatMessage::new("user", &prompt),
        ],
        true,
    )?;
    let mut session = engine.session()?;
    let prefill_start = Instant::now();
    session.sync(&prompt_tokens)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    eprintln!(
        "sparkserve: prefill tokens={} seconds={:.6} tok/s={:.3}",
        prompt_tokens.len(),
        prefill_seconds,
        prompt_tokens.len() as f64 / prefill_seconds
    );

    let mut rng = 1u64;
    let mut decode_seconds = 0.0;
    let mut evaluated = 0usize;
    let mut generated = Vec::new();
    let mut stdout = io::stdout().lock();
    for index in 0..predict {
        let token = session.sample(0.0, 0, 1.0, 0.0, &mut rng)?;
        generated.push(token);
        stdout.write_all(&engine.token_piece(token)?)?;
        stdout.flush()?;
        if engine.is_stop_token(token) || index + 1 == predict {
            break;
        }
        let decode_start = Instant::now();
        session.eval(token)?;
        decode_seconds += decode_start.elapsed().as_secs_f64();
        evaluated += 1;
    }
    writeln!(stdout)?;
    if evaluated > 0 {
        eprintln!(
            "sparkserve: decode evaluated={} seconds={:.6} tok/s={:.3} position={}",
            evaluated,
            decode_seconds,
            evaluated as f64 / decode_seconds,
            session.position()
        );
    }
    eprintln!("sparkserve: token_ids={generated:?}");
    Ok(())
}

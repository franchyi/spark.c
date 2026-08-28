use std::env;
use std::sync::Arc;

use sparkserve_runtime::ds4_glm53::{Engine, EngineConfig, Generator};
use sparkserve_runtime::openai_server::{OpenAiServer, TokenGenerator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: glm53_ds4_serve MODEL [BIND] [CONTEXT]")?;
    let bind = args.next().unwrap_or_else(|| "127.0.0.1:8010".to_owned());
    let context = args
        .next()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(16_384);

    let engine = Engine::open(EngineConfig::resident(&model, context))?;
    let tokenizer = Arc::new(engine.clone());
    let generator = Arc::new(Generator::new(engine)?);
    eprintln!(
        "sparkserve: serving model={} bind={} context={}",
        generator.model_id(),
        bind,
        context
    );
    OpenAiServer::new(tokenizer, generator).serve(&bind)?;
    Ok(())
}

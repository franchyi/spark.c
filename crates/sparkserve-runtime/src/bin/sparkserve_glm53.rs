use std::env;
use std::sync::Arc;

use sparkserve_runtime::ds4_glm53::{Engine, EngineConfig, Generator};
use sparkserve_runtime::openai_server::{OpenAiServer, TokenGenerator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model = args.next().ok_or(
        "usage: sparkserve-glm53 MODEL [--bind HOST:PORT] [--context TOKENS] [--mtp]",
    )?;
    let mut bind = "127.0.0.1:8010".to_owned();
    let mut context = 16_384u32;
    let mut flags = 0u32;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--bind" => {
                bind = args.next().ok_or("--bind requires HOST:PORT")?;
            }
            "--context" => {
                context = args
                    .next()
                    .ok_or("--context requires a token count")?
                    .parse()?;
            }
            "--mtp" => flags |= sparkserve_runtime::ds4_glm53::MTP,
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }

    let mut config = EngineConfig::resident(&model, context);
    config.flags = flags;
    let engine = Engine::open(config)?;
    let tokenizer = Arc::new(engine.clone());
    let generator = Arc::new(Generator::new(engine)?);
    eprintln!(
        "sparkserve: serving model={} bind={} context={} mtp={}",
        generator.model_id(),
        bind,
        context,
        flags & sparkserve_runtime::ds4_glm53::MTP != 0
    );
    OpenAiServer::new(tokenizer, generator).serve(&bind)?;
    Ok(())
}

//! OpenAI-compatible Qwen3.8 Flash-Next service over one native GPU owner.

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use spark_flash_next::engine::QwenNativeEngine;
use spark_flash_next::openai_server::{
    FinishReason, GenerationError, GenerationRequest, OpenAiServer, TokenGenerator,
};
use spark_flash_next::tokenizer::NativeQwenTokenizer;

enum EngineEvent {
    Token(u32),
    Finished(FinishReason),
    Failed(String),
}

struct EngineCommand {
    request: GenerationRequest,
    events: mpsc::Sender<EngineEvent>,
}

/// HTTP workers are free to move between CPU threads; all CUDA and coherent
/// memory ownership stays inside the one engine thread created below.
struct QwenThreadBackend {
    model_id: String,
    commands: mpsc::SyncSender<EngineCommand>,
}

impl QwenThreadBackend {
    fn start(model_root: PathBuf, model_id: String) -> Result<Self, GenerationError> {
        let (commands, receiver) = mpsc::sync_channel::<EngineCommand>(1);
        let (ready_sender, ready_receiver) = mpsc::sync_channel::<Result<(), String>>(1);
        thread::Builder::new()
            .name("flash-qwen-gpu".to_owned())
            .spawn(move || {
                let mut engine = match QwenNativeEngine::create(&model_root) {
                    Ok(engine) => {
                        let _ = ready_sender.send(Ok(()));
                        engine
                    }
                    Err(error) => {
                        let _ = ready_sender.send(Err(error.to_string()));
                        return;
                    }
                };
                while let Ok(command) = receiver.recv() {
                    run_generation(&mut engine, command);
                }
            })
            .map_err(|error| GenerationError::new(format!("cannot start GPU owner: {error}")))?;
        ready_receiver
            .recv()
            .map_err(|_| GenerationError::new("GPU owner exited during initialization"))?
            .map_err(GenerationError::new)?;
        Ok(Self { model_id, commands })
    }
}

impl TokenGenerator for QwenThreadBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn generate(
        &self,
        request: GenerationRequest,
        emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
    ) -> Result<FinishReason, GenerationError> {
        let (events, receiver) = mpsc::channel();
        self.commands
            .send(EngineCommand { request, events })
            .map_err(|_| GenerationError::new("Qwen GPU owner is unavailable"))?;
        loop {
            match receiver.recv() {
                Ok(EngineEvent::Token(token)) => emit(token)?,
                Ok(EngineEvent::Finished(reason)) => return Ok(reason),
                Ok(EngineEvent::Failed(message)) => return Err(GenerationError::new(message)),
                Err(_) => return Err(GenerationError::new("Qwen GPU owner exited")),
            }
        }
    }
}

fn run_generation(engine: &mut QwenNativeEngine, command: EngineCommand) {
    let EngineCommand { request, events } = command;
    let trace_layers = std::env::var_os("FLASH_QWEN_TRACE_LAYERS").is_some();
    // Profile one steady-state decode token without synchronizing every stage
    // of prompt prefill. This keeps the diagnostic canary short and prevents
    // tracing itself from turning a multi-token prompt into minutes of work.
    let profile_decode = std::env::var_os("FLASH_QWEN_PROFILE_DECODE").is_some();
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        if request.prompt_token_ids.is_empty() {
            return Err("Qwen prompt cannot be empty".into());
        }
        // The current native head returns greedy argmax. Refuse to pretend that
        // temperature/top-p sampling is active until the sampler owns logits.
        if request.temperature != 0.0 || request.top_p != 1.0 {
            return Err("native Qwen service currently requires temperature=0 and top_p=1".into());
        }
        engine.reset_sequence()?;

        let prompt_tokens = request.prompt_token_ids.len();
        eprintln!(
            "Qwen request: {prompt_tokens} prompt tokens, {} maximum completion tokens",
            request.max_new_tokens,
        );
        let prefill_started = Instant::now();
        let mut candidate = None;
        let prompt = request.prompt_token_ids;
        let mut offset = 0_usize;
        while offset < prompt.len() {
            let remaining = prompt.len() - offset;
            let chunk_tokens = [16_usize, 8, 4, 2, 1]
                .into_iter()
                .find(|bucket| *bucket <= remaining)
                .ok_or("Qwen prefill bucket decomposition failed")?;
            let end = offset + chunk_tokens;
            if end == prompt.len() {
                let step = engine.forward_tokens(&prompt[offset..end], trace_layers)?;
                candidate = Some(step.token);
                eprintln!(
                    "Qwen prefill {end}/{prompt_tokens}: T={chunk_tokens}, next {}, {:.3} s, experts {}/{} hit/pack, {} evictions",
                    step.token,
                    step.elapsed_seconds,
                    step.expert_hits,
                    step.expert_misses,
                    step.expert_evictions,
                );
            } else {
                let step = engine.prefill_tokens(&prompt[offset..end], trace_layers)?;
                // Per-bucket logging is useful while tracing but turns a 12K
                // prompt into almost 800 synchronous stderr writes. Keep the
                // normal service progress coarse and leave full detail opt-in.
                if trace_layers || end.is_multiple_of(1024) {
                    eprintln!(
                        "Qwen prefill {end}/{prompt_tokens}: T={chunk_tokens}, {:.3} s, experts {}/{} hit/pack, {} evictions",
                        step.elapsed_seconds,
                        step.expert_hits,
                        step.expert_misses,
                        step.expert_evictions,
                    );
                }
            }
            offset = end;
        }
        eprintln!(
            "Qwen prefill complete in {:.3} s",
            prefill_started.elapsed().as_secs_f64()
        );
        let mut token = candidate.ok_or("Qwen prompt did not produce a decode token")?;
        for generated in 0..request.max_new_tokens {
            if request.stop_token_ids.contains(&token) {
                let _ = events.send(EngineEvent::Finished(FinishReason::Stop));
                return Ok(());
            }
            if events.send(EngineEvent::Token(token)).is_err() {
                return Ok(());
            }
            if generated + 1 == request.max_new_tokens {
                let _ = events.send(EngineEvent::Finished(FinishReason::Length));
                return Ok(());
            }
            let step = engine
                .forward_token(token, trace_layers || (profile_decode && generated == 0))?;
            eprintln!(
                "Qwen decode {}: input {token}, next {}, {:.3} s, experts {}/{} hit/pack, {} evictions",
                generated + 1,
                step.token,
                step.elapsed_seconds,
                step.expert_hits,
                step.expert_misses,
                step.expert_evictions,
            );
            token = step.token;
        }
        let _ = events.send(EngineEvent::Finished(FinishReason::Length));
        Ok(())
    })();
    if let Err(error) = result {
        let _ = events.send(EngineEvent::Failed(error.to_string()));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model_root = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen_serve <model-root> [bind] [model-id]"));
    let bind = arguments
        .next()
        .unwrap_or_else(|| "0.0.0.0:30000".to_owned());
    let model_id = arguments
        .next()
        .unwrap_or_else(|| "RadixArk/Qwen3.8-Flash-Next-NVFP4".to_owned());
    if arguments.next().is_some() {
        return Err("invalid qwen_serve arguments".into());
    }

    let model_root = Path::new(&model_root);
    let tokenizer = Arc::new(NativeQwenTokenizer::from_model_root(model_root)?);
    let backend = Arc::new(QwenThreadBackend::start(
        model_root.to_path_buf(),
        model_id.clone(),
    )?);
    println!("Flash {model_id} ready on http://{bind}");
    OpenAiServer::new(tokenizer, backend).serve(&bind)?;
    Ok(())
}

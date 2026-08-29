//! Model-specific OpenAI service for the native Qwen3.8-27B capsule.
//!
//! This process owns exactly one long-lived native model and one serialized
//! generation queue. Prompt prefill is the native decode-only MVP: after a
//! reset, each prompt token is fed in order and the final prompt step produces
//! the first completion token. There is no Python or framework runtime.

mod checkpoint;
mod mapping;
mod model_plan;
mod scale_sidecar;

// Reuse only the repository's lightweight, pinned HTTP/tokenizer boundary.
// This binary does not depend on or link the general sparkserve-runtime crate.
#[path = "../../../../crates/sparkserve-runtime/src/openai_server.rs"]
mod openai_server;
#[path = "../../../../crates/sparkserve-runtime/src/tokenizer.rs"]
mod tokenizer;

use model_plan::{EagerWeightPlan, MODEL_ABI_VERSION, ModelWeights};
use openai_server::{
    FinishReason, GenerationError, GenerationRequest, OpenAiServer, TokenGenerator,
};
use std::env;
use std::error::Error;
use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;
use tokenizer::NativeQwenTokenizer;

const DEFAULT_BIND: &str = "0.0.0.0:30000";
const DEFAULT_MODEL_ID: &str = "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead";
const DEFAULT_CONTEXT_CAPACITY: u32 = 4096;

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
    fn q27_model_decode_greedy(
        model: *mut NativeModel,
        token: u32,
        output_token: *mut u32,
    ) -> ModelStatus;
    fn q27_model_destroy(model: *mut NativeModel) -> ModelStatus;
}

fn native_status(result: ModelStatus) -> Result<(), String> {
    if result.code == 0 {
        return Ok(());
    }
    let detail = if result.message.is_null() {
        format!("native q27 status {}", result.code)
    } else {
        // SAFETY: q27 status messages remain valid until the next native call;
        // copy the text before returning across the Rust boundary.
        unsafe { CStr::from_ptr(result.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(detail)
}

struct ModelGuard(NonNull<NativeModel>);

impl Drop for ModelGuard {
    fn drop(&mut self) {
        // SAFETY: this owner thread creates and destroys the native model once.
        let _ = unsafe { q27_model_destroy(self.0.as_ptr()) };
    }
}

enum EngineEvent {
    Token(u32),
    Finished(FinishReason),
    Failed(String),
}

struct EngineCommand {
    request: GenerationRequest,
    events: mpsc::Sender<EngineEvent>,
}

/// Clone-free backend handle shared by the small HTTP worker threads. CUDA,
/// mmaps, recurrent state, and KV cache never leave the single owner thread.
struct Q27Backend {
    model_id: String,
    commands: mpsc::SyncSender<EngineCommand>,
}

impl Q27Backend {
    fn start(
        checkpoint: PathBuf,
        sidecar: PathBuf,
        model_id: String,
        context_capacity: u32,
    ) -> Result<Self, GenerationError> {
        let (commands, receiver) = mpsc::sync_channel::<EngineCommand>(1);
        let (ready_sender, ready_receiver) = mpsc::sync_channel::<Result<(), String>>(1);
        thread::Builder::new()
            .name("q27-gpu-owner".to_owned())
            .spawn(move || {
                let plan = match EagerWeightPlan::open(&checkpoint, &sidecar) {
                    Ok(plan) => plan,
                    Err(error) => {
                        let _ = ready_sender.send(Err(format!(
                            "q27 weight plan initialization failed: {error}"
                        )));
                        return;
                    }
                };
                let options = ModelOptions {
                    struct_size: size_of::<ModelOptions>() as u32,
                    abi_version: MODEL_ABI_VERSION,
                    context_capacity,
                    device_id: 0,
                };
                let mut raw = std::ptr::null_mut();
                // SAFETY: the strict mapped plan remains alive in this thread
                // until after ModelGuard destroys the native model.
                if let Err(error) = native_status(unsafe {
                    q27_model_create(plan.weights(), &options, &mut raw)
                }) {
                    let _ = ready_sender.send(Err(format!(
                        "q27 native model initialization failed: {error}"
                    )));
                    return;
                }
                let Some(model) = NonNull::new(raw) else {
                    let _ = ready_sender.send(Err(
                        "q27 native model initialization returned null".to_owned(),
                    ));
                    return;
                };
                let model = ModelGuard(model);
                if ready_sender.send(Ok(())).is_err() {
                    return;
                }

                while let Ok(command) = receiver.recv() {
                    run_generation(&model, context_capacity, command);
                }
                drop(model);
                drop(plan);
            })
            .map_err(|error| GenerationError::new(format!("cannot start q27 owner: {error}")))?;

        ready_receiver
            .recv()
            .map_err(|_| GenerationError::new("q27 owner exited during initialization"))?
            .map_err(GenerationError::new)?;
        Ok(Self { model_id, commands })
    }
}

impl TokenGenerator for Q27Backend {
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
            .map_err(|_| GenerationError::new("q27 owner is unavailable"))?;
        loop {
            match receiver.recv() {
                Ok(EngineEvent::Token(token)) => emit(token)?,
                Ok(EngineEvent::Finished(reason)) => return Ok(reason),
                Ok(EngineEvent::Failed(message)) => {
                    return Err(GenerationError::new(message));
                }
                Err(_) => return Err(GenerationError::new("q27 owner exited")),
            }
        }
    }
}

fn run_generation(model: &ModelGuard, context_capacity: u32, command: EngineCommand) {
    let EngineCommand { request, events } = command;
    let prompt_tokens = request.prompt_token_ids.len();
    let max_new_tokens = request.max_new_tokens;
    let started = Instant::now();
    let result = drive_greedy(
        request,
        context_capacity,
        || {
            // SAFETY: all calls are serialized on the model's owner thread.
            native_status(unsafe { q27_model_reset(model.0.as_ptr()) })
        },
        |token| {
            let mut output = 0_u32;
            // SAFETY: all calls are serialized on the model's owner thread.
            native_status(unsafe {
                q27_model_decode_greedy(model.0.as_ptr(), token, &mut output)
            })?;
            Ok(output)
        },
        |token| {
            events
                .send(EngineEvent::Token(token))
                .map_err(|_| "q27 request receiver closed".to_owned())
        },
    );
    match result {
        Ok(reason) => {
            eprintln!(
                "q27 request prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens} elapsed_seconds={:.3} finish={reason:?}",
                started.elapsed().as_secs_f64(),
            );
            let _ = events.send(EngineEvent::Finished(reason));
        }
        Err(error) => {
            eprintln!("q27 request failed after {:.3}s: {error}", started.elapsed().as_secs_f64());
            let _ = events.send(EngineEvent::Failed(error));
        }
    }
}

/// Pure scheduling seam: unit tests exercise prompt/decode/stop semantics on
/// Spark without loading the checkpoint or touching a GPU.
fn drive_greedy(
    request: GenerationRequest,
    context_capacity: u32,
    mut reset: impl FnMut() -> Result<(), String>,
    mut decode: impl FnMut(u32) -> Result<u32, String>,
    mut emit: impl FnMut(u32) -> Result<(), String>,
) -> Result<FinishReason, String> {
    if request.prompt_token_ids.is_empty() {
        return Err("q27 prompt cannot be empty".to_owned());
    }
    if request.temperature != 0.0 || request.top_p != 1.0 {
        return Err("q27 native service requires temperature=0 and top_p=1".to_owned());
    }
    let required = request
        .prompt_token_ids
        .len()
        .checked_add(request.max_new_tokens as usize)
        .ok_or_else(|| "q27 context length overflow".to_owned())?;
    if required > context_capacity as usize {
        return Err(format!(
            "q27 request requires {required} tokens but this service slot has capacity {context_capacity}"
        ));
    }

    reset()?;
    let mut candidate = 0_u32;
    for token in request.prompt_token_ids {
        candidate = decode(token)?;
    }
    for generated in 0..request.max_new_tokens {
        if request.stop_token_ids.contains(&candidate) {
            return Ok(FinishReason::Stop);
        }
        emit(candidate)?;
        if generated + 1 == request.max_new_tokens {
            return Ok(FinishReason::Length);
        }
        candidate = decode(candidate)?;
    }
    Ok(FinishReason::Length)
}

fn usage(program: &Path) -> String {
    format!(
        "usage: {} CHECKPOINT SCALE_SIDECAR [BIND] [MODEL_ID] [CONTEXT_CAPACITY]",
        program.display()
    )
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let checkpoint = arguments
        .next()
        .ok_or_else(|| usage(Path::new(&program)))?;
    let sidecar = arguments
        .next()
        .ok_or_else(|| usage(Path::new(&program)))?;
    let bind = arguments
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| DEFAULT_BIND.to_owned());
    let model_id = arguments
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_owned());
    let context_capacity = arguments
        .next()
        .map(|value| value.to_string_lossy().parse::<u32>())
        .transpose()
        .map_err(|_| "CONTEXT_CAPACITY must be an unsigned integer")?
        .unwrap_or(DEFAULT_CONTEXT_CAPACITY);
    if context_capacity == 0 || arguments.next().is_some() {
        return Err(usage(Path::new(&program)).into());
    }

    let checkpoint = PathBuf::from(checkpoint);
    let tokenizer = Arc::new(NativeQwenTokenizer::from_model_root(&checkpoint)?);
    let backend = Arc::new(Q27Backend::start(
        checkpoint,
        PathBuf::from(sidecar),
        model_id.clone(),
        context_capacity,
    )?);
    println!(
        "q27 service ready model={model_id} bind=http://{bind} context_capacity={context_capacity} slot_count=1"
    );
    OpenAiServer::new(tokenizer, backend).serve(&bind)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("q27 service failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(prompt: &[u32], maximum: u32, stops: &[u32]) -> GenerationRequest {
        GenerationRequest {
            prompt_token_ids: prompt.to_vec(),
            max_new_tokens: maximum,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
            stop_token_ids: stops.to_vec(),
        }
    }

    #[test]
    fn sequential_prefill_returns_final_prompt_prediction_first() {
        let mut decoded_inputs = Vec::new();
        let mut emitted = Vec::new();
        let mut resets = 0;
        let finish = drive_greedy(
            request(&[10, 20], 3, &[]),
            8,
            || {
                resets += 1;
                Ok(())
            },
            |token| {
                decoded_inputs.push(token);
                Ok(token + 1)
            },
            |token| {
                emitted.push(token);
                Ok(())
            },
        )
        .expect("greedy sequence");
        assert_eq!(resets, 1);
        assert_eq!(decoded_inputs, [10, 20, 21, 22]);
        assert_eq!(emitted, [21, 22, 23]);
        assert_eq!(finish, FinishReason::Length);
    }

    #[test]
    fn stop_token_is_not_emitted() {
        let mut emitted = Vec::new();
        let finish = drive_greedy(
            request(&[10, 20], 3, &[21]),
            8,
            || Ok(()),
            |token| Ok(token + 1),
            |token| {
                emitted.push(token);
                Ok(())
            },
        )
        .expect("stop");
        assert!(emitted.is_empty());
        assert_eq!(finish, FinishReason::Stop);
    }

    #[test]
    fn rejects_sampling_and_slot_overflow_before_reset() {
        let mut sampling = request(&[1], 1, &[]);
        sampling.temperature = 0.5;
        let error = drive_greedy(sampling, 2, || panic!("must not reset"), |_| Ok(2), |_| Ok(()))
            .expect_err("sampling rejection");
        assert!(error.contains("temperature=0"));

        let error = drive_greedy(
            request(&[1, 2], 2, &[]),
            3,
            || panic!("must not reset"),
            |_| Ok(2),
            |_| Ok(()),
        )
        .expect_err("capacity rejection");
        assert!(error.contains("requires 4 tokens"));
    }
}

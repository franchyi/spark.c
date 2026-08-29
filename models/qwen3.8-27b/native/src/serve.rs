//! Model-specific OpenAI service for the native Qwen3.8-27B capsule.
//!
//! This process owns exactly one long-lived native model and one serialized
//! generation queue. Prompt prefill advances recurrent/KV state without an LM
//! head for every non-final prompt token; the final prompt step produces the
//! first completion token. There is no Python or framework runtime.

mod checkpoint;
mod mapping;
mod model_plan;
mod scale_sidecar;

// Reuse only the repository's lightweight, pinned HTTP/tokenizer boundary.
// This binary does not depend on or link the Flash-Next native crate.
#[path = "../../../../common/openai.rs"]
mod openai_server;
#[path = "../../../../common/qwen_tokenizer.rs"]
mod tokenizer;

use model_plan::{EagerWeightPlan, MODEL_ABI_VERSION, ModelWeights};
use openai_server::{
    FinishReason, GenerationError, GenerationRequest, OpenAiServer, TokenGenerator,
};
use serde::Serialize;
use std::env;
use std::error::Error;
use std::ffi::{CStr, c_char};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;
use tokenizer::NativeQwenTokenizer;

const DEFAULT_BIND: &str = "0.0.0.0:30000";
const DEFAULT_MODEL_ID: &str = "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead";
const DEFAULT_CONTEXT_CAPACITY: u32 = 4096;
const TOKEN_TRACE_PATH_ENV: &str = "SPARK_ENGINE_TOKEN_TRACE_PATH";
const TOKEN_TRACE_MAX_RECORDS: u64 = 1024;
const TOKEN_TRACE_MAX_BYTES: u64 = 64 * 1024 * 1024;

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
    fn q27_model_consume_token(model: *mut NativeModel, token: u32) -> ModelStatus;
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
    events: mpsc::SyncSender<EngineEvent>,
}

#[derive(Serialize)]
struct TokenTraceRecord<'a> {
    schema: &'static str,
    record_id: u64,
    prompt_token_ids: &'a [u32],
    generated_token_ids: &'a [u32],
    finish_reason: &'static str,
    terminal_stop_token_id: Option<u32>,
}

struct TokenTrace {
    output: File,
    records_written: u64,
    bytes_written: u64,
    max_records: u64,
    max_bytes: u64,
    disabled: bool,
}

impl TokenTrace {
    fn open(path: &Path) -> Result<Self, String> {
        Self::open_with_limits(path, TOKEN_TRACE_MAX_RECORDS, TOKEN_TRACE_MAX_BYTES)
    }

    fn open_with_limits(path: &Path, max_records: u64, max_bytes: u64) -> Result<Self, String> {
        let output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .map_err(|error| format!("cannot open q27 token trace {}: {error}", path.display()))?;
        Ok(Self {
            output,
            records_written: 0,
            bytes_written: 0,
            max_records,
            max_bytes,
            disabled: false,
        })
    }

    fn write_completed(
        &mut self,
        prompt_token_ids: &[u32],
        generated_token_ids: &[u32],
        outcome: GreedyOutcome,
    ) -> Result<(), String> {
        if self.disabled {
            return Err("q27 token trace is disabled after an earlier failure".to_owned());
        }
        if self.records_written >= self.max_records {
            self.disabled = true;
            return Err(format!(
                "q27 token trace reached its {}-record limit",
                self.max_records
            ));
        }
        let line = format_token_trace(
            self.records_written,
            prompt_token_ids,
            generated_token_ids,
            outcome,
        )
            .map_err(|error| format!("cannot encode q27 token trace: {error}"))?;
        let mut record = line.into_bytes();
        record.push(b'\n');
        let record_bytes = record.len() as u64;
        let Some(total_bytes) = self.bytes_written.checked_add(record_bytes) else {
            self.disabled = true;
            return Err("q27 token trace byte count overflow".to_owned());
        };
        if total_bytes > self.max_bytes {
            self.disabled = true;
            return Err(format!(
                "q27 token trace reached its {}-byte limit",
                self.max_bytes
            ));
        }
        if let Err(error) = self
            .output
            .write_all(&record)
            .and_then(|()| self.output.sync_data())
        {
            // A failed write may have left a partial final line. Never append
            // another record after that point, so later JSON cannot be folded
            // into the damaged line and mistaken for a valid request.
            self.disabled = true;
            return Err(format!("cannot write q27 token trace: {error}"));
        }
        self.bytes_written = total_bytes;
        self.records_written += 1;
        Ok(())
    }
}

fn format_token_trace(
    record_id: u64,
    prompt_token_ids: &[u32],
    generated_token_ids: &[u32],
    outcome: GreedyOutcome,
) -> Result<String, serde_json::Error> {
    serde_json::to_string(&TokenTraceRecord {
        schema: "q27.q27.token-trace.v1",
        record_id,
        prompt_token_ids,
        generated_token_ids,
        finish_reason: match outcome.finish_reason {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        },
        terminal_stop_token_id: outcome.terminal_stop_token_id,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GreedyOutcome {
    finish_reason: FinishReason,
    terminal_stop_token_id: Option<u32>,
}

/// Clone-free backend handle shared by the small HTTP worker threads. CUDA,
/// mmaps, recurrent state, and KV cache never leave the single owner thread.
struct Q27Backend {
    model_id: String,
    context_capacity: u32,
    commands: mpsc::SyncSender<EngineCommand>,
}

impl Q27Backend {
    fn start(
        checkpoint: PathBuf,
        sidecar: PathBuf,
        model_id: String,
        context_capacity: u32,
        token_trace_path: Option<PathBuf>,
    ) -> Result<Self, GenerationError> {
        let (commands, receiver) = mpsc::sync_channel::<EngineCommand>(1);
        let (ready_sender, ready_receiver) = mpsc::sync_channel::<Result<(), String>>(1);
        thread::Builder::new()
            .name("q27-gpu-owner".to_owned())
            .spawn(move || {
                // The sole GPU-owner thread is also the sole trace writer.
                // When the opt-in path is absent, no file is opened and no
                // prompt or generated token IDs are retained.
                let mut token_trace = match token_trace_path
                    .as_deref()
                    .map(TokenTrace::open)
                    .transpose()
                {
                    Ok(trace) => trace,
                    Err(error) => {
                        let _ = ready_sender.send(Err(error));
                        return;
                    }
                };
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
                    run_generation(
                        &model,
                        context_capacity,
                        command,
                        token_trace.as_mut(),
                    );
                }
                drop(model);
                drop(plan);
            })
            .map_err(|error| GenerationError::new(format!("cannot start q27 owner: {error}")))?;

        ready_receiver
            .recv()
            .map_err(|_| GenerationError::new("q27 owner exited during initialization"))?
            .map_err(GenerationError::new)?;
        Ok(Self {
            model_id,
            context_capacity,
            commands,
        })
    }
}

impl TokenGenerator for Q27Backend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn default_temperature(&self) -> f32 {
        0.0
    }

    fn max_in_flight_requests(&self) -> usize {
        2
    }

    fn validate_generation(&self, request: &GenerationRequest) -> Result<(), GenerationError> {
        if request.temperature != 0.0 || request.top_p != 1.0 {
            return Err(GenerationError::new(
                "q27 native service requires temperature=0 and top_p=1",
            ));
        }
        let required = request
            .prompt_token_ids
            .len()
            .checked_add(request.max_new_tokens as usize)
            .ok_or_else(|| GenerationError::new("q27 context length overflow"))?;
        if required > self.context_capacity as usize {
            return Err(GenerationError::new(format!(
                "q27 request requires {required} tokens but this service slot has capacity {}",
                self.context_capacity
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        request: GenerationRequest,
        emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
    ) -> Result<FinishReason, GenerationError> {
        // Bound generated-token buffering so a slow/disconnected HTTP client
        // cannot make the single-slot engine accumulate an unbounded queue.
        let (events, receiver) = mpsc::sync_channel(1);
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

fn run_generation(
    model: &ModelGuard,
    context_capacity: u32,
    command: EngineCommand,
    mut token_trace: Option<&mut TokenTrace>,
) {
    let EngineCommand { request, events } = command;
    let prompt_tokens = request.prompt_token_ids.len();
    let max_new_tokens = request.max_new_tokens;
    let trace_enabled = token_trace
        .as_ref()
        .is_some_and(|trace| !trace.disabled);
    let prompt_token_ids = trace_enabled.then(|| request.prompt_token_ids.clone());
    let mut generated_token_ids = Vec::new();
    let started = Instant::now();
    let result = drive_greedy(
        request,
        context_capacity,
        || {
            // SAFETY: all calls are serialized on the model's owner thread.
            native_status(unsafe { q27_model_reset(model.0.as_ptr()) })
        },
        |token| {
            // SAFETY: all calls are serialized on the model's owner thread.
            native_status(unsafe { q27_model_consume_token(model.0.as_ptr(), token) })
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
            if trace_enabled {
                generated_token_ids.push(token);
            }
            events
                .send(EngineEvent::Token(token))
                .map_err(|_| "q27 request receiver closed".to_owned())
        },
    );
    match result {
        Ok(outcome) => {
            // This record means native model generation completed. It does
            // not claim that the HTTP client received the final response.
            if let (Some(trace), Some(prompt_token_ids)) =
                (token_trace.as_mut(), prompt_token_ids.as_deref())
            {
                if let Err(error) = trace.write_completed(
                    prompt_token_ids,
                    &generated_token_ids,
                    outcome,
                )
                {
                    eprintln!("q27 token trace failed: {error}");
                }
            }
            let reason = outcome.finish_reason;
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
    mut consume: impl FnMut(u32) -> Result<(), String>,
    mut decode: impl FnMut(u32) -> Result<u32, String>,
    mut emit: impl FnMut(u32) -> Result<(), String>,
) -> Result<GreedyOutcome, String> {
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
    let (final_prompt, prompt_prefix) = request
        .prompt_token_ids
        .split_last()
        .expect("empty prompt was rejected above");
    for &token in prompt_prefix {
        consume(token)?;
    }
    let mut candidate = decode(*final_prompt)?;
    for generated in 0..request.max_new_tokens {
        if request.stop_token_ids.contains(&candidate) {
            return Ok(GreedyOutcome {
                finish_reason: FinishReason::Stop,
                terminal_stop_token_id: Some(candidate),
            });
        }
        emit(candidate)?;
        if generated + 1 == request.max_new_tokens {
            return Ok(GreedyOutcome {
                finish_reason: FinishReason::Length,
                terminal_stop_token_id: None,
            });
        }
        candidate = decode(candidate)?;
    }
    Ok(GreedyOutcome {
        finish_reason: FinishReason::Length,
        terminal_stop_token_id: None,
    })
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
    let token_trace_path = env::var_os(TOKEN_TRACE_PATH_ENV)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty());
    let tokenizer = Arc::new(NativeQwenTokenizer::from_model_root(&checkpoint)?);
    let backend = Arc::new(Q27Backend::start(
        checkpoint,
        PathBuf::from(sidecar),
        model_id.clone(),
        context_capacity,
        token_trace_path,
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
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TRACE_PATH: AtomicU64 = AtomicU64::new(0);

    fn trace_path(label: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "q27-{label}-{}-{}.jsonl",
            std::process::id(),
            NEXT_TRACE_PATH.fetch_add(1, Ordering::Relaxed),
        ))
    }

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
        let mut consumed_inputs = Vec::new();
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
                consumed_inputs.push(token);
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
        assert_eq!(consumed_inputs, [10]);
        assert_eq!(decoded_inputs, [20, 21, 22]);
        assert_eq!(emitted, [21, 22, 23]);
        assert_eq!(finish.finish_reason, FinishReason::Length);
        assert_eq!(finish.terminal_stop_token_id, None);
    }

    #[test]
    fn stop_token_is_not_emitted() {
        let mut emitted = Vec::new();
        let finish = drive_greedy(
            request(&[10, 20], 3, &[21]),
            8,
            || Ok(()),
            |_| Ok(()),
            |token| Ok(token + 1),
            |token| {
                emitted.push(token);
                Ok(())
            },
        )
        .expect("stop");
        assert!(emitted.is_empty());
        assert_eq!(
            finish,
            GreedyOutcome {
                finish_reason: FinishReason::Stop,
                terminal_stop_token_id: Some(21),
            }
        );
    }

    #[test]
    fn rejects_sampling_and_slot_overflow_before_reset() {
        let mut sampling = request(&[1], 1, &[]);
        sampling.temperature = 0.5;
        let error = drive_greedy(
            sampling,
            2,
            || panic!("must not reset"),
            |_| panic!("must not consume"),
            |_| Ok(2),
            |_| Ok(()),
        )
        .expect_err("sampling rejection");
        assert!(error.contains("temperature=0"));

        let error = drive_greedy(
            request(&[1, 2], 2, &[]),
            3,
            || panic!("must not reset"),
            |_| panic!("must not consume"),
            |_| Ok(2),
            |_| Ok(()),
        )
        .expect_err("capacity rejection");
        assert!(error.contains("requires 4 tokens"));
    }

    #[test]
    fn backend_validation_rejects_unsupported_work_before_admission() {
        let (commands, _receiver) = mpsc::sync_channel(1);
        let backend = Q27Backend {
            model_id: "q27-test".to_owned(),
            context_capacity: 3,
            commands,
        };
        let mut sampling = request(&[1], 1, &[]);
        sampling.top_p = 0.9;
        assert!(
            backend
                .validate_generation(&sampling)
                .expect_err("sampling rejection")
                .to_string()
                .contains("top_p=1")
        );
        assert!(
            backend
                .validate_generation(&request(&[1, 2], 2, &[]))
                .expect_err("capacity rejection")
                .to_string()
                .contains("requires 4 tokens")
        );
        backend
            .validate_generation(&request(&[1, 2], 1, &[]))
            .expect("request at capacity");
    }

    #[test]
    fn consume_failure_stops_before_final_prompt_decode() {
        let mut decoded = false;
        let mut emitted = false;
        let error = drive_greedy(
            request(&[10, 20, 30], 1, &[]),
            8,
            || Ok(()),
            |token| {
                if token == 20 {
                    Err("consume failed".to_owned())
                } else {
                    Ok(())
                }
            },
            |_| {
                decoded = true;
                Ok(31)
            },
            |_| {
                emitted = true;
                Ok(())
            },
        )
        .expect_err("consume failure");
        assert_eq!(error, "consume failed");
        assert!(!decoded);
        assert!(!emitted);
    }

    #[test]
    fn token_trace_is_one_stable_jsonl_record_without_prompt_text() {
        let line = format_token_trace(
            7,
            &[248_045, 9707, 151_644],
            &[8678, 198],
            GreedyOutcome {
                finish_reason: FinishReason::Length,
                terminal_stop_token_id: None,
            },
        )
        .expect("token trace");
        assert_eq!(
            line,
            r#"{"schema":"q27.q27.token-trace.v1","record_id":7,"prompt_token_ids":[248045,9707,151644],"generated_token_ids":[8678,198],"finish_reason":"length","terminal_stop_token_id":null}"#
        );
        assert!(!line.contains("content"));
        assert!(!line.contains('\n'));

        let stopped = format_token_trace(
            8,
            &[248_045],
            &[],
            GreedyOutcome {
                finish_reason: FinishReason::Stop,
                terminal_stop_token_id: Some(248_046),
            },
        )
        .expect("stop trace");
        assert_eq!(
            stopped,
            r#"{"schema":"q27.q27.token-trace.v1","record_id":8,"prompt_token_ids":[248045],"generated_token_ids":[],"finish_reason":"stop","terminal_stop_token_id":248046}"#
        );
    }

    #[test]
    fn token_trace_is_private_refuses_existing_and_stops_at_record_limit() {
        let path = trace_path("private");
        let mut trace = TokenTrace::open_with_limits(&path, 2, 4096).expect("private trace");
        assert_eq!(
            fs::metadata(&path)
                .expect("trace metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert!(TokenTrace::open(&path).is_err());

        let outcome = GreedyOutcome {
            finish_reason: FinishReason::Length,
            terminal_stop_token_id: None,
        };
        trace
            .write_completed(&[1, 2], &[3], outcome)
            .expect("first record");
        trace
            .write_completed(
                &[4],
                &[],
                GreedyOutcome {
                    finish_reason: FinishReason::Stop,
                    terminal_stop_token_id: Some(248_046),
                },
            )
            .expect("second record");
        assert!(trace.write_completed(&[5], &[6], outcome).is_err());
        assert!(trace.disabled);
        let payload = fs::read_to_string(&path).expect("trace payload");
        let records = payload
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSONL record"))
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["record_id"], 0);
        assert_eq!(records[1]["record_id"], 1);
        assert_eq!(records[0]["terminal_stop_token_id"], serde_json::Value::Null);
        assert_eq!(records[1]["terminal_stop_token_id"], 248_046);
        drop(trace);
        fs::remove_file(path).expect("remove trace");
    }

    #[test]
    fn token_trace_stops_before_exceeding_byte_limit() {
        let path = trace_path("bytes");
        let mut trace = TokenTrace::open_with_limits(&path, 10, 1).expect("bounded trace");
        let error = trace
            .write_completed(
                &[1],
                &[2],
                GreedyOutcome {
                    finish_reason: FinishReason::Length,
                    terminal_stop_token_id: None,
                },
            )
            .expect_err("byte limit");
        assert!(error.contains("1-byte limit"));
        assert!(trace.disabled);
        assert_eq!(fs::metadata(&path).expect("trace metadata").len(), 0);
        drop(trace);
        fs::remove_file(path).expect("remove trace");
    }
}

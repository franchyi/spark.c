//! Lightweight OpenAI service for the fixed Qwen3.8-27B DFlash2 capsule.
//!
//! The Rust process owns checkpoint mappings, HTTP/tokenization, admission,
//! and one bounded command queue. One native owner thread exclusively owns
//! the target plus draft CUDA engine. There is no framework runtime, dynamic
//! batching, sampling, or generic speculative-decoding scheduler.

mod checkpoint;
mod dflash2_checkpoint;
mod mapping;
mod model_plan;
mod scale_sidecar;

#[path = "../../../../common/openai.rs"]
mod openai_server;
#[path = "../../../../common/qwen_tokenizer.rs"]
mod tokenizer;

use dflash2_checkpoint::{DFlash2WeightPlan, DFlash2Weights};
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

const DFLASH2_ENGINE_ABI_VERSION: u32 = 1;
const DFLASH2_PROFILE_ABI_VERSION: u32 = 1;
const DFLASH2_BLOCK_SIZE: usize = 8;
const DFLASH2_DRAFT_TOKENS: u32 = 7;
const DFLASH2_MAX_POSITION: u32 = 262_144;
const DFLASH2_VOCABULARY: u32 = 248_320;
const DEFAULT_BIND: &str = "0.0.0.0:30000";
const DEFAULT_MODEL_ID: &str = "spark/Qwen3.8-27B-DFlash2";
const DEFAULT_CONTEXT_CAPACITY: u32 = 16_384;

#[repr(C)]
struct DFlash2Status {
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
struct NativeDFlash2Engine {
    _private: [u8; 0],
}

#[repr(C)]
struct DFlash2EngineCreateArgs {
    struct_size: u32,
    abi_version: u32,
    target_weights: *const ModelWeights,
    target_options: *const ModelOptions,
    draft_weights: *const DFlash2Weights,
    rms_epsilon: f32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DFlash2BlockResult {
    struct_size: u32,
    abi_version: u32,
    base_position: u32,
    new_position: u32,
    anchor_token: u32,
    accepted_draft_tokens: u32,
    emitted_count: u32,
    bonus_token: u32,
    proposed_tokens: [u32; DFLASH2_BLOCK_SIZE],
    target_top1: [u32; DFLASH2_BLOCK_SIZE],
    emitted_tokens: [u32; DFLASH2_BLOCK_SIZE],
    elapsed_us: u64,
}

impl DFlash2BlockResult {
    fn empty() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: DFLASH2_ENGINE_ABI_VERSION,
            base_position: 0,
            new_position: 0,
            anchor_token: 0,
            accepted_draft_tokens: 0,
            emitted_count: 0,
            bonus_token: 0,
            proposed_tokens: [0; DFLASH2_BLOCK_SIZE],
            target_top1: [0; DFLASH2_BLOCK_SIZE],
            emitted_tokens: [0; DFLASH2_BLOCK_SIZE],
            elapsed_us: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DFlash2EngineStats {
    struct_size: u32,
    abi_version: u32,
    target_resident_weight_bytes: u64,
    draft_checkpoint_weight_bytes: u64,
    state_bytes: u64,
    scratch_bytes: u64,
    prompt_tokens: u64,
    verify_calls: u64,
    proposed_draft_tokens: u64,
    accepted_draft_tokens: u64,
    emitted_tokens: u64,
    last_prefill_us: u64,
    last_block_us: u64,
    context_capacity: u32,
    position: u32,
    next_anchor_token: u32,
    ready_to_decode: u32,
}

impl DFlash2EngineStats {
    fn empty() -> Self {
        // SAFETY: this C aggregate contains only integer fields. The native
        // API requires the leading size/version fields on input.
        let mut stats: Self = unsafe { std::mem::zeroed() };
        stats.struct_size = size_of::<Self>() as u32;
        stats.abi_version = DFLASH2_ENGINE_ABI_VERSION;
        stats
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DFlash2ProfileStats {
    struct_size: u32,
    abi_version: u32,
    draft_prepare_embed_us: u64,
    draft_forward_us: u64,
    lm_head_top16_us: u64,
    selector_us: u64,
    proposal_copy_sync_us: u64,
    target_verify_total_us: u64,
    target_snapshot_us: u64,
    target_speculative_pass_us: u64,
    target_speculative_result_sync_us: u64,
    target_rollback_us: u64,
    target_committed_replay_us: u64,
    target_committed_result_sync_us: u64,
    enabled: u32,
    valid: u32,
}

impl DFlash2ProfileStats {
    fn empty() -> Self {
        // SAFETY: this C aggregate contains only integer fields.
        let mut stats: Self = unsafe { std::mem::zeroed() };
        stats.struct_size = size_of::<Self>() as u32;
        stats.abi_version = DFLASH2_PROFILE_ABI_VERSION;
        stats
    }
}

#[link(name = "q27-dflash2-engine")]
unsafe extern "C" {
    fn q27_dflash2_engine_create(
        args: *const DFlash2EngineCreateArgs,
        output: *mut *mut NativeDFlash2Engine,
    ) -> DFlash2Status;
    fn q27_dflash2_engine_reset(engine: *mut NativeDFlash2Engine) -> DFlash2Status;
    fn q27_dflash2_engine_prefill(
        engine: *mut NativeDFlash2Engine,
        host_tokens: *const u32,
        count: u32,
        first_token: *mut u32,
    ) -> DFlash2Status;
    fn q27_dflash2_engine_decode_block(
        engine: *mut NativeDFlash2Engine,
        output: *mut DFlash2BlockResult,
    ) -> DFlash2Status;
    fn q27_dflash2_engine_get_stats(
        engine: *const NativeDFlash2Engine,
        output: *mut DFlash2EngineStats,
    ) -> DFlash2Status;
    fn q27_dflash2_engine_get_profile_stats(
        engine: *const NativeDFlash2Engine,
        output: *mut DFlash2ProfileStats,
    ) -> DFlash2Status;
    fn q27_dflash2_engine_destroy(engine: *mut NativeDFlash2Engine) -> DFlash2Status;
}

fn native_status(result: DFlash2Status) -> Result<(), String> {
    if result.code == 0 {
        return Ok(());
    }
    let detail = if result.message.is_null() {
        format!("native DFlash2 status {}", result.code)
    } else {
        // SAFETY: the native ABI keeps this thread-local status string valid
        // until the next call. Copy it before crossing the Rust boundary.
        unsafe { CStr::from_ptr(result.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(detail)
}

struct EngineGuard(NonNull<NativeDFlash2Engine>);

impl Drop for EngineGuard {
    fn drop(&mut self) {
        // SAFETY: creation, all calls, and destruction occur on this thread.
        let _ = unsafe { q27_dflash2_engine_destroy(self.0.as_ptr()) };
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DFlash2Outcome {
    finish_reason: FinishReason,
    streamed_tokens: u32,
    verify_calls: u32,
}

/// Clone-free HTTP handle. All native pointers and mutable model state remain
/// confined to the sole owner thread behind this one-command queue.
struct DFlash2Backend {
    model_id: String,
    context_capacity: u32,
    commands: mpsc::SyncSender<EngineCommand>,
}

impl DFlash2Backend {
    fn start(
        target_checkpoint: PathBuf,
        target_sidecar: PathBuf,
        draft_checkpoint: PathBuf,
        model_id: String,
        context_capacity: u32,
    ) -> Result<Self, GenerationError> {
        let (commands, receiver) = mpsc::sync_channel::<EngineCommand>(1);
        let (ready_sender, ready_receiver) = mpsc::sync_channel::<Result<(), String>>(1);
        thread::Builder::new()
            .name("q27-dflash2-gpu-owner".to_owned())
            .spawn(move || {
                let target_plan = match EagerWeightPlan::open(&target_checkpoint, &target_sidecar) {
                    Ok(plan) => plan,
                    Err(error) => {
                        let _ = ready_sender.send(Err(format!(
                            "q27 target weight plan initialization failed: {error}"
                        )));
                        return;
                    }
                };
                let draft_plan = match DFlash2WeightPlan::open(&draft_checkpoint) {
                    Ok(plan) => plan,
                    Err(error) => {
                        let _ = ready_sender.send(Err(format!(
                            "q27 DFlash2 weight plan initialization failed: {error}"
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
                let create = DFlash2EngineCreateArgs {
                    struct_size: size_of::<DFlash2EngineCreateArgs>() as u32,
                    abi_version: DFLASH2_ENGINE_ABI_VERSION,
                    target_weights: target_plan.weights(),
                    target_options: &options,
                    draft_weights: draft_plan.weights(),
                    rms_epsilon: 1.0e-6,
                    reserved: 0,
                };
                let mut raw = std::ptr::null_mut();
                // SAFETY: both strict mapping plans outlive the engine and
                // retain all device-visible checkpoint pointers.
                if let Err(error) = native_status(unsafe {
                    q27_dflash2_engine_create(&create, &mut raw)
                }) {
                    let _ = ready_sender.send(Err(format!(
                        "q27 DFlash2 engine initialization failed: {error}"
                    )));
                    return;
                }
                let Some(engine) = NonNull::new(raw).map(EngineGuard) else {
                    let _ = ready_sender.send(Err(
                        "q27 DFlash2 engine initialization returned null".to_owned(),
                    ));
                    return;
                };
                if ready_sender.send(Ok(())).is_err() {
                    return;
                }

                while let Ok(command) = receiver.recv() {
                    run_generation(&engine, context_capacity, command);
                }
                // Make the pointer lifetime order explicit: destroy the CUDA
                // owner before unmapping either target or draft checkpoint.
                drop(engine);
                drop(draft_plan);
                drop(target_plan);
            })
            .map_err(|error| {
                GenerationError::new(format!("cannot start q27 DFlash2 owner: {error}"))
            })?;

        ready_receiver
            .recv()
            .map_err(|_| GenerationError::new("q27 DFlash2 owner exited during initialization"))?
            .map_err(GenerationError::new)?;
        Ok(Self {
            model_id,
            context_capacity,
            commands,
        })
    }
}

impl TokenGenerator for DFlash2Backend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn default_temperature(&self) -> f32 {
        0.0
    }

    fn max_in_flight_requests(&self) -> usize {
        // One active native slot plus one bounded queued HTTP request.
        2
    }

    fn validate_generation(&self, request: &GenerationRequest) -> Result<(), GenerationError> {
        validate_request(request, self.context_capacity).map_err(GenerationError::new)
    }

    fn generate(
        &self,
        request: GenerationRequest,
        emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
    ) -> Result<FinishReason, GenerationError> {
        // A disconnected or slow SSE client backpressures the sole engine at
        // one token instead of accumulating an unbounded response.
        let (events, receiver) = mpsc::sync_channel(1);
        self.commands
            .send(EngineCommand { request, events })
            .map_err(|_| GenerationError::new("q27 DFlash2 owner is unavailable"))?;
        loop {
            match receiver.recv() {
                Ok(EngineEvent::Token(token)) => emit(token)?,
                Ok(EngineEvent::Finished(reason)) => return Ok(reason),
                Ok(EngineEvent::Failed(message)) => return Err(GenerationError::new(message)),
                Err(_) => return Err(GenerationError::new("q27 DFlash2 owner exited")),
            }
        }
    }
}

fn required_context_rows(prompt_tokens: usize, max_new_tokens: u32) -> Result<usize, String> {
    if max_new_tokens <= 1 {
        return Ok(prompt_tokens);
    }
    // Prefill predicts and emits the first anchor without consuming a target
    // row. Every later decode may make only one useful token yet requires a
    // full eight-row verification reservation. Before the last possible call,
    // at most max_new_tokens-2 rows have been committed.
    prompt_tokens
        .checked_add(max_new_tokens as usize - 1)
        .and_then(|rows| rows.checked_add(DFLASH2_BLOCK_SIZE - 1))
        .ok_or_else(|| "q27 DFlash2 context length overflow".to_owned())
}

fn validate_request(request: &GenerationRequest, context_capacity: u32) -> Result<(), String> {
    if request.prompt_token_ids.is_empty() {
        return Err("q27 DFlash2 prompt cannot be empty".to_owned());
    }
    if request.max_new_tokens == 0 {
        return Err("q27 DFlash2 max_new_tokens must be greater than zero".to_owned());
    }
    if request.temperature != 0.0 || request.top_p != 1.0 {
        return Err("q27 DFlash2 service requires temperature=0 and top_p=1".to_owned());
    }
    if request.prompt_token_ids.iter().any(|&token| token >= DFLASH2_VOCABULARY) {
        return Err("q27 DFlash2 prompt contains an out-of-range token".to_owned());
    }
    let required = required_context_rows(request.prompt_token_ids.len(), request.max_new_tokens)?;
    if required > context_capacity as usize {
        return Err(format!(
            "q27 DFlash2 request reserves {required} target rows but this service slot has capacity {context_capacity}"
        ));
    }
    Ok(())
}

fn run_generation(engine: &EngineGuard, context_capacity: u32, command: EngineCommand) {
    let EngineCommand { request, events } = command;
    let prompt_tokens = request.prompt_token_ids.len();
    let max_new_tokens = request.max_new_tokens;
    let started = Instant::now();
    let profile_requested =
        env::var("Q27_DFLASH2_PROFILE").ok().as_deref() == Some("1");
    let result = drive_dflash2(
        request,
        context_capacity,
        |tokens| {
            let count = u32::try_from(tokens.len())
                .map_err(|_| "q27 DFlash2 prompt token count exceeds u32".to_owned())?;
            let mut first_token = 0_u32;
            // SAFETY: this owner thread serializes every engine call. Native
            // prefill resets target and draft state before consuming prompt.
            native_status(unsafe {
                q27_dflash2_engine_prefill(
                    engine.0.as_ptr(),
                    tokens.as_ptr(),
                    count,
                    &mut first_token,
                )
            })?;
            Ok(first_token)
        },
        || {
            let mut block = DFlash2BlockResult::empty();
            // SAFETY: output is a correctly sized host aggregate and this is
            // the sole native owner thread.
            native_status(unsafe {
                q27_dflash2_engine_decode_block(engine.0.as_ptr(), &mut block)
            })?;
            Ok(block)
        },
        |token| {
            events
                .send(EngineEvent::Token(token))
                .map_err(|_| "q27 DFlash2 request receiver closed".to_owned())
        },
    );

    let mut stats = DFlash2EngineStats::empty();
    let stats_result = native_status(unsafe {
        q27_dflash2_engine_get_stats(engine.0.as_ptr(), &mut stats)
    });
    match (&result, &stats_result) {
        (Ok(outcome), Ok(())) => {
            let acceptance = if stats.proposed_draft_tokens == 0 {
                0.0
            } else {
                stats.accepted_draft_tokens as f64 / stats.proposed_draft_tokens as f64
            };
            eprintln!(
                "q27 DFlash2 request prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens} streamed_tokens={} finish={:?} scheduler_verify_calls={} engine_verify_calls={} proposed={} accepted={} acceptance={acceptance:.6} engine_emitted={} position={} prefill_ms={:.3} last_block_ms={:.3} elapsed_seconds={:.3}",
                outcome.streamed_tokens,
                outcome.finish_reason,
                outcome.verify_calls,
                stats.verify_calls,
                stats.proposed_draft_tokens,
                stats.accepted_draft_tokens,
                stats.emitted_tokens,
                stats.position,
                stats.last_prefill_us as f64 / 1_000.0,
                stats.last_block_us as f64 / 1_000.0,
                started.elapsed().as_secs_f64(),
            );
        }
        (Ok(_), Err(error)) => {
            eprintln!("q27 DFlash2 stats failed: {error}");
        }
        (Err(error), _) => {
            eprintln!(
                "q27 DFlash2 request failed after {:.3}s: {error}",
                started.elapsed().as_secs_f64()
            );
        }
    }

    if profile_requested {
        let mut profile = DFlash2ProfileStats::empty();
        match native_status(unsafe {
            q27_dflash2_engine_get_profile_stats(engine.0.as_ptr(), &mut profile)
        }) {
            Ok(()) if profile.enabled != 0 && profile.valid != 0 => {
                eprintln!(
                    "q27_dflash2_profile draft_prepare_embed_ms={:.3} draft_forward_ms={:.3} lm_head_top16_ms={:.3} selector_ms={:.3} proposal_copy_sync_ms={:.3} target_verify_total_ms={:.3} target_snapshot_ms={:.3} target_speculative_pass_ms={:.3} target_speculative_result_sync_ms={:.3} target_rollback_ms={:.3} target_committed_replay_ms={:.3} target_committed_result_sync_ms={:.3}",
                    profile.draft_prepare_embed_us as f64 / 1_000.0,
                    profile.draft_forward_us as f64 / 1_000.0,
                    profile.lm_head_top16_us as f64 / 1_000.0,
                    profile.selector_us as f64 / 1_000.0,
                    profile.proposal_copy_sync_us as f64 / 1_000.0,
                    profile.target_verify_total_us as f64 / 1_000.0,
                    profile.target_snapshot_us as f64 / 1_000.0,
                    profile.target_speculative_pass_us as f64 / 1_000.0,
                    profile.target_speculative_result_sync_us as f64 / 1_000.0,
                    profile.target_rollback_us as f64 / 1_000.0,
                    profile.target_committed_replay_us as f64 / 1_000.0,
                    profile.target_committed_result_sync_us as f64 / 1_000.0,
                );
            }
            Ok(()) => eprintln!(
                "q27_dflash2_profile enabled={} valid={}",
                profile.enabled, profile.valid
            ),
            Err(error) => eprintln!("q27 DFlash2 profile stats failed: {error}"),
        }
    }

    match result {
        Ok(outcome) => {
            let _ = events.send(EngineEvent::Finished(outcome.finish_reason));
        }
        Err(error) => {
            // Prefill resets at every request boundary. Recover immediately
            // after cancellation/native failure so no partial block remains
            // live while this single-slot service is idle.
            let recovery = native_status(unsafe { q27_dflash2_engine_reset(engine.0.as_ptr()) });
            let message = match recovery {
                Ok(()) => error,
                Err(reset) => format!("{error}; DFlash2 reset failed: {reset}"),
            };
            let _ = events.send(EngineEvent::Failed(message));
        }
    }
}

fn validate_block(
    block: &DFlash2BlockResult,
    expected_position: u32,
    expected_anchor: u32,
) -> Result<usize, String> {
    if block.struct_size != size_of::<DFlash2BlockResult>() as u32
        || block.abi_version != DFLASH2_ENGINE_ABI_VERSION
    {
        return Err("q27 DFlash2 engine returned an incompatible block ABI".to_owned());
    }
    let count = usize::try_from(block.emitted_count)
        .map_err(|_| "q27 DFlash2 emitted count conversion failed".to_owned())?;
    let Some(expected_new_position) = block.base_position.checked_add(block.emitted_count) else {
        return Err("q27 DFlash2 engine returned an overflowing position".to_owned());
    };
    if !(1..=DFLASH2_BLOCK_SIZE).contains(&count)
        || block.accepted_draft_tokens > DFLASH2_DRAFT_TOKENS
        || block.emitted_count != block.accepted_draft_tokens + 1
        || block.base_position != expected_position
        || block.anchor_token != expected_anchor
        || block.proposed_tokens[0] != expected_anchor
        || block.new_position != expected_new_position
        || block.bonus_token != block.emitted_tokens[count - 1]
        || block.bonus_token != block.target_top1[block.accepted_draft_tokens as usize]
        || block.emitted_tokens[..block.accepted_draft_tokens as usize]
            != block.proposed_tokens[1..count]
        || block.emitted_tokens[count..].iter().any(|&token| token != 0)
        || block.proposed_tokens.iter().any(|&token| token >= DFLASH2_VOCABULARY)
        || block.target_top1.iter().any(|&token| token >= DFLASH2_VOCABULARY)
        || block.emitted_tokens[..count]
            .iter()
            .any(|&token| token >= DFLASH2_VOCABULARY)
    {
        return Err("q27 DFlash2 engine returned invalid block metadata".to_owned());
    }
    Ok(count)
}

/// Pure scheduling seam kept independent of HTTP and CUDA ownership.
fn drive_dflash2(
    request: GenerationRequest,
    context_capacity: u32,
    mut prefill: impl FnMut(&[u32]) -> Result<u32, String>,
    mut decode_block: impl FnMut() -> Result<DFlash2BlockResult, String>,
    mut emit: impl FnMut(u32) -> Result<(), String>,
) -> Result<DFlash2Outcome, String> {
    validate_request(&request, context_capacity)?;
    let mut anchor = prefill(&request.prompt_token_ids)?;
    if anchor >= DFLASH2_VOCABULARY {
        return Err("q27 DFlash2 prefill returned an out-of-range anchor".to_owned());
    }
    if request.stop_token_ids.contains(&anchor) {
        return Ok(DFlash2Outcome {
            finish_reason: FinishReason::Stop,
            streamed_tokens: 0,
            verify_calls: 0,
        });
    }
    emit(anchor)?;
    let mut streamed = 1_u32;
    let mut verify_calls = 0_u32;
    let mut position = u32::try_from(request.prompt_token_ids.len())
        .map_err(|_| "q27 DFlash2 prompt token count exceeds u32".to_owned())?;
    if streamed == request.max_new_tokens {
        return Ok(DFlash2Outcome {
            finish_reason: FinishReason::Length,
            streamed_tokens: streamed,
            verify_calls,
        });
    }

    loop {
        let block = decode_block()?;
        verify_calls = verify_calls
            .checked_add(1)
            .ok_or_else(|| "q27 DFlash2 verify count overflow".to_owned())?;
        let count = validate_block(&block, position, anchor)?;
        position = block.new_position;
        anchor = block.bonus_token;
        for &token in &block.emitted_tokens[..count] {
            if request.stop_token_ids.contains(&token) {
                return Ok(DFlash2Outcome {
                    finish_reason: FinishReason::Stop,
                    streamed_tokens: streamed,
                    verify_calls,
                });
            }
            emit(token)?;
            streamed += 1;
            if streamed == request.max_new_tokens {
                return Ok(DFlash2Outcome {
                    finish_reason: FinishReason::Length,
                    streamed_tokens: streamed,
                    verify_calls,
                });
            }
        }
    }
}

fn usage(program: &Path) -> String {
    format!(
        "usage: {} TARGET_CHECKPOINT SCALE_SIDECAR DFLASH2_CHECKPOINT [BIND] [MODEL_ID] [CONTEXT_CAPACITY]",
        program.display()
    )
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let target_checkpoint = arguments
        .next()
        .ok_or_else(|| usage(Path::new(&program)))?;
    let sidecar = arguments
        .next()
        .ok_or_else(|| usage(Path::new(&program)))?;
    let draft_checkpoint = arguments
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
    if context_capacity < DFLASH2_BLOCK_SIZE as u32
        || context_capacity > DFLASH2_MAX_POSITION
        || arguments.next().is_some()
    {
        return Err(usage(Path::new(&program)).into());
    }

    let target_checkpoint = PathBuf::from(target_checkpoint);
    let tokenizer = Arc::new(NativeQwenTokenizer::from_model_root(&target_checkpoint)?);
    let backend = Arc::new(DFlash2Backend::start(
        target_checkpoint,
        PathBuf::from(sidecar),
        PathBuf::from(draft_checkpoint),
        model_id.clone(),
        context_capacity,
    )?);
    println!(
        "q27 DFlash2 service ready model={model_id} bind=http://{bind} context_capacity={context_capacity} slot_count=1 block_size={DFLASH2_BLOCK_SIZE}"
    );
    OpenAiServer::new(tokenizer, backend).serve(&bind)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("q27 DFlash2 service failed: {error}");
        std::process::exit(1);
    }
}

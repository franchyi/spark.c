//! Lightweight OpenAI-compatible HTTP and SSE boundary.
//!
//! `tiny_http` owns HTTP parsing and chunked transfer. Spark.C owns the
//! request contract, tokenizer, admission inputs, token-by-token callback, and
//! response shapes. The CUDA runtime implements [`TokenGenerator`]; the server
//! never imports Python, Torch, SGLang, or an inference-framework scheduler.

use std::fmt::{Display, Formatter};
use std::io::{self, Cursor, Read, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{Value, json};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::tokenizer::{
    ChatMessage, ChatRole, ChatTemplateOptions, NativeQwenTokenizer, QWEN_END_OF_TEXT_ID,
    QWEN_IM_END_ID, QWEN_MODEL_MAX_LENGTH, ReasoningEffort, TokenizerError,
};

const MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_MAX_NEW_TOKENS: u32 = 256;
const DEFAULT_MAX_IN_FLIGHT_REQUESTS: usize = 64;
const SSE_FRAME_CHANNEL_CAPACITY: usize = 1;
static NEXT_COMPLETION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    pub prompt_token_ids: Vec<u32>,
    pub max_new_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    pub stop_token_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationError {
    message: String,
}

impl GenerationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for GenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for GenerationError {}

/// Minimal handoff from HTTP into the Rust/CUDA runtime. Implementations own
/// admission, fixed arena leases, paging, graph launch, sampling, and stopping.
pub trait TokenGenerator: Send + Sync + 'static {
    fn model_id(&self) -> &str;

    /// Sampling default used only when the wire request omits `temperature`.
    /// Stochastic backends keep the OpenAI-compatible default; greedy-only
    /// capsules override this with zero instead of rejecting an ordinary
    /// request after admission.
    fn default_temperature(&self) -> f32 {
        1.0
    }

    /// Hard process-level admission limit. A model-specific single-slot
    /// capsule should normally allow one active and one queued request.
    fn max_in_flight_requests(&self) -> usize {
        DEFAULT_MAX_IN_FLIGHT_REQUESTS
    }

    /// Model-specific admission policy evaluated after tokenization and before
    /// the request is assigned an ID or enters a backend queue.
    fn validate_generation(
        &self,
        _request: &GenerationRequest,
    ) -> Result<(), GenerationError> {
        Ok(())
    }

    fn generate(
        &self,
        request: GenerationRequest,
        emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
    ) -> Result<FinishReason, GenerationError>;
}

/// Model-specific native tokenizer boundary used by the shared OpenAI layer.
/// Implementations may wrap a Rust tokenizer or a statically linked engine
/// tokenizer, but may not delegate serving-time work to Python/framework code.
pub trait OpenAiTokenizer: Send + Sync + 'static {
    fn encode_chat(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<Vec<u32>, TokenizerError>;

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError>;

    fn model_max_length(&self) -> usize;

    fn stop_token_ids(&self) -> Vec<u32>;
}

impl OpenAiTokenizer for NativeQwenTokenizer {
    fn encode_chat(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<Vec<u32>, TokenizerError> {
        NativeQwenTokenizer::encode_chat(self, messages, options)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        NativeQwenTokenizer::decode(self, ids, skip_special_tokens)
    }

    fn model_max_length(&self) -> usize {
        QWEN_MODEL_MAX_LENGTH
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        // The pinned Qwen generation_config.json declares both tokens as EOS.
        vec![QWEN_IM_END_ID, QWEN_END_OF_TEXT_ID]
    }
}

struct InFlightPermit {
    counter: Arc<AtomicUsize>,
}

impl InFlightPermit {
    fn try_acquire(counter: Arc<AtomicUsize>, limit: usize) -> Option<Self> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < limit).then_some(current + 1)
            })
            .ok()?;
        Some(Self { counter })
    }
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);
    }
}

pub struct OpenAiServer<T: OpenAiTokenizer, B: TokenGenerator> {
    tokenizer: Arc<T>,
    backend: Arc<B>,
}

impl<T: OpenAiTokenizer, B: TokenGenerator> OpenAiServer<T, B> {
    pub fn new(tokenizer: Arc<T>, backend: Arc<B>) -> Self {
        Self { tokenizer, backend }
    }

    pub fn serve(self, bind: &str) -> Result<(), ServerError> {
        let server = Server::http(bind).map_err(|error| ServerError::Bind(error.to_string()))?;
        let service = Arc::new(self);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let limit = service.backend.max_in_flight_requests();
        for request in server.incoming_requests() {
            let Some(permit) = InFlightPermit::try_acquire(Arc::clone(&in_flight), limit) else {
                respond_error(
                    request,
                    503,
                    "server is at its in-flight request limit",
                    "server_error",
                    None,
                );
                continue;
            };
            let service = Arc::clone(&service);
            thread::spawn(move || {
                let _permit = permit;
                service.handle(request);
            });
        }
        Ok(())
    }

    fn handle(&self, request: Request) {
        let method = request.method().clone();
        let path = request.url().split('?').next().unwrap_or(request.url());
        match (method, path) {
            (Method::Get, "/health") => respond_json(request, 200, json!({"status": "ok"})),
            (Method::Get, "/v1/models") => {
                let model = self.backend.model_id();
                respond_json(
                    request,
                    200,
                    json!({
                        "object": "list",
                        "data": [{
                            "id": model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "spark.c"
                        }]
                    }),
                );
            }
            (Method::Post, "/v1/chat/completions") => self.handle_chat_completion(request),
            (Method::Post, "/v1/responses") => self.handle_response(request),
            _ => respond_error(request, 404, "route not found", "not_found_error", None),
        }
    }

    fn handle_chat_completion(&self, mut request: Request) {
        let mut body = Vec::new();
        let read_result = request
            .as_reader()
            .take(MAX_REQUEST_BYTES + 1)
            .read_to_end(&mut body);
        if let Err(error) = read_result {
            respond_error(
                request,
                400,
                &format!("cannot read request body: {error}"),
                "invalid_request_error",
                None,
            );
            return;
        }
        if body.len() as u64 > MAX_REQUEST_BYTES {
            respond_error(
                request,
                413,
                "request body exceeds 4 MiB",
                "invalid_request_error",
                None,
            );
            return;
        }

        let parsed: ChatCompletionRequest = match serde_json::from_slice(&body) {
            Ok(parsed) => parsed,
            Err(error) => {
                respond_error(
                    request,
                    400,
                    &format!("invalid JSON request: {error}"),
                    "invalid_request_error",
                    None,
                );
                return;
            }
        };
        let stream = parsed.stream;
        let include_usage = parsed
            .stream_options
            .as_ref()
            .is_some_and(|options| options.include_usage);
        let prepared = match self.prepare(parsed) {
            Ok(prepared) => prepared,
            Err(error) => {
                respond_api_error(request, error);
                return;
            }
        };
        if stream {
            self.respond_streaming(request, prepared, include_usage);
        } else {
            self.respond_non_streaming(request, prepared);
        }
    }

    fn handle_response(&self, mut request: Request) {
        let mut body = Vec::new();
        let read_result = request
            .as_reader()
            .take(MAX_REQUEST_BYTES + 1)
            .read_to_end(&mut body);
        if let Err(error) = read_result {
            respond_error(
                request,
                400,
                &format!("cannot read request body: {error}"),
                "invalid_request_error",
                None,
            );
            return;
        }
        if body.len() as u64 > MAX_REQUEST_BYTES {
            respond_error(
                request,
                413,
                "request body exceeds 4 MiB",
                "invalid_request_error",
                None,
            );
            return;
        }

        let parsed: ResponsesRequest = match serde_json::from_slice(&body) {
            Ok(parsed) => parsed,
            Err(error) => {
                respond_error(
                    request,
                    400,
                    &format!("invalid JSON request: {error}"),
                    "invalid_request_error",
                    None,
                );
                return;
            }
        };
        let stream = parsed.stream;
        let prepared = match self.prepare_response(parsed) {
            Ok(prepared) => prepared,
            Err(error) => {
                respond_api_error(request, error);
                return;
            }
        };
        if stream {
            self.respond_response_streaming(request, prepared);
        } else {
            self.respond_response_non_streaming(request, prepared);
        }
    }

    fn prepare(&self, request: ChatCompletionRequest) -> Result<PreparedCompletion, ApiError> {
        if request.model != self.backend.model_id() {
            return Err(ApiError::invalid(
                format!("unknown model '{}'", request.model),
                Some("model"),
            ));
        }
        if request.stop.is_some() {
            return Err(ApiError::invalid(
                "custom stop sequences are not supported; this service uses the model EOS tokens",
                Some("stop"),
            ));
        }
        if request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
        {
            return Err(ApiError::invalid(
                "tools are not supported by this lightweight Chat Completions endpoint",
                Some("tools"),
            ));
        }
        if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
            return Err(ApiError::invalid(
                "max_tokens and max_completion_tokens cannot both be set",
                Some("max_tokens"),
            ));
        }
        let max_new_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        if max_new_tokens == 0 {
            return Err(ApiError::invalid(
                "max_tokens must be greater than zero",
                Some("max_tokens"),
            ));
        }
        let temperature = request
            .temperature
            .unwrap_or_else(|| self.backend.default_temperature());
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::invalid(
                "temperature must be finite and between 0 and 2",
                Some("temperature"),
            ));
        }
        let top_p = request.top_p.unwrap_or(1.0);
        if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
            return Err(ApiError::invalid(
                "top_p must be finite, greater than 0, and at most 1",
                Some("top_p"),
            ));
        }
        let messages = request
            .messages
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<ChatMessage>, ApiError>>()?;
        let template = request.chat_template_kwargs.unwrap_or_default();
        let options = ChatTemplateOptions {
            enable_thinking: template.enable_thinking,
            preserve_thinking: template.preserve_thinking,
            reasoning_effort: request.reasoning_effort.unwrap_or_default().into(),
            add_generation_prompt: true,
        };
        let prompt_token_ids = self
            .tokenizer
            .encode_chat(&messages, options)
            .map_err(ApiError::tokenizer)?;
        let total_tokens = prompt_token_ids
            .len()
            .checked_add(max_new_tokens as usize)
            .ok_or_else(|| ApiError::invalid("context length overflow", Some("max_tokens")))?;
        let model_max_length = self.tokenizer.model_max_length();
        if total_tokens > model_max_length {
            return Err(ApiError::invalid(
                format!(
                    "prompt plus completion is {total_tokens} tokens, model maximum is {model_max_length}"
                ),
                Some("max_tokens"),
            ));
        }
        let generation = GenerationRequest {
            prompt_token_ids,
            max_new_tokens,
            temperature,
            top_p,
            seed: request.seed,
            stop_token_ids: self.tokenizer.stop_token_ids(),
        };
        self.backend
            .validate_generation(&generation)
            .map_err(|error| ApiError::invalid(error.to_string(), None))?;
        let id = NEXT_COMPLETION_ID.fetch_add(1, Ordering::Relaxed);
        Ok(PreparedCompletion {
            completion_id: format!("chatcmpl-spark-{id:016x}"),
            created: unix_seconds(),
            model: request.model,
            generation,
        })
    }

    fn respond_non_streaming(&self, request: Request, prepared: PreparedCompletion) {
        let prompt_tokens = prepared.generation.prompt_token_ids.len();
        let mut completion_ids = Vec::new();
        let result = self.backend.generate(prepared.generation, &mut |token| {
            completion_ids.push(token);
            Ok(())
        });
        let finish_reason = match result {
            Ok(reason) => reason,
            Err(error) => {
                respond_error(request, 500, &error.to_string(), "server_error", None);
                return;
            }
        };
        let content = match self.tokenizer.decode(&completion_ids, true) {
            Ok(content) => content,
            Err(error) => {
                respond_error(request, 500, &error.to_string(), "server_error", None);
                return;
            }
        };
        let completion_tokens = completion_ids.len();
        respond_json(
            request,
            200,
            json!({
                "id": prepared.completion_id,
                "object": "chat.completion",
                "created": prepared.created,
                "model": prepared.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": null,
                    "finish_reason": finish_reason.as_str()
                }],
                "usage": usage(prompt_tokens, completion_tokens)
            }),
        );
    }

    fn respond_streaming(
        &self,
        request: Request,
        prepared: PreparedCompletion,
        include_usage: bool,
    ) {
        let (sender, receiver) = sse_frame_channel();
        let backend = Arc::clone(&self.backend);
        let tokenizer = Arc::clone(&self.tokenizer);
        thread::spawn(move || {
            produce_stream(backend, tokenizer, prepared, include_usage, sender);
        });
        respond_sse_frames(request, receiver);
    }

    fn prepare_response(&self, request: ResponsesRequest) -> Result<PreparedResponse, ApiError> {
        if request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
        {
            return Err(ApiError::invalid(
                "tools are not supported by this lightweight Responses endpoint",
                Some("tools"),
            ));
        }
        if request.previous_response_id.is_some() {
            return Err(ApiError::invalid(
                "previous_response_id is not supported; resend the conversation in input",
                Some("previous_response_id"),
            ));
        }

        let mut messages = Vec::new();
        if let Some(instructions) = request.instructions.as_ref() {
            messages.push(WireMessage {
                role: "system".to_owned(),
                content: Value::String(instructions.clone()),
                reasoning_content: None,
            });
        }
        messages.extend(parse_response_input(request.input)?);

        let reasoning_effort = request.reasoning.and_then(|reasoning| reasoning.effort);
        let chat = ChatCompletionRequest {
            model: request.model,
            messages,
            stream: request.stream,
            max_tokens: None,
            max_completion_tokens: request.max_output_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            reasoning_effort,
            chat_template_kwargs: request.chat_template_kwargs,
            stream_options: None,
            stop: None,
            tools: None,
        };
        let completion = self.prepare(chat)?;
        let suffix = completion
            .completion_id
            .strip_prefix("chatcmpl-spark-")
            .unwrap_or(&completion.completion_id);
        Ok(PreparedResponse {
            response_id: format!("resp-spark-{suffix}"),
            message_id: format!("msg-spark-{suffix}"),
            created: completion.created,
            model: completion.model,
            max_output_tokens: completion.generation.max_new_tokens,
            temperature: completion.generation.temperature,
            top_p: completion.generation.top_p,
            generation: completion.generation,
            instructions: request.instructions,
            reasoning_effort,
            store: request.store.unwrap_or(false),
            metadata: request.metadata.unwrap_or_else(|| json!({})),
        })
    }

    fn respond_response_non_streaming(&self, request: Request, prepared: PreparedResponse) {
        let prompt_tokens = prepared.generation.prompt_token_ids.len();
        let mut completion_ids = Vec::new();
        let result = self
            .backend
            .generate(prepared.generation.clone(), &mut |token| {
                completion_ids.push(token);
                Ok(())
            });
        let finish_reason = match result {
            Ok(reason) => reason,
            Err(error) => {
                respond_error(request, 500, &error.to_string(), "server_error", None);
                return;
            }
        };
        let content = match self.tokenizer.decode(&completion_ids, true) {
            Ok(content) => content,
            Err(error) => {
                respond_error(request, 500, &error.to_string(), "server_error", None);
                return;
            }
        };
        respond_json(
            request,
            200,
            response_value(
                &prepared,
                finish_reason,
                &content,
                prompt_tokens,
                completion_ids.len(),
            ),
        );
    }

    fn respond_response_streaming(&self, request: Request, prepared: PreparedResponse) {
        let (sender, receiver) = sse_frame_channel();
        let backend = Arc::clone(&self.backend);
        let tokenizer = Arc::clone(&self.tokenizer);
        thread::spawn(move || {
            produce_response_stream(backend, tokenizer, prepared, sender);
        });
        respond_sse_frames(request, receiver);
    }
}

fn produce_stream<T: OpenAiTokenizer, B: TokenGenerator>(
    backend: Arc<B>,
    tokenizer: Arc<T>,
    prepared: PreparedCompletion,
    include_usage: bool,
    sender: mpsc::SyncSender<Vec<u8>>,
) {
    let prompt_tokens = prepared.generation.prompt_token_ids.len();
    if send_sse(
        &sender,
        json!({
            "id": prepared.completion_id,
            "object": "chat.completion.chunk",
            "created": prepared.created,
            "model": prepared.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": null,
                "finish_reason": null
            }]
        }),
    )
    .is_err()
    {
        return;
    }

    let mut completion_ids = Vec::new();
    let mut emitted = String::new();
    let result = backend.generate(prepared.generation, &mut |token| {
        completion_ids.push(token);
        let decoded = tokenizer
            .decode(&completion_ids, true)
            .map_err(|error| GenerationError::new(error.to_string()))?;
        let stable = decoded.trim_end_matches('\u{fffd}');
        if !stable.starts_with(&emitted) {
            return Err(GenerationError::new(
                "streaming tokenizer decode was not prefix-monotonic",
            ));
        }
        let delta = &stable[emitted.len()..];
        if !delta.is_empty() {
            send_sse(
                &sender,
                json!({
                    "id": prepared.completion_id,
                    "object": "chat.completion.chunk",
                    "created": prepared.created,
                    "model": prepared.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta},
                        "logprobs": null,
                        "finish_reason": null
                    }]
                }),
            )?;
        }
        emitted.clear();
        emitted.push_str(stable);
        Ok(())
    });

    let finish_reason = match result {
        Ok(reason) => reason,
        Err(error) => {
            let _ = send_sse(
                &sender,
                json!({"error": api_error_value(&error.to_string(), "server_error", None)}),
            );
            let _ = sender.send(b"data: [DONE]\n\n".to_vec());
            return;
        }
    };
    let decoded = match tokenizer.decode(&completion_ids, true) {
        Ok(decoded) => decoded,
        Err(error) => {
            let _ = send_sse(
                &sender,
                json!({"error": api_error_value(&error.to_string(), "server_error", None)}),
            );
            let _ = sender.send(b"data: [DONE]\n\n".to_vec());
            return;
        }
    };
    let final_delta = &decoded[emitted.len()..];
    if !final_delta.is_empty()
        && send_sse(
            &sender,
            json!({
                "id": prepared.completion_id,
                "object": "chat.completion.chunk",
                "created": prepared.created,
                "model": prepared.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": final_delta},
                    "logprobs": null,
                    "finish_reason": null
                }]
            }),
        )
        .is_err()
    {
        return;
    }
    let completion_tokens = completion_ids.len();
    let usage_value = include_usage.then(|| usage(prompt_tokens, completion_tokens));
    if send_sse(
        &sender,
        json!({
            "id": prepared.completion_id,
            "object": "chat.completion.chunk",
            "created": prepared.created,
            "model": prepared.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "logprobs": null,
                "finish_reason": finish_reason.as_str()
            }],
            "usage": usage_value
        }),
    )
    .is_ok()
    {
        let _ = sender.send(b"data: [DONE]\n\n".to_vec());
    }
}

fn produce_response_stream<T: OpenAiTokenizer, B: TokenGenerator>(
    backend: Arc<B>,
    tokenizer: Arc<T>,
    prepared: PreparedResponse,
    sender: mpsc::SyncSender<Vec<u8>>,
) {
    let prompt_tokens = prepared.generation.prompt_token_ids.len();
    let mut sequence = 0_u64;
    let initial = response_envelope(
        &prepared,
        "in_progress",
        None,
        Value::Null,
        json!([]),
        Value::Null,
    );
    if send_response_sse(
        &sender,
        "response.created",
        json!({
            "type": "response.created",
            "sequence_number": sequence,
            "response": initial
        }),
    )
    .is_err()
    {
        return;
    }
    sequence += 1;
    if send_response_sse(
        &sender,
        "response.in_progress",
        json!({
            "type": "response.in_progress",
            "sequence_number": sequence,
            "response": response_envelope(
                &prepared,
                "in_progress",
                None,
                Value::Null,
                json!([]),
                Value::Null,
            )
        }),
    )
    .is_err()
    {
        return;
    }
    sequence += 1;
    if send_response_sse(
        &sender,
        "response.output_item.added",
        json!({
            "type": "response.output_item.added",
            "sequence_number": sequence,
            "output_index": 0,
            "item": response_message_item(&prepared, "in_progress", "")
        }),
    )
    .is_err()
    {
        return;
    }
    sequence += 1;
    if send_response_sse(
        &sender,
        "response.content_part.added",
        json!({
            "type": "response.content_part.added",
            "sequence_number": sequence,
            "item_id": prepared.message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
        }),
    )
    .is_err()
    {
        return;
    }
    sequence += 1;

    let mut completion_ids = Vec::new();
    let mut emitted = String::new();
    let result = backend.generate(prepared.generation.clone(), &mut |token| {
        completion_ids.push(token);
        let decoded = tokenizer
            .decode(&completion_ids, true)
            .map_err(|error| GenerationError::new(error.to_string()))?;
        let stable = decoded.trim_end_matches('\u{fffd}');
        if !stable.starts_with(&emitted) {
            return Err(GenerationError::new(
                "streaming tokenizer decode was not prefix-monotonic",
            ));
        }
        let delta = &stable[emitted.len()..];
        if !delta.is_empty() {
            send_response_sse(
                &sender,
                "response.output_text.delta",
                json!({
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": prepared.message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                    "logprobs": []
                }),
            )?;
            sequence += 1;
        }
        emitted.clear();
        emitted.push_str(stable);
        Ok(())
    });

    let finish_reason = match result {
        Ok(reason) => reason,
        Err(error) => {
            let failed = response_envelope(
                &prepared,
                "failed",
                Some(unix_seconds()),
                Value::Null,
                json!([]),
                Value::Null,
            );
            let _ = send_response_sse(
                &sender,
                "response.failed",
                json!({
                    "type": "response.failed",
                    "sequence_number": sequence,
                    "response": failed,
                    "error": api_error_value(&error.to_string(), "server_error", None)
                }),
            );
            return;
        }
    };
    let decoded = match tokenizer.decode(&completion_ids, true) {
        Ok(decoded) => decoded,
        Err(error) => {
            let _ = send_response_sse(
                &sender,
                "response.failed",
                json!({
                    "type": "response.failed",
                    "sequence_number": sequence,
                    "error": api_error_value(&error.to_string(), "server_error", None)
                }),
            );
            return;
        }
    };
    let final_delta = &decoded[emitted.len()..];
    if !final_delta.is_empty() {
        if send_response_sse(
            &sender,
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "sequence_number": sequence,
                "item_id": prepared.message_id,
                "output_index": 0,
                "content_index": 0,
                "delta": final_delta,
                "logprobs": []
            }),
        )
        .is_err()
        {
            return;
        }
        sequence += 1;
    }
    let item_status = response_status(finish_reason);
    let completion_tokens = completion_ids.len();
    let events = [
        (
            "response.output_text.done",
            json!({
                "type": "response.output_text.done",
                "item_id": prepared.message_id,
                "output_index": 0,
                "content_index": 0,
                "text": decoded,
                "logprobs": []
            }),
        ),
        (
            "response.content_part.done",
            json!({
                "type": "response.content_part.done",
                "item_id": prepared.message_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": decoded, "annotations": [], "logprobs": []}
            }),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": response_message_item(&prepared, item_status, &decoded)
            }),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "response": response_value(
                    &prepared,
                    finish_reason,
                    &decoded,
                    prompt_tokens,
                    completion_tokens,
                )
            }),
        ),
    ];
    for (event, mut value) in events {
        if let Some(object) = value.as_object_mut() {
            object.insert("sequence_number".to_owned(), json!(sequence));
        }
        if send_response_sse(&sender, event, value).is_err() {
            return;
        }
        sequence += 1;
    }
}

fn sse_frame_channel() -> (mpsc::SyncSender<Vec<u8>>, mpsc::Receiver<Vec<u8>>) {
    mpsc::sync_channel(SSE_FRAME_CHANNEL_CAPACITY)
}

fn send_sse(sender: &mpsc::SyncSender<Vec<u8>>, value: Value) -> Result<(), GenerationError> {
    let mut frame = b"data: ".to_vec();
    serde_json::to_writer(&mut frame, &value)
        .map_err(|error| GenerationError::new(error.to_string()))?;
    frame.extend_from_slice(b"\n\n");
    sender
        .send(frame)
        .map_err(|_| GenerationError::new("client disconnected"))
}

fn send_response_sse(
    sender: &mpsc::SyncSender<Vec<u8>>,
    event: &str,
    value: Value,
) -> Result<(), GenerationError> {
    let mut frame = format!("event: {event}\ndata: ").into_bytes();
    serde_json::to_writer(&mut frame, &value)
        .map_err(|error| GenerationError::new(error.to_string()))?;
    frame.extend_from_slice(b"\n\n");
    sender
        .send(frame)
        .map_err(|_| GenerationError::new("client disconnected"))
}

/// Write each SSE frame as one HTTP chunk and flush it immediately.
///
/// tiny_http 0.12 wraps unknown-length responses in
/// `chunked_transfer::Encoder::new`, whose 8192-byte buffer is only flushed
/// when full or dropped. A token stream smaller than that therefore arrives
/// at the client when generation ends. `Request::into_writer` is tiny_http's
/// public raw-response boundary; using it here preserves its connection
/// sequencing while making frame delivery explicit.
fn respond_sse_frames(request: Request, receiver: mpsc::Receiver<Vec<u8>>) {
    if *request.http_version() <= (1, 0) {
        let response = Response::new(
            StatusCode(200),
            vec![
                header("Content-Type", "text/event-stream; charset=utf-8"),
                header("Cache-Control", "no-cache"),
                header("X-Accel-Buffering", "no"),
            ],
            ChannelReader::new(receiver),
            None,
            None,
        );
        let _ = request.respond(response);
        return;
    }

    let version = request.http_version().clone();
    let mut writer = request.into_writer();
    let result = (|| -> io::Result<()> {
        write!(
            writer,
            "HTTP/{} 200 OK\r\n\
             Content-Type: text/event-stream; charset=utf-8\r\n\
             Cache-Control: no-cache\r\n\
             X-Accel-Buffering: no\r\n\
             Transfer-Encoding: chunked\r\n\r\n",
            version
        )?;
        writer.flush()?;

        for frame in receiver {
            write!(writer, "{:x}\r\n", frame.len())?;
            writer.write_all(&frame)?;
            writer.write_all(b"\r\n")?;
            writer.flush()?;
        }
        writer.write_all(b"0\r\n\r\n")?;
        writer.flush()
    })();
    let _ = result;
}

struct ChannelReader {
    receiver: mpsc::Receiver<Vec<u8>>,
    current: Cursor<Vec<u8>>,
    closed: bool,
}

impl ChannelReader {
    fn new(receiver: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            current: Cursor::new(Vec::new()),
            closed: false,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let read = self.current.read(output)?;
            if read != 0 || output.is_empty() {
                return Ok(read);
            }
            if self.closed {
                return Ok(0);
            }
            match self.receiver.recv() {
                Ok(bytes) => self.current = Cursor::new(bytes),
                Err(_) => self.closed = true,
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    reasoning_effort: Option<WireReasoningEffort>,
    chat_template_kwargs: Option<WireChatTemplateOptions>,
    stream_options: Option<StreamOptions>,
    stop: Option<Value>,
    tools: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    model: String,
    input: Value,
    instructions: Option<String>,
    #[serde(default)]
    stream: bool,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    reasoning: Option<ResponsesReasoning>,
    chat_template_kwargs: Option<WireChatTemplateOptions>,
    store: Option<bool>,
    metadata: Option<Value>,
    tools: Option<Vec<Value>>,
    previous_response_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ResponsesReasoning {
    effort: Option<WireReasoningEffort>,
}

fn parse_response_input(input: Value) -> Result<Vec<WireMessage>, ApiError> {
    match input {
        Value::String(content) => Ok(vec![WireMessage {
            role: "user".to_owned(),
            content: Value::String(content),
            reasoning_content: None,
        }]),
        Value::Array(items) => items.into_iter().map(parse_response_input_item).collect(),
        _ => Err(ApiError::invalid(
            "input must be a string or an array of text messages",
            Some("input"),
        )),
    }
}

fn parse_response_input_item(item: Value) -> Result<WireMessage, ApiError> {
    let object = item
        .as_object()
        .ok_or_else(|| ApiError::invalid("input items must be message objects", Some("input")))?;
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::invalid("input message role is required", Some("input")))?;
    let role = match role {
        "developer" => "system",
        "system" | "user" | "assistant" | "tool" => role,
        _ => {
            return Err(ApiError::invalid(
                format!("unsupported input message role '{role}'"),
                Some("input"),
            ));
        }
    };
    let content = object
        .get("content")
        .ok_or_else(|| ApiError::invalid("input message content is required", Some("input")))?;
    let content = match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let part = part.as_object().ok_or_else(|| {
                    ApiError::invalid("input content parts must be objects", Some("input"))
                })?;
                let kind = part.get("type").and_then(Value::as_str).ok_or_else(|| {
                    ApiError::invalid("input content part type is required", Some("input"))
                })?;
                if !matches!(kind, "input_text" | "output_text") {
                    return Err(ApiError::invalid(
                        format!("unsupported input content type '{kind}'"),
                        Some("input"),
                    ));
                }
                text.push_str(part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ApiError::invalid("text content part requires text", Some("input"))
                })?);
            }
            text
        }
        _ => {
            return Err(ApiError::invalid(
                "input message content must be text",
                Some("input"),
            ));
        }
    };
    Ok(WireMessage {
        role: role.to_owned(),
        content: Value::String(content),
        reasoning_content: None,
    })
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    role: String,
    content: Value,
    reasoning_content: Option<String>,
}

impl TryFrom<WireMessage> for ChatMessage {
    type Error = ApiError;

    fn try_from(message: WireMessage) -> Result<Self, Self::Error> {
        let role = match message.role.as_str() {
            "system" => ChatRole::System,
            "developer" => ChatRole::System,
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            "tool" => ChatRole::Tool,
            _ => {
                return Err(ApiError::invalid(
                    format!("unsupported message role '{}'", message.role),
                    Some("messages"),
                ));
            }
        };
        let content = message.content.as_str().ok_or_else(|| {
            ApiError::invalid("only text message content is supported", Some("messages"))
        })?;
        Ok(ChatMessage {
            role,
            content: content.to_owned(),
            reasoning_content: message.reasoning_content,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum WireReasoningEffort {
    #[serde(rename = "xhigh")]
    #[default]
    XHigh,
    High,
    Medium,
    Low,
    Minimal,
}

impl From<WireReasoningEffort> for ReasoningEffort {
    fn from(value: WireReasoningEffort) -> Self {
        match value {
            WireReasoningEffort::XHigh => Self::XHigh,
            WireReasoningEffort::High => Self::XHigh,
            WireReasoningEffort::Medium => Self::Medium,
            WireReasoningEffort::Low => Self::Low,
            WireReasoningEffort::Minimal => Self::Low,
        }
    }
}

impl WireReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::XHigh => "xhigh",
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
            Self::Minimal => "minimal",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct WireChatTemplateOptions {
    #[serde(default = "enabled")]
    enable_thinking: bool,
    #[serde(default = "enabled")]
    preserve_thinking: bool,
}

impl Default for WireChatTemplateOptions {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            preserve_thinking: true,
        }
    }
}

#[derive(Debug, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

fn enabled() -> bool {
    true
}

struct PreparedCompletion {
    completion_id: String,
    created: u64,
    model: String,
    generation: GenerationRequest,
}

struct PreparedResponse {
    response_id: String,
    message_id: String,
    created: u64,
    model: String,
    generation: GenerationRequest,
    instructions: Option<String>,
    max_output_tokens: u32,
    temperature: f32,
    top_p: f32,
    reasoning_effort: Option<WireReasoningEffort>,
    store: bool,
    metadata: Value,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ApiError {
    status: u16,
    message: String,
    kind: &'static str,
    param: Option<&'static str>,
}

impl ApiError {
    fn invalid(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            kind: "invalid_request_error",
            param,
        }
    }

    fn tokenizer(error: TokenizerError) -> Self {
        Self::invalid(error.to_string(), Some("messages"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServerError {
    Bind(String),
}

impl Display for ServerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind(error) => write!(formatter, "cannot bind HTTP server: {error}"),
        }
    }
}

impl std::error::Error for ServerError {}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    })
}

fn responses_usage(input_tokens: usize, output_tokens: usize) -> Value {
    json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens
    })
}

fn response_status(finish_reason: FinishReason) -> &'static str {
    match finish_reason {
        FinishReason::Stop => "completed",
        FinishReason::Length => "incomplete",
    }
}

fn response_message_item(prepared: &PreparedResponse, status: &str, content: &str) -> Value {
    json!({
        "id": prepared.message_id.as_str(),
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": content,
            "annotations": [],
            "logprobs": []
        }]
    })
}

fn response_envelope(
    prepared: &PreparedResponse,
    status: &str,
    completed_at: Option<u64>,
    incomplete_details: Value,
    output: Value,
    usage: Value,
) -> Value {
    json!({
        "id": prepared.response_id.as_str(),
        "object": "response",
        "created_at": prepared.created,
        "completed_at": completed_at,
        "status": status,
        "error": null,
        "incomplete_details": incomplete_details,
        "instructions": prepared.instructions.as_deref(),
        "max_output_tokens": prepared.max_output_tokens,
        "model": prepared.model.as_str(),
        "output": output,
        "parallel_tool_calls": false,
        "previous_response_id": null,
        "reasoning": {
            "effort": prepared.reasoning_effort.map(WireReasoningEffort::as_str),
            "summary": null
        },
        "store": prepared.store,
        "temperature": prepared.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": "none",
        "tools": [],
        "top_p": prepared.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": &prepared.metadata
    })
}

fn response_value(
    prepared: &PreparedResponse,
    finish_reason: FinishReason,
    content: &str,
    input_tokens: usize,
    output_tokens: usize,
) -> Value {
    let status = response_status(finish_reason);
    let incomplete_details = match finish_reason {
        FinishReason::Stop => Value::Null,
        FinishReason::Length => json!({"reason": "max_output_tokens"}),
    };
    response_envelope(
        prepared,
        status,
        (finish_reason == FinishReason::Stop).then(unix_seconds),
        incomplete_details,
        json!([response_message_item(prepared, status, content)]),
        responses_usage(input_tokens, output_tokens),
    )
}

fn api_error_value(message: &str, kind: &str, param: Option<&str>) -> Value {
    json!({
        "message": message,
        "type": kind,
        "param": param,
        "code": null
    })
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("static HTTP header")
}

fn respond_json(request: Request, status: u16, value: Value) {
    let response = Response::from_string(value.to_string())
        .with_status_code(StatusCode(status))
        .with_header(header("Content-Type", "application/json; charset=utf-8"));
    let _ = request.respond(response);
}

fn respond_error(request: Request, status: u16, message: &str, kind: &str, param: Option<&str>) {
    respond_json(
        request,
        status,
        json!({"error": api_error_value(message, kind, param)}),
    );
}

fn respond_api_error(request: Request, error: ApiError) {
    respond_error(
        request,
        error.status,
        &error.message,
        error.kind,
        error.param,
    );
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::{Shutdown, TcpStream};

    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    struct FixedBackend;

    impl TokenGenerator for FixedBackend {
        fn model_id(&self) -> &str {
            "qwen3.8-flash-next-nvfp4"
        }

        fn generate(
            &self,
            _request: GenerationRequest,
            emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
        ) -> Result<FinishReason, GenerationError> {
            emit(1)?;
            emit(2)?;
            Ok(FinishReason::Stop)
        }
    }

    struct GreedyDefaultBackend;

    impl TokenGenerator for GreedyDefaultBackend {
        fn model_id(&self) -> &str {
            "qwen3.8-27b-native"
        }

        fn default_temperature(&self) -> f32 {
            0.0
        }

        fn generate(
            &self,
            _request: GenerationRequest,
            _emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
        ) -> Result<FinishReason, GenerationError> {
            unreachable!("prepare-only test backend")
        }
    }

    struct AdmissionBackend;

    impl TokenGenerator for AdmissionBackend {
        fn model_id(&self) -> &str {
            "qwen3.8-27b-admission"
        }

        fn default_temperature(&self) -> f32 {
            0.0
        }

        fn validate_generation(
            &self,
            request: &GenerationRequest,
        ) -> Result<(), GenerationError> {
            if request.temperature != 0.0 || request.top_p != 1.0 {
                return Err(GenerationError::new("backend is greedy-only"));
            }
            if request.max_new_tokens > 1 {
                return Err(GenerationError::new("backend slot capacity exceeded"));
            }
            Ok(())
        }

        fn generate(
            &self,
            _request: GenerationRequest,
            _emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
        ) -> Result<FinishReason, GenerationError> {
            unreachable!("admission-only test backend")
        }
    }

    fn tokenizer() -> Arc<NativeQwenTokenizer> {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_owned(), 0),
                    ("hello".to_owned(), 1),
                    ("world".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_owned())
            .build()
            .expect("word-level model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        Arc::new(NativeQwenTokenizer::from_inner(tokenizer))
    }

    #[test]
    fn qwen_stop_ids_match_pinned_generation_config() {
        assert_eq!(
            OpenAiTokenizer::stop_token_ids(tokenizer().as_ref()),
            [QWEN_IM_END_ID, QWEN_END_OF_TEXT_ID]
        );
    }

    #[test]
    fn backend_temperature_default_applies_only_when_wire_value_is_absent() {
        let service = OpenAiServer::new(tokenizer(), Arc::new(GreedyDefaultBackend));
        let omitted: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "qwen3.8-27b-native",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("request without temperature");
        assert_eq!(
            service.prepare(omitted).expect("prepared").generation.temperature,
            0.0
        );

        let explicit: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "qwen3.8-27b-native",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "temperature": 0.25,
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("request with temperature");
        assert_eq!(
            service.prepare(explicit).expect("prepared").generation.temperature,
            0.25
        );
    }

    #[test]
    fn backend_generation_policy_maps_to_prequeue_http_400() {
        let service = OpenAiServer::new(tokenizer(), Arc::new(AdmissionBackend));
        let request = |temperature: Option<f32>, max_tokens: u32| {
            serde_json::from_value(json!({
                "model": "qwen3.8-27b-admission",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": false}
            }))
            .expect("wire request")
        };

        assert!(service.prepare(request(None, 1)).is_ok());
        for (wire, expected) in [
            (request(Some(0.25), 1), "greedy-only"),
            (request(None, 2), "slot capacity"),
        ] {
            let error = match service.prepare(wire) {
                Err(error) => error,
                Ok(_) => panic!("backend policy rejection was admitted"),
            };
            assert_eq!(error.status, 400);
            assert!(error.message.contains(expected));
        }
    }

    #[test]
    fn unsupported_chat_stop_and_tools_are_rejected_instead_of_ignored() {
        let service = OpenAiServer::new(tokenizer(), Arc::new(GreedyDefaultBackend));
        for (field, field_value) in [
            ("stop", json!("END")),
            ("tools", json!([{"type": "function", "function": {"name": "lookup"}}])),
        ] {
            let mut value = json!({
                "model": "qwen3.8-27b-native",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1
            });
            value
                .as_object_mut()
                .expect("request object")
                .insert(field.to_owned(), field_value);
            let request = serde_json::from_value(value).expect("wire request");
            let error = match service.prepare(request) {
                Err(error) => error,
                Ok(_) => panic!("unsupported field was accepted"),
            };
            assert_eq!(error.status, 400);
            assert_eq!(error.param, Some(field));
        }
    }

    #[test]
    fn in_flight_admission_rejects_saturation_and_reopens_on_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let first = InFlightPermit::try_acquire(Arc::clone(&counter), 2).expect("first permit");
        let second =
            InFlightPermit::try_acquire(Arc::clone(&counter), 2).expect("second permit");
        assert!(InFlightPermit::try_acquire(Arc::clone(&counter), 2).is_none());
        assert_eq!(counter.load(Ordering::Acquire), 2);

        drop(first);
        let replacement =
            InFlightPermit::try_acquire(Arc::clone(&counter), 2).expect("replacement permit");
        assert_eq!(counter.load(Ordering::Acquire), 2);
        drop(second);
        drop(replacement);
        assert_eq!(counter.load(Ordering::Acquire), 0);
        assert!(InFlightPermit::try_acquire(counter, 0).is_none());
    }

    #[test]
    fn sse_frame_channel_backpressures_after_one_buffered_frame() {
        assert_eq!(SSE_FRAME_CHANNEL_CAPACITY, 1);
        let (sender, receiver) = sse_frame_channel();
        sender.send(b"first".to_vec()).expect("fill channel");
        let (attempted, observe_attempt) = mpsc::sync_channel(0);
        let (finished, observe_finish) = mpsc::sync_channel(0);
        let producer = thread::spawn(move || {
            attempted.send(()).expect("announce send");
            sender.send(b"second".to_vec()).expect("backpressured send");
            finished.send(()).expect("announce finish");
        });

        observe_attempt.recv().expect("producer reached send");
        assert!(matches!(
            observe_finish.recv_timeout(std::time::Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert_eq!(receiver.recv().expect("first frame"), b"first");
        observe_finish
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("producer unblocked");
        assert_eq!(receiver.recv().expect("second frame"), b"second");
        producer.join().expect("producer thread");
    }

    #[test]
    fn parses_text_messages_and_qwen_options() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "qwen3.8-flash-next-nvfp4",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer", "reasoning_content": "thought"},
                {"role": "tool", "content": "result"}
            ],
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": false},
            "stream": true
        }))
        .expect("request");
        assert!(request.stream);
        assert!(matches!(
            request.reasoning_effort,
            Some(WireReasoningEffort::Low)
        ));
        let template = request.chat_template_kwargs.expect("template");
        assert!(!template.enable_thinking);
        assert!(template.preserve_thinking);
        let messages = request
            .messages
            .into_iter()
            .map(ChatMessage::try_from)
            .collect::<Result<Vec<_>, _>>()
            .expect("messages");
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[1].reasoning_content.as_deref(), Some("thought"));
        assert_eq!(messages[2].role, ChatRole::Tool);
    }

    #[test]
    fn text_only_boundary_rejects_multimodal_content() {
        let message = WireMessage {
            role: "user".to_owned(),
            content: json!([{"type": "text", "text": "hello"}]),
            reasoning_content: None,
        };
        let error = ChatMessage::try_from(message).expect_err("multimodal rejected");
        assert_eq!(error.param, Some("messages"));
    }

    #[test]
    fn responses_input_accepts_string_and_text_message_parts() {
        let string_messages = parse_response_input(json!("hello")).expect("string input");
        assert_eq!(string_messages.len(), 1);
        assert_eq!(string_messages[0].role, "user");
        assert_eq!(string_messages[0].content, json!("hello"));

        let item_messages = parse_response_input(json!([{
            "role": "developer",
            "content": [
                {"type": "input_text", "text": "be "},
                {"type": "input_text", "text": "brief"}
            ]
        }]))
        .expect("message input");
        assert_eq!(item_messages[0].role, "system");
        assert_eq!(item_messages[0].content, json!("be brief"));
    }

    #[test]
    fn channel_reader_preserves_sse_frame_boundaries_and_eof() {
        let (sender, receiver) = mpsc::channel();
        sender.send(b"data: one\n\n".to_vec()).expect("first");
        sender.send(b"data: two\n\n".to_vec()).expect("second");
        drop(sender);
        let mut reader = ChannelReader::new(receiver);
        let mut body = String::new();
        reader.read_to_string(&mut body).expect("read");
        assert_eq!(body, "data: one\n\ndata: two\n\n");
    }

    #[test]
    fn generation_error_is_stable_for_backend_and_client_disconnects() {
        let error = GenerationError::new("CUDA graph launch failed");
        assert_eq!(error.to_string(), "CUDA graph launch failed");
    }

    #[test]
    fn actual_http_adapter_streams_openai_sse_chunks() {
        let listener = Server::http("127.0.0.1:0").expect("test listener");
        let address = listener.server_addr().to_ip().expect("IP listener");
        let service = OpenAiServer::new(tokenizer(), Arc::new(FixedBackend));
        let server_thread = thread::spawn(move || {
            let request = listener.recv().expect("request");
            service.handle(request);
        });

        let body = json!({
            "model": "qwen3.8-flash-next-nvfp4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 2,
            "chat_template_kwargs": {"enable_thinking": false}
        })
        .to_string();
        let mut client = TcpStream::connect(address).expect("connect");
        write!(
            client,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .expect("write request");
        client.shutdown(Shutdown::Write).expect("request EOF");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("read response");
        server_thread.join().expect("server thread");

        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.contains("Content-Type: text/event-stream"));
        assert!(response.contains("\"role\":\"assistant\""));
        assert!(response.contains("\"content\":\"hello\""));
        assert!(response.contains("\"content\":\" world\""));
        assert!(response.contains("\"finish_reason\":\"stop\""));
        assert!(response.contains("\"completion_tokens\":2"));
        assert!(response.contains("data: [DONE]"));
    }

    #[test]
    fn actual_http_adapter_streams_responses_events() {
        let listener = Server::http("127.0.0.1:0").expect("test listener");
        let address = listener.server_addr().to_ip().expect("IP listener");
        let service = OpenAiServer::new(tokenizer(), Arc::new(FixedBackend));
        let server_thread = thread::spawn(move || {
            let request = listener.recv().expect("request");
            service.handle(request);
        });

        let body = json!({
            "model": "qwen3.8-flash-next-nvfp4",
            "input": "hello",
            "stream": true,
            "max_output_tokens": 2,
            "chat_template_kwargs": {"enable_thinking": false}
        })
        .to_string();
        let mut client = TcpStream::connect(address).expect("connect");
        write!(
            client,
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .expect("write request");
        client.shutdown(Shutdown::Write).expect("request EOF");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("read response");
        server_thread.join().expect("server thread");

        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.contains("event: response.created"));
        assert!(response.contains("event: response.output_text.delta"));
        assert!(response.contains("\"delta\":\"hello\""));
        assert!(response.contains("\"delta\":\" world\""));
        assert!(response.contains("event: response.completed"));
        assert!(response.contains("\"output_tokens\":2"));
        assert!(!response.contains("data: [DONE]"));
    }
}

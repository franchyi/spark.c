//! Safe Rust owner for the pinned ds4 GLM-5.3 CUDA source engine.
//!
//! The linked code is source-built into the SparkServe binary. Rust owns the
//! engine/session lifetime and later layers scheduling and HTTP semantics over
//! this narrow ABI; no ds4 process or shared-library installation is required.

use std::ffi::{CStr, CString, c_char};
use std::fmt::{Display, Formatter};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use crate::openai_server::{
    FinishReason, GenerationError, GenerationRequest, OpenAiTokenizer, TokenGenerator,
};
use crate::tokenizer::{
    ChatMessage as OpenAiChatMessage, ChatRole, ChatTemplateOptions, TokenizerError,
};

const ABI_VERSION: u32 = 1;

pub const WARM_WEIGHTS: u32 = 1 << 0;
pub const QUALITY: u32 = 1 << 1;
pub const SSD_STREAMING: u32 = 1 << 2;
pub const SSD_STREAMING_COLD: u32 = 1 << 3;
pub const MTP: u32 = 1 << 4;

#[repr(C)]
struct NativeEngine {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeSession {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeStatus {
    code: i32,
    message: *const c_char,
}

#[repr(C)]
struct NativeConfig {
    struct_size: u32,
    abi_version: u32,
    model_path: *const c_char,
    context_size: u32,
    prefill_chunk: u32,
    threads: u32,
    power_percent: u32,
    flags: u32,
    ssd_streaming_cache_bytes: u64,
    ssd_streaming_cache_experts: u32,
    reserved: u32,
}

#[repr(C)]
struct NativeMessage {
    role: *const c_char,
    content: *const c_char,
}

#[repr(C)]
struct NativeTokens {
    data: *mut i32,
    len: u32,
    reserved: u32,
}

unsafe extern "C" {
    fn sparkserve_ds4_glm53_open(
        out: *mut *mut NativeEngine,
        config: *const NativeConfig,
    ) -> NativeStatus;
    fn sparkserve_ds4_glm53_close(engine: *mut NativeEngine);
    fn sparkserve_ds4_glm53_model_name(engine: *const NativeEngine) -> *const c_char;
    fn sparkserve_ds4_glm53_eos_token(engine: *const NativeEngine) -> i32;
    fn sparkserve_ds4_glm53_is_stop_token(engine: *const NativeEngine, token: i32) -> u32;
    fn sparkserve_ds4_glm53_encode_messages(
        engine: *mut NativeEngine,
        messages: *const NativeMessage,
        message_count: u32,
        enable_thinking: u32,
        out: *mut NativeTokens,
    ) -> NativeStatus;
    fn sparkserve_ds4_glm53_encode_text(
        engine: *mut NativeEngine,
        text: *const c_char,
        out: *mut NativeTokens,
    ) -> NativeStatus;
    fn sparkserve_ds4_glm53_tokens_free(tokens: *mut NativeTokens);
    fn sparkserve_ds4_glm53_session_create(
        out: *mut *mut NativeSession,
        engine: *mut NativeEngine,
    ) -> NativeStatus;
    fn sparkserve_ds4_glm53_session_free(session: *mut NativeSession);
    fn sparkserve_ds4_glm53_session_sync(
        session: *mut NativeSession,
        tokens: *const i32,
        token_count: u32,
    ) -> NativeStatus;
    fn sparkserve_ds4_glm53_session_sample(
        session: *mut NativeSession,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: *mut u64,
    ) -> i32;
    fn sparkserve_ds4_glm53_session_eval(
        session: *mut NativeSession,
        token: i32,
    ) -> NativeStatus;
    fn sparkserve_ds4_glm53_session_position(session: *const NativeSession) -> i32;
    fn sparkserve_ds4_glm53_token_piece(
        engine: *mut NativeEngine,
        token: i32,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> NativeStatus;
    fn sparkserve_ds4_glm53_text_free(text: *mut u8);
}

#[derive(Clone, Debug)]
pub struct EngineConfig<'a> {
    pub model_path: &'a str,
    pub context_size: u32,
    pub prefill_chunk: u32,
    pub threads: u32,
    pub power_percent: u32,
    pub flags: u32,
    pub ssd_streaming_cache_bytes: u64,
    pub ssd_streaming_cache_experts: u32,
}

impl<'a> EngineConfig<'a> {
    pub fn resident(model_path: &'a str, context_size: u32) -> Self {
        Self {
            model_path,
            context_size,
            prefill_chunk: 0,
            threads: 0,
            power_percent: 0,
            flags: 0,
            ssd_streaming_cache_bytes: 0,
            ssd_streaming_cache_experts: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

impl<'a> ChatMessage<'a> {
    pub fn new(role: &'a str, content: &'a str) -> Self {
        Self { role, content }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error(String);

impl Error {
    fn native(status: NativeStatus) -> Result<(), Self> {
        if status.code == 0 {
            return Ok(());
        }
        let message = if status.message.is_null() {
            "unknown ds4 GLM-5.3 error".to_owned()
        } else {
            // SAFETY: Adapter error strings are thread-local NUL-terminated
            // buffers that remain valid until the next adapter call.
            unsafe { CStr::from_ptr(status.message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Self(message))
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

struct EngineInner {
    native: NonNull<NativeEngine>,
    context_size: usize,
}

// ds4 owns immutable engine weights. Mutable graph state lives in Session and
// SparkServe serializes a session lease before entering the native engine.
unsafe impl Send for EngineInner {}
unsafe impl Sync for EngineInner {}

impl Drop for EngineInner {
    fn drop(&mut self) {
        // SAFETY: This is the sole owner of the engine returned by open.
        unsafe { sparkserve_ds4_glm53_close(self.native.as_ptr()) };
    }
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

impl Engine {
    pub fn open(config: EngineConfig<'_>) -> Result<Self, Error> {
        let model_path = CString::new(config.model_path)
            .map_err(|_| Error("GLM-5.3 model path contains a NUL byte".to_owned()))?;
        let native_config = NativeConfig {
            struct_size: u32::try_from(std::mem::size_of::<NativeConfig>())
                .expect("native config size fits u32"),
            abi_version: ABI_VERSION,
            model_path: model_path.as_ptr(),
            context_size: config.context_size,
            prefill_chunk: config.prefill_chunk,
            threads: config.threads,
            power_percent: config.power_percent,
            flags: config.flags,
            ssd_streaming_cache_bytes: config.ssd_streaming_cache_bytes,
            ssd_streaming_cache_experts: config.ssd_streaming_cache_experts,
            reserved: 0,
        };
        let mut native = std::ptr::null_mut();
        // SAFETY: Pointers remain valid for the duration of the call and the
        // adapter initializes `native` only on success.
        Error::native(unsafe { sparkserve_ds4_glm53_open(&mut native, &native_config) })?;
        let native = NonNull::new(native)
            .ok_or_else(|| Error("ds4 returned a null GLM-5.3 engine".to_owned()))?;
        Ok(Self {
            inner: Arc::new(EngineInner {
                native,
                context_size: config.context_size as usize,
            }),
        })
    }

    pub fn model_name(&self) -> String {
        // SAFETY: The engine owns this string for its full lifetime.
        let name = unsafe { sparkserve_ds4_glm53_model_name(self.inner.native.as_ptr()) };
        if name.is_null() {
            return "glm-5.3-flash".to_owned();
        }
        // SAFETY: ds4 model names are NUL terminated.
        unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
    }

    pub fn eos_token(&self) -> i32 {
        // SAFETY: Engine pointer is live.
        unsafe { sparkserve_ds4_glm53_eos_token(self.inner.native.as_ptr()) }
    }

    pub fn is_stop_token(&self, token: i32) -> bool {
        // SAFETY: Engine pointer is live and token is an ordinary scalar.
        unsafe { sparkserve_ds4_glm53_is_stop_token(self.inner.native.as_ptr(), token) != 0 }
    }

    pub fn encode_messages(
        &self,
        messages: &[ChatMessage<'_>],
        enable_thinking: bool,
    ) -> Result<Vec<i32>, Error> {
        let roles = messages
            .iter()
            .map(|message| CString::new(message.role))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Error("GLM-5.3 message role contains a NUL byte".to_owned()))?;
        let contents = messages
            .iter()
            .map(|message| CString::new(message.content))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Error("GLM-5.3 message content contains a NUL byte".to_owned()))?;
        let native_messages = roles
            .iter()
            .zip(&contents)
            .map(|(role, content)| NativeMessage {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();
        let count = u32::try_from(native_messages.len())
            .map_err(|_| Error("too many GLM-5.3 chat messages".to_owned()))?;
        let mut tokens = NativeTokens {
            data: std::ptr::null_mut(),
            len: 0,
            reserved: 0,
        };
        // SAFETY: Message CString storage and the output struct remain live.
        let status = unsafe {
            sparkserve_ds4_glm53_encode_messages(
                self.inner.native.as_ptr(),
                native_messages.as_ptr(),
                count,
                u32::from(enable_thinking),
                &mut tokens,
            )
        };
        Error::native(status)?;
        let output = if tokens.len == 0 {
            Vec::new()
        } else {
            if tokens.data.is_null() {
                // SAFETY: A successful native allocation can be freed through
                // the matching adapter function even if its pointer is null.
                unsafe { sparkserve_ds4_glm53_tokens_free(&mut tokens) };
                return Err(Error("ds4 returned null prompt tokens".to_owned()));
            }
            // SAFETY: Adapter allocated exactly `len` i32 values.
            unsafe { std::slice::from_raw_parts(tokens.data, tokens.len as usize) }.to_vec()
        };
        // SAFETY: Releases the adapter-owned token allocation once.
        unsafe { sparkserve_ds4_glm53_tokens_free(&mut tokens) };
        Ok(output)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<i32>, Error> {
        let text = CString::new(text)
            .map_err(|_| Error("GLM-5.3 input text contains a NUL byte".to_owned()))?;
        let mut tokens = NativeTokens {
            data: std::ptr::null_mut(),
            len: 0,
            reserved: 0,
        };
        // SAFETY: Text and output storage remain live for the native call.
        Error::native(unsafe {
            sparkserve_ds4_glm53_encode_text(
                self.inner.native.as_ptr(),
                text.as_ptr(),
                &mut tokens,
            )
        })?;
        let output = if tokens.len == 0 {
            Vec::new()
        } else {
            if tokens.data.is_null() {
                // SAFETY: Matching native free accepts a null data pointer.
                unsafe { sparkserve_ds4_glm53_tokens_free(&mut tokens) };
                return Err(Error("ds4 returned null text tokens".to_owned()));
            }
            // SAFETY: Adapter allocated exactly `len` i32 values.
            unsafe { std::slice::from_raw_parts(tokens.data, tokens.len as usize) }.to_vec()
        };
        // SAFETY: Releases the adapter-owned token allocation once.
        unsafe { sparkserve_ds4_glm53_tokens_free(&mut tokens) };
        Ok(output)
    }

    pub fn session(&self) -> Result<Session, Error> {
        let mut native = std::ptr::null_mut();
        // SAFETY: Engine pointer remains live through the Arc cloned below.
        Error::native(unsafe {
            sparkserve_ds4_glm53_session_create(&mut native, self.inner.native.as_ptr())
        })?;
        let native = NonNull::new(native)
            .ok_or_else(|| Error("ds4 returned a null GLM-5.3 session".to_owned()))?;
        Ok(Session {
            native,
            engine: Arc::clone(&self.inner),
        })
    }

    pub fn token_piece(&self, token: i32) -> Result<Vec<u8>, Error> {
        let mut pointer = std::ptr::null_mut();
        let mut len = 0usize;
        // SAFETY: Engine and output pointers are live for this call.
        Error::native(unsafe {
            sparkserve_ds4_glm53_token_piece(
                self.inner.native.as_ptr(),
                token,
                &mut pointer,
                &mut len,
            )
        })?;
        let output = if len == 0 {
            Vec::new()
        } else {
            if pointer.is_null() {
                return Err(Error("ds4 returned a null token piece".to_owned()));
            }
            // SAFETY: Adapter allocated exactly `len` bytes.
            unsafe { std::slice::from_raw_parts(pointer, len) }.to_vec()
        };
        // SAFETY: Releases the adapter-owned token piece once.
        unsafe { sparkserve_ds4_glm53_text_free(pointer) };
        Ok(output)
    }
}

pub struct Session {
    native: NonNull<NativeSession>,
    engine: Arc<EngineInner>,
}

unsafe impl Send for Session {}

impl Session {
    pub fn sync(&mut self, tokens: &[i32]) -> Result<(), Error> {
        let count = u32::try_from(tokens.len())
            .map_err(|_| Error("GLM-5.3 prompt exceeds u32 tokens".to_owned()))?;
        // SAFETY: Session is exclusively borrowed and token slice is live.
        Error::native(unsafe {
            sparkserve_ds4_glm53_session_sync(self.native.as_ptr(), tokens.as_ptr(), count)
        })
    }

    pub fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> Result<i32, Error> {
        // SAFETY: Session is exclusively borrowed and RNG is live.
        let token = unsafe {
            sparkserve_ds4_glm53_session_sample(
                self.native.as_ptr(),
                temperature,
                top_k,
                top_p,
                min_p,
                rng,
            )
        };
        if token < 0 {
            Err(Error("ds4 GLM-5.3 sampling failed".to_owned()))
        } else {
            Ok(token)
        }
    }

    pub fn eval(&mut self, token: i32) -> Result<(), Error> {
        // SAFETY: Session is exclusively borrowed.
        Error::native(unsafe {
            sparkserve_ds4_glm53_session_eval(self.native.as_ptr(), token)
        })
    }

    pub fn position(&self) -> i32 {
        // SAFETY: Session remains live.
        unsafe { sparkserve_ds4_glm53_session_position(self.native.as_ptr()) }
    }

    pub fn engine(&self) -> Engine {
        Engine {
            inner: Arc::clone(&self.engine),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: This is the sole owner of the native session.
        unsafe { sparkserve_ds4_glm53_session_free(self.native.as_ptr()) };
    }
}

impl OpenAiTokenizer for Engine {
    fn encode_chat(
        &self,
        messages: &[OpenAiChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<Vec<u32>, TokenizerError> {
        if messages.is_empty() {
            return Err(TokenizerError::NoMessages);
        }
        let mut roles = Vec::with_capacity(messages.len());
        let mut contents = Vec::with_capacity(messages.len());
        for message in messages {
            let role = match message.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => "tool",
            };
            roles.push(role);
            if message.role == ChatRole::Assistant
                && options.preserve_thinking
                && message.reasoning_content.is_some()
            {
                contents.push(format!(
                    "<think>\n{}\n</think>\n\n{}",
                    message.reasoning_content.as_deref().unwrap_or_default(),
                    message.content
                ));
            } else {
                contents.push(message.content.clone());
            }
        }
        let native_messages = roles
            .iter()
            .zip(&contents)
            .map(|(role, content)| ChatMessage::new(role, content))
            .collect::<Vec<_>>();
        self.encode_messages(&native_messages, options.enable_thinking)
            .and_then(|tokens| {
                tokens
                    .into_iter()
                    .map(|token| {
                        u32::try_from(token)
                            .map_err(|_| Error(format!("negative GLM-5.3 token id {token}")))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(|error| TokenizerError::Encode(error.to_string()))
    }

    fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for &id in ids {
            let token = i32::try_from(id)
                .map_err(|_| TokenizerError::Decode(format!("token id {id} exceeds i32")))?;
            if skip_special_tokens && self.is_stop_token(token) {
                continue;
            }
            bytes.extend_from_slice(
                &self
                    .token_piece(token)
                    .map_err(|error| TokenizerError::Decode(error.to_string()))?,
            );
        }
        // Byte-level token pieces may end in the middle of one UTF-8 scalar.
        // The shared SSE layer holds back a trailing replacement character and
        // retries after the next token, matching the Qwen streaming contract.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn model_max_length(&self) -> usize {
        self.inner.context_size
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        u32::try_from(self.eos_token()).map_or_else(|_| Vec::new(), |token| vec![token])
    }
}

/// Single-session correctness owner. The mutex is the initial Rust session
/// lease: HTTP workers may arrive concurrently, while one ds4 graph timeline
/// advances at a time. A bounded multi-session pool can replace it without
/// widening the native ABI.
pub struct Generator {
    engine: Engine,
    session: Mutex<Session>,
    model_id: String,
}

impl Generator {
    pub fn new(engine: Engine) -> Result<Self, Error> {
        let session = engine.session()?;
        let model_id = engine.model_name();
        Ok(Self {
            engine,
            session: Mutex::new(session),
            model_id,
        })
    }
}

impl TokenGenerator for Generator {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn generate(
        &self,
        request: GenerationRequest,
        emit: &mut dyn FnMut(u32) -> Result<(), GenerationError>,
    ) -> Result<FinishReason, GenerationError> {
        let prompt = request
            .prompt_token_ids
            .iter()
            .map(|&token| {
                i32::try_from(token)
                    .map_err(|_| GenerationError::new(format!("token id {token} exceeds i32")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| GenerationError::new("GLM-5.3 session lease is poisoned"))?;
        session
            .sync(&prompt)
            .map_err(|error| GenerationError::new(error.to_string()))?;
        let mut rng = request.seed.unwrap_or(1);
        for index in 0..request.max_new_tokens {
            let token = session
                .sample(request.temperature, 0, request.top_p, 0.05, &mut rng)
                .map_err(|error| GenerationError::new(error.to_string()))?;
            let output = u32::try_from(token)
                .map_err(|_| GenerationError::new(format!("negative token id {token}")))?;
            if self.engine.is_stop_token(token) || request.stop_token_ids.contains(&output) {
                return Ok(FinishReason::Stop);
            }
            emit(output)?;
            if index + 1 == request.max_new_tokens {
                return Ok(FinishReason::Length);
            }
            session
                .eval(token)
                .map_err(|error| GenerationError::new(error.to_string()))?;
        }
        Ok(FinishReason::Length)
    }
}

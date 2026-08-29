//! Native Qwen tokenizer and text-only chat-template boundary.
//!
//! Hugging Face's Rust `tokenizers` crate owns byte-level BPE, added-token,
//! regex, and decode semantics. Spark.C owns request validation and renders
//! the pinned Qwen3.8 text chat contract without Python or Transformers.

use std::fmt::{Display, Formatter};
use std::path::Path;

use tokenizers::Tokenizer;

pub const QWEN_END_OF_TEXT_ID: u32 = 248_044;
pub const QWEN_IM_START_ID: u32 = 248_045;
pub const QWEN_IM_END_ID: u32 = 248_046;
pub const QWEN_MODEL_MAX_LENGTH: usize = 262_144;

const XHIGH_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW_INSTRUCTION: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReasoningEffort {
    #[default]
    XHigh,
    Medium,
    Low,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatTemplateOptions {
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
    pub reasoning_effort: ReasoningEffort,
    pub add_generation_prompt: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            preserve_thinking: true,
            reasoning_effort: ReasoningEffort::XHigh,
            add_generation_prompt: true,
        }
    }
}

pub fn render_qwen_text_chat(
    messages: &[ChatMessage],
    options: ChatTemplateOptions,
) -> Result<String, TokenizerError> {
    if messages.is_empty() {
        return Err(TokenizerError::NoMessages);
    }
    if messages
        .iter()
        .enumerate()
        .any(|(index, message)| message.role == ChatRole::System && index != 0)
    {
        return Err(TokenizerError::MisplacedSystemMessage);
    }
    if !messages
        .iter()
        .any(|message| message.role == ChatRole::User)
    {
        return Err(TokenizerError::NoUserQuery);
    }

    let reasoning_instruction = if !options.enable_thinking {
        ""
    } else {
        match options.reasoning_effort {
            ReasoningEffort::XHigh => XHIGH_INSTRUCTION,
            ReasoningEffort::Medium => "",
            ReasoningEffort::Low => LOW_INSTRUCTION,
        }
    };
    let mut output = String::new();
    let mut first_message = 0;
    if messages[0].role == ChatRole::System {
        let system = messages[0].content.trim();
        if !system.is_empty() || !reasoning_instruction.is_empty() {
            output.push_str("<|im_start|>system\n");
            if !reasoning_instruction.is_empty() {
                output.push_str(reasoning_instruction);
                if !system.is_empty() {
                    output.push_str("\n\n");
                }
            }
            output.push_str(system);
            output.push_str("<|im_end|>\n");
        }
        first_message = 1;
    } else if !reasoning_instruction.is_empty() {
        output.push_str("<|im_start|>system\n");
        output.push_str(reasoning_instruction);
        output.push_str("<|im_end|>\n");
    }

    let last_user = messages
        .iter()
        .rposition(|message| message.role == ChatRole::User)
        .ok_or(TokenizerError::NoUserQuery)?;
    let mut index = first_message;
    while index < messages.len() {
        let message = &messages[index];
        match message.role {
            ChatRole::System => return Err(TokenizerError::MisplacedSystemMessage),
            ChatRole::User => {
                output.push_str("<|im_start|>user\n");
                output.push_str(message.content.trim());
                output.push_str("<|im_end|>\n");
                index += 1;
            }
            ChatRole::Assistant => {
                output.push_str("<|im_start|>assistant\n");
                if options.preserve_thinking || index > last_user {
                    output.push_str("<think>\n");
                    if let Some(reasoning) = &message.reasoning_content {
                        output.push_str(reasoning.trim());
                    }
                    output.push_str("\n</think>\n\n");
                }
                output.push_str(message.content.trim());
                output.push_str("<|im_end|>\n");
                index += 1;
            }
            ChatRole::Tool => {
                output.push_str("<|im_start|>user");
                while index < messages.len() && messages[index].role == ChatRole::Tool {
                    output.push_str("\n<tool_response>\n");
                    output.push_str(messages[index].content.trim());
                    output.push_str("\n</tool_response>");
                    index += 1;
                }
                output.push_str("<|im_end|>\n");
            }
        }
    }

    if options.add_generation_prompt {
        output.push_str("<|im_start|>assistant\n<think>\n");
        if !options.enable_thinking {
            output.push_str("\n</think>\n\n");
        }
    }
    Ok(output)
}

pub struct NativeQwenTokenizer {
    inner: Tokenizer,
}

impl NativeQwenTokenizer {
    pub fn from_model_root(model_root: &Path) -> Result<Self, TokenizerError> {
        let path = model_root.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|error| TokenizerError::Load(path.display().to_string(), error.to_string()))?;
        let tokenizer = Self { inner };
        tokenizer.require_token("<|endoftext|>", QWEN_END_OF_TEXT_ID)?;
        tokenizer.require_token("<|im_start|>", QWEN_IM_START_ID)?;
        tokenizer.require_token("<|im_end|>", QWEN_IM_END_ID)?;
        Ok(tokenizer)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| TokenizerError::Encode(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<Vec<u32>, TokenizerError> {
        let rendered = render_qwen_text_chat(messages, options)?;
        self.encode_text(&rendered)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|error| TokenizerError::Decode(error.to_string()))
    }

    pub fn decoder(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            ids: Vec::new(),
            emitted: String::new(),
            skip_special_tokens,
        }
    }

    fn require_token(&self, token: &'static str, expected: u32) -> Result<(), TokenizerError> {
        let actual = self.inner.token_to_id(token);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(TokenizerError::TokenId {
                token,
                expected,
                actual,
            })
        }
    }

    #[cfg(test)]
    pub(crate) fn from_inner(inner: Tokenizer) -> Self {
        Self { inner }
    }
}

/// Conservative streaming decoder. It holds back a trailing replacement
/// character because byte-level tokens can split one UTF-8 codepoint.
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a NativeQwenTokenizer,
    ids: Vec<u32>,
    emitted: String,
    skip_special_tokens: bool,
}

impl IncrementalDecoder<'_> {
    pub fn push(&mut self, token: u32) -> Result<String, TokenizerError> {
        self.ids.push(token);
        let decoded = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        let stable = decoded.trim_end_matches('\u{fffd}');
        if !stable.starts_with(&self.emitted) {
            return Err(TokenizerError::NonMonotonicDecode);
        }
        let delta = stable[self.emitted.len()..].to_owned();
        self.emitted.clear();
        self.emitted.push_str(stable);
        Ok(delta)
    }

    pub fn finish(&mut self) -> Result<String, TokenizerError> {
        let decoded = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        if !decoded.starts_with(&self.emitted) {
            return Err(TokenizerError::NonMonotonicDecode);
        }
        let delta = decoded[self.emitted.len()..].to_owned();
        self.emitted = decoded;
        Ok(delta)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerError {
    NoMessages,
    NoUserQuery,
    MisplacedSystemMessage,
    Load(String, String),
    Encode(String),
    Decode(String),
    TokenId {
        token: &'static str,
        expected: u32,
        actual: Option<u32>,
    },
    NonMonotonicDecode,
}

impl Display for TokenizerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMessages => formatter.write_str("Qwen chat requires at least one message"),
            Self::NoUserQuery => formatter.write_str("Qwen chat contains no user query"),
            Self::MisplacedSystemMessage => {
                formatter.write_str("Qwen system message must be first")
            }
            Self::Load(path, error) => write!(formatter, "cannot load tokenizer {path}: {error}"),
            Self::Encode(error) => write!(formatter, "tokenizer encode failed: {error}"),
            Self::Decode(error) => write!(formatter, "tokenizer decode failed: {error}"),
            Self::TokenId {
                token,
                expected,
                actual,
            } => write!(
                formatter,
                "tokenizer special token {token} is {actual:?}, expected {expected}"
            ),
            Self::NonMonotonicDecode => {
                formatter.write_str("streaming tokenizer decode was not prefix-monotonic")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

#[cfg(test)]
mod tests {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    fn tokenizer() -> NativeQwenTokenizer {
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
        NativeQwenTokenizer::from_inner(tokenizer)
    }

    #[test]
    fn renders_pinned_reasoning_and_generation_prompt() {
        let rendered = render_qwen_text_chat(
            &[ChatMessage::new(ChatRole::User, "  hello  ")],
            ChatTemplateOptions::default(),
        )
        .expect("render");
        assert_eq!(
            rendered,
            format!(
                "<|im_start|>system\n{XHIGH_INSTRUCTION}<|im_end|>\n\
                 <|im_start|>user\nhello<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n"
            )
        );
    }

    #[test]
    fn disabled_thinking_matches_empty_think_contract() {
        let rendered = render_qwen_text_chat(
            &[
                ChatMessage::new(ChatRole::System, " concise "),
                ChatMessage::new(ChatRole::User, "hello"),
            ],
            ChatTemplateOptions {
                enable_thinking: false,
                ..ChatTemplateOptions::default()
            },
        )
        .expect("render");
        assert_eq!(
            rendered,
            "<|im_start|>system\nconcise<|im_end|>\n\
             <|im_start|>user\nhello<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn old_assistant_reasoning_can_be_stripped_but_latest_is_preserved() {
        let mut old = ChatMessage::new(ChatRole::Assistant, "first answer");
        old.reasoning_content = Some("old thought".to_owned());
        let mut latest = ChatMessage::new(ChatRole::Assistant, "latest answer");
        latest.reasoning_content = Some("latest thought".to_owned());
        let rendered = render_qwen_text_chat(
            &[
                ChatMessage::new(ChatRole::User, "first"),
                old,
                ChatMessage::new(ChatRole::User, "latest"),
                latest,
            ],
            ChatTemplateOptions {
                preserve_thinking: false,
                add_generation_prompt: false,
                ..ChatTemplateOptions::default()
            },
        )
        .expect("render");
        assert!(!rendered.contains("old thought"));
        assert!(rendered.contains("latest thought"));
    }

    #[test]
    fn groups_consecutive_tool_responses_as_one_user_turn() {
        let rendered = render_qwen_text_chat(
            &[
                ChatMessage::new(ChatRole::User, "call tools"),
                ChatMessage::new(ChatRole::Tool, "one"),
                ChatMessage::new(ChatRole::Tool, "two"),
            ],
            ChatTemplateOptions {
                add_generation_prompt: false,
                ..ChatTemplateOptions::default()
            },
        )
        .expect("render");
        assert!(rendered.contains(
            "<|im_start|>user\n<tool_response>\none\n</tool_response>\n\
             <tool_response>\ntwo\n</tool_response><|im_end|>\n"
        ));
    }

    #[test]
    fn borrowed_tokenizer_encodes_and_stream_decodes() {
        let tokenizer = tokenizer();
        assert_eq!(
            tokenizer.encode_text("hello world").expect("encode"),
            [1, 2]
        );
        let mut stream = tokenizer.decoder(false);
        assert_eq!(stream.push(1).expect("first"), "hello");
        assert_eq!(stream.push(2).expect("second"), " world");
        assert_eq!(stream.finish().expect("finish"), "");
    }

    #[test]
    fn rejects_empty_and_misplaced_system_conversations() {
        assert_eq!(
            render_qwen_text_chat(&[], ChatTemplateOptions::default()),
            Err(TokenizerError::NoMessages)
        );
        assert_eq!(
            render_qwen_text_chat(
                &[
                    ChatMessage::new(ChatRole::User, "hello"),
                    ChatMessage::new(ChatRole::System, "late"),
                ],
                ChatTemplateOptions::default(),
            ),
            Err(TokenizerError::MisplacedSystemMessage)
        );
    }
}

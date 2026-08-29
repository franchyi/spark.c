use std::fs;
use std::io;
use std::path::Path;

use serde_json::Value;
use spark_flash_next::tokenizer::{
    ChatMessage, ChatRole, ChatTemplateOptions, NativeQwenTokenizer, ReasoningEffort,
    render_qwen_text_chat,
};

fn field<'a>(value: &'a Value, name: &str) -> io::Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing {name}")))
}

fn string(value: &Value, name: &str) -> io::Result<String> {
    field(value, name)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("invalid {name}")))
}

fn boolean(value: &Value, name: &str) -> io::Result<bool> {
    field(value, name)?
        .as_bool()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("invalid {name}")))
}

fn role(value: &str) -> io::Result<ChatRole> {
    match value {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        "tool" => Ok(ChatRole::Tool),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid role {value}"),
        )),
    }
}

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let tokenizer_root = args
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing tokenizer root"))?;
    let fixture_path = args
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing fixture"))?;
    if args.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: tokenizer_fixture_check <tokenizer-root> <fixture.json>",
        ));
    }
    let tokenizer = NativeQwenTokenizer::from_model_root(Path::new(&tokenizer_root))
        .map_err(io::Error::other)?;
    let fixture: Value = serde_json::from_slice(&fs::read(fixture_path)?)?;
    let cases = field(&fixture, "cases")?
        .as_array()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid cases"))?;
    for case in cases {
        let name = string(case, "name")?;
        let mut messages = Vec::new();
        for message in field(case, "messages")?
            .as_array()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid messages"))?
        {
            messages.push(ChatMessage {
                role: role(&string(message, "role")?)?,
                content: string(message, "content")?,
                reasoning_content: message
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
        }
        let raw_options = field(case, "options")?;
        let effort = match string(raw_options, "reasoning_effort")?.as_str() {
            "xhigh" => ReasoningEffort::XHigh,
            "medium" => ReasoningEffort::Medium,
            "low" => ReasoningEffort::Low,
            value => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid reasoning effort {value}"),
                ));
            }
        };
        let options = ChatTemplateOptions {
            enable_thinking: boolean(raw_options, "enable_thinking")?,
            preserve_thinking: boolean(raw_options, "preserve_thinking")?,
            reasoning_effort: effort,
            add_generation_prompt: boolean(raw_options, "add_generation_prompt")?,
        };
        let expected_rendered = string(case, "rendered")?;
        let rendered = render_qwen_text_chat(&messages, options).map_err(io::Error::other)?;
        if rendered != expected_rendered {
            return Err(io::Error::other(format!(
                "chat rendering differs for {name}"
            )));
        }
        let expected_ids: Vec<u32> = field(case, "ids")?
            .as_array()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid ids"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid token id"))
            })
            .collect::<io::Result<_>>()?;
        let actual_ids = tokenizer
            .encode_chat(&messages, options)
            .map_err(io::Error::other)?;
        if actual_ids != expected_ids {
            return Err(io::Error::other(format!(
                "token IDs differ for {name}: {} native vs {} oracle",
                actual_ids.len(),
                expected_ids.len()
            )));
        }
        let decoded = tokenizer
            .decode(&actual_ids, false)
            .map_err(io::Error::other)?;
        if decoded != expected_rendered {
            return Err(io::Error::other(format!("decode differs for {name}")));
        }
        println!("{name}: {} exact token IDs", actual_ids.len());
    }
    println!("native Qwen tokenizer/chat parity: exact");
    Ok(())
}

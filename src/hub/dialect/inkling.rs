//! Inkling 2 chat framing from the checkpoint's `tokenizer.chat_template`.
use super::conversation::{ConversationMessage, Rendered, ToolCall, split_preamble};
use super::piece::{Piece, encode_prompt};
use super::{Adapter as DialectAdapter, OutputParser};
use crate::protocol::Effort;
use eyre::{Result, eyre};
use oco::Tokenizer;
use serde_json::{Value, json};

mod output;
#[cfg(test)]
mod tests;

const SYSTEM: &str = "<|message_system|>";
const USER: &str = "<|message_user|>";
const MODEL: &str = "<|message_model|>";
const TOOL: &str = "<|message_tool|>";
const TEXT: &str = "<|content_text|>";
const THINKING: &str = "<|content_thinking|>";
const CALL: &str = "<|content_invoke_tool_json|>";
const END: &str = "<|end_message|>";
const STOP: &str = "<|content_model_end_sampling|>";

fn frame(prompt: &mut Vec<Piece>, role: &'static str, name: &str, kind: &'static str, text: &str) {
    prompt.push(Piece::Marker(role));
    if !name.is_empty() {
        prompt.push(Piece::Text(name.to_owned()));
    }
    prompt.push(Piece::Marker(kind));
    if !text.is_empty() {
        prompt.push(Piece::Text(text.to_owned()));
    }
    prompt.push(Piece::Marker(END));
}

/// Flatten the function declarations and sort their JSON keys as the checkpoint expects.
fn tool_declarations() -> String {
    let tools = super::conversation::tool_schemas();
    let mut tools = Value::Array(
        tools
            .into_iter()
            .map(|mut tool| {
                let mut declaration = tool["function"].take();
                declaration["type"] = tool["type"].take();
                declaration
            })
            .collect(),
    );
    tools.sort_all_objects();
    tools.to_string()
}

/// The checkpoint's thinking-effort scale, from 0 to 1.
fn effort(level: Effort) -> &'static str {
    match level {
        Effort::Low => "0.2",
        Effort::Medium => "0.7",
        Effort::High => "0.9",
    }
}

/// Replay one model step and its results, which may interleave with its calls.
fn step(prompt: &mut Vec<Piece>, messages: &[ConversationMessage]) -> Result<usize> {
    use ConversationMessage::*;
    let text = messages
        .iter()
        .take_while(|message| matches!(message, Thinking(_) | Answer(_)))
        .count();
    let tools = messages[text..]
        .iter()
        .take_while(|message| matches!(message, ToolCall(_) | ToolResult { .. }))
        .count();
    let end = text + tools;
    for message in &messages[..end] {
        match message {
            Thinking(text) if !text.is_empty() => frame(prompt, MODEL, "", THINKING, text),
            Answer(text) if !text.is_empty() => frame(prompt, MODEL, "", TEXT, text),
            ToolCall(call) => emit_call(prompt, call)?,
            _ => {}
        }
    }
    prompt.push(Piece::Marker(STOP));
    for message in &messages[text..end] {
        if let ToolResult { name, content } = message {
            frame(prompt, TOOL, name, TEXT, content);
        }
    }
    Ok(end)
}

fn emit_call(prompt: &mut Vec<Piece>, call: &ToolCall) -> Result<()> {
    let ToolCall { name, arguments } = call;
    let arguments = arguments
        .as_object()
        .ok_or_else(|| eyre!("Inkling tool arguments must be an object"))?;
    let mut arguments = arguments.clone();
    arguments.sort_keys();
    let payload = json!({"name": name, "args": arguments}).to_string();
    frame(prompt, MODEL, name, CALL, &payload);
    Ok(())
}

/// Render complete messages. The next role token is sampled by the model.
fn render(messages: &[ConversationMessage]) -> Result<(Vec<Piece>, usize)> {
    use ConversationMessage::*;
    let mut prompt = Vec::new();
    let declarations = tool_declarations();
    frame(
        &mut prompt,
        SYSTEM,
        "tool_declare",
        "<|content_xml|>",
        &declarations,
    );
    let (head, mut rest) = split_preamble(messages);
    let mut level = crate::protocol::Effort::High;
    let mut preamble = None;
    for message in head {
        match message {
            Effort(effort) => level = *effort,
            Instructions(text) => frame(&mut prompt, SYSTEM, "", TEXT, text),
            Fact(text) => {
                preamble.get_or_insert(prompt.len());
                frame(&mut prompt, SYSTEM, "", TEXT, text);
            }
            _ => unreachable!(),
        }
    }
    let preamble = preamble.unwrap_or(prompt.len());
    let level = effort(level);
    frame(
        &mut prompt,
        SYSTEM,
        "",
        TEXT,
        &format!("Thinking effort level: {level}"),
    );

    while let Some((first, next)) = rest.split_first() {
        match first {
            Thinking(_) | Answer(_) | ToolCall(_) => {
                let consumed = step(&mut prompt, rest)?;
                rest = &rest[consumed..];
                continue;
            }
            Effort(_) => {}
            Instructions(text) | Fact(text) | Note(text) => {
                frame(&mut prompt, SYSTEM, "", TEXT, text)
            }
            User(text) => frame(&mut prompt, USER, "", TEXT, text),
            ToolResult { name, content } => frame(&mut prompt, TOOL, name, TEXT, content),
        }
        rest = next;
    }
    Ok((prompt, preamble))
}

/// Inkling's native conversation and JSON tool-call codec.
pub struct Adapter;

impl DialectAdapter for Adapter {
    fn encode(&self, tokenizer: &Tokenizer, messages: &[ConversationMessage]) -> Result<Rendered> {
        let (pieces, head) = render(messages)?;
        encode_prompt(tokenizer, &pieces, head)
    }

    fn output_parser<'a>(&self, tokenizer: &'a Tokenizer) -> Result<Box<dyn OutputParser + 'a>> {
        Ok(Box::new(output::Parser::new(tokenizer)?))
    }

    fn is_stop_token(&self, tokenizer: &Tokenizer, token: u32) -> bool {
        tokenizer.eos_token() == Some(token)
    }
}

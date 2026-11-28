//! Inkling 2 chat framing from the checkpoint's `tokenizer.chat_template`.
use super::piece::{Piece, encode_prompt};
use super::prompt::{FunctionCall, Preamble, PromptMessage, Rendered};
use crate::hub::wording;
use crate::protocol::Effort;
use eyre::{Result, eyre};
use oco::Tokenizer;
use serde_json::{Value, json};

mod output;
pub use output::Parser;
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
    let tools = super::prompt::tool_schemas();
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

/// The requested effort as the checkpoint's template states it, on its scale from 0 to 1.
fn effort(requested: Effort) -> String {
    // Debug formatting, so a whole level keeps its decimal point: `1.0`, not `1`.
    format!("Thinking effort level: {:?}", requested.level())
}

/// Replay one model step and its results, which may interleave with its calls.
fn step(prompt: &mut Vec<Piece>, messages: &[PromptMessage]) -> Result<usize> {
    use PromptMessage::*;
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

fn emit_call(prompt: &mut Vec<Piece>, call: &FunctionCall) -> Result<()> {
    let FunctionCall { name, arguments } = call;
    let arguments = arguments
        .as_object()
        .ok_or_else(|| eyre!("Inkling tool arguments must be an object"))?;
    let mut arguments = arguments.clone();
    arguments.sort_keys();
    let payload = json!({"name": name, "args": arguments}).to_string();
    frame(prompt, MODEL, name, CALL, &payload);
    Ok(())
}

/// Render the preamble and complete messages. The next role token is sampled by the model.
fn render(preamble: &Preamble, messages: &[PromptMessage]) -> Result<(Vec<Piece>, usize)> {
    use PromptMessage::*;
    let mut prompt = Vec::new();
    let declarations = tool_declarations();
    frame(
        &mut prompt,
        SYSTEM,
        "tool_declare",
        "<|content_xml|>",
        &declarations,
    );
    for instructions in &preamble.instructions {
        frame(&mut prompt, SYSTEM, "", TEXT, instructions);
    }
    // The cacheable prefix ends here: what follows differs from one invocation to the next.
    let prefix = prompt.len();
    let today = wording::date(&preamble.today);
    frame(&mut prompt, SYSTEM, "", TEXT, &today);
    frame(&mut prompt, SYSTEM, "", TEXT, &effort(preamble.effort));

    let mut rest = messages;
    while let Some((first, next)) = rest.split_first() {
        match first {
            Thinking(_) | Answer(_) | ToolCall(_) => {
                let consumed = step(&mut prompt, rest)?;
                rest = &rest[consumed..];
                continue;
            }
            Note(text) => frame(&mut prompt, SYSTEM, "", TEXT, text),
            User(text) => frame(&mut prompt, USER, "", TEXT, text),
            ToolResult { name, content } => frame(&mut prompt, TOOL, name, TEXT, content),
        }
        rest = next;
    }
    Ok((prompt, prefix))
}

/// Encode the preamble and worded messages in Inkling's native framing.
pub fn encode(
    tokenizer: &Tokenizer,
    preamble: &Preamble,
    messages: &[PromptMessage],
) -> Result<Rendered> {
    let (pieces, prefix) = render(preamble, messages)?;
    encode_prompt(tokenizer, &pieces, prefix)
}

pub fn is_stop_token(tokenizer: &Tokenizer, token: u32) -> bool {
    tokenizer.eos_token() == Some(token)
}

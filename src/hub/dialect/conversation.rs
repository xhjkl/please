//! Neutral conversation data shared by the hub-side adapters.
use crate::protocol::Message;
use crate::tools::ParamType;
use eyre::{Result, eyre};
use serde_json::{Map, Value, json};

/// Probe history with context expanded and tool exchanges split into calls and results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationMessage {
    /// Requested reasoning effort, stated wherever each dialect's chat template expects it.
    Effort(String),
    System(String),
    Developer(String),
    /// Per-invocation fact or input (date, stdin, redirection), placed after all static text.
    Fact(String),
    User(String),
    Answer(String),
    Thinking(String),
    ToolCall(ToolCall),
    ToolResult {
        name: String,
        content: String,
    },
}

/// A dialect-neutral call, shared by decoded output and replayed tool history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ConversationMessage {
    /// Leading entries that dialects fold into their system and developer header.
    pub fn is_preamble(&self) -> bool {
        matches!(
            self,
            Self::Effort(_) | Self::System(_) | Self::Developer(_) | Self::Fact(_)
        )
    }
}

/// Encoded prompt and how many of its leading tokens are the same for every invocation
/// with this effort: the part a hub may prefill once and cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub tokens: Vec<u32>,
    pub preamble: usize,
}

/// Newly visible generated text, by the section it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delta {
    Answer(String),
    Thinking(String),
}

/// Declare every tool as an OpenAI-style function schema for the chat templates.
pub fn tool_schemas() -> Vec<Value> {
    let mut tools: Vec<_> = crate::tools::all_tools().into_iter().collect();
    tools.sort_by_key(|(name, _)| *name);
    tools
        .into_iter()
        .map(|(name, (description, _, parameters))| {
            let mut properties = Map::new();
            let mut required = Vec::new();
            for parameter in parameters {
                let mut schema = match &parameter.param_type {
                    ParamType::String => json!({"type": "string"}),
                    ParamType::StringArray => json!({"type": "array", "items": {"type": "string"}}),
                    ParamType::Number => json!({"type": "number"}),
                    ParamType::Choice(choices) => json!({"type": "string", "enum": choices}),
                };
                schema["description"] = parameter.desc.into();
                if parameter.required {
                    required.push(parameter.name);
                }
                properties.insert(parameter.name.to_string(), schema);
            }
            json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": properties, "required": required},
                },
            })
        })
        .collect()
}

/// Halve the longer output stream of a command result, counting the dropped bytes.
/// Returns how many bytes were dropped; zero means nothing is left to trim.
fn trim_command_stream(result: &mut Value) -> usize {
    let length = |stream| {
        result
            .get(stream)
            .and_then(Value::as_str)
            .map_or(0, str::len)
    };
    // Reducing the longer stream first; short diagnostics in the other remain visible.
    let (stream, counter) = if length("stdout") >= length("stderr") {
        ("stdout", "stdoutBytesOmitted")
    } else {
        ("stderr", "stderrBytesOmitted")
    };
    let Some(Value::String(output)) = result.get_mut(stream) else {
        return 0;
    };
    let mut start = output.len().div_ceil(2);
    while !output.is_char_boundary(start) {
        start += 1;
    }
    output.drain(..start);
    let omitted = result.get(counter).and_then(Value::as_u64).unwrap_or(0);
    result[counter] = Value::from(omitted.saturating_add(start as u64));
    start
}

/// Give each tool exchange half the tail budget for command results, keeping the newest output.
/// Trimming per exchange keeps earlier rendered steps reusable as a prefix.
pub fn fit_command_results(
    messages: &mut [ConversationMessage],
    tail_budget: usize,
    count: impl Fn(&str) -> Result<usize>,
) -> Result<()> {
    let command_result = |message: &ConversationMessage| {
        matches!(message, ConversationMessage::ToolResult { name, .. }
            if crate::tools::kind_of(name).has_command_output())
    };
    let exchanges = messages.split_mut(|message| {
        !matches!(
            message,
            ConversationMessage::ToolCall(_) | ConversationMessage::ToolResult { .. }
        )
    });
    for exchange in exchanges {
        let count_results = exchange
            .iter()
            .filter(|message| command_result(message))
            .count();
        if count_results == 0 {
            continue;
        }
        let per_result = tail_budget / 2 / count_results;
        for message in exchange
            .iter_mut()
            .filter(|message| command_result(message))
        {
            let ConversationMessage::ToolResult { content, .. } = message else {
                unreachable!();
            };
            let mut tokens = count(content)?;
            if tokens <= per_result {
                continue;
            }
            let mut result: Value = serde_json::from_str(content)?;
            while tokens > per_result {
                if trim_command_stream(&mut result) == 0 {
                    if tokens + 64 > tail_budget {
                        eyre::bail!("command result metadata exceeds available context");
                    }
                    break;
                }
                *content = serde_json::to_string(&result)?;
                tokens = count(content)?;
            }
        }
    }
    Ok(())
}

/// Expand probe context and tool exchanges before dialect-specific encoding.
pub fn prepare(history: &[Message]) -> Result<Vec<ConversationMessage>> {
    let mut out = Vec::new();
    for (index, message) in history.iter().enumerate() {
        match message {
            Message::Context(context) => {
                if index != 0 {
                    return Err(eyre!("session context must be the first history entry"));
                }
                out.extend(super::prompting::expand(context));
            }
            Message::System(content) => out.push(ConversationMessage::System(content.clone())),
            Message::Developer(content) => {
                out.push(ConversationMessage::Developer(content.clone()))
            }
            Message::User(content) => out.push(ConversationMessage::User(content.clone())),
            Message::Reasoning(content) => out.push(ConversationMessage::Thinking(content.clone())),
            Message::Assistant(content) => out.push(ConversationMessage::Answer(content.clone())),
            Message::Tool(content) => push_tool_history(&mut out, content),
        }
    }
    Ok(out)
}

/// Split one probe tool record into the call and result it holds; unreadable records become results.
fn push_tool_history(out: &mut Vec<ConversationMessage>, content: &str) {
    let value = serde_json::from_str::<Value>(content);
    let Ok(value) = value else {
        out.push(ConversationMessage::ToolResult {
            name: "tool".to_string(),
            content: content.to_string(),
        });
        return;
    };

    let name = value
        .get("tool")
        .and_then(Value::as_str)
        .unwrap_or("tool")
        .to_string();
    if let Some(arguments) = value.get("arguments") {
        out.push(ConversationMessage::ToolCall(ToolCall {
            name: name.clone(),
            arguments: arguments.clone(),
        }));
    }
    if let Some(result) = value.get("result") {
        let content = result
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| result.to_string());
        out.push(ConversationMessage::ToolResult {
            name: name.clone(),
            content,
        });
    }
    if value.get("arguments").is_none() && value.get("result").is_none() {
        out.push(ConversationMessage::ToolResult {
            name,
            content: content.to_string(),
        });
    }
}

//! Neutral conversation data shared by the hub-side adapters.
use crate::protocol::{Effort, Message, Note, Setting};
use crate::tools::ParamType;
use eyre::{Result, WrapErr};
use serde_json::{Map, Value, json};

/// A conversation with every message worded, ready for a dialect to frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationMessage {
    /// Requested reasoning effort, stated wherever each dialect's chat template expects it.
    Effort(Effort),
    /// The harness's standing instructions, the same for every invocation.
    Instructions(String),
    /// A fact about this invocation (the date), placed after all standing text.
    Fact(String),
    /// The harness's word in the middle of the conversation.
    Note(String),
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
    /// Leading entries that dialects fold into their header.
    pub fn is_preamble(&self) -> bool {
        matches!(
            self,
            Self::Effort(_) | Self::Instructions(_) | Self::Fact(_)
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

/// A tool's name, description, and parameters, as the model is told them.
pub type ToolSpec = (&'static str, &'static str, &'static [crate::tools::Param]);

/// Every tool the model may call, the probe's and the hub's own, by name.
pub fn tool_specs() -> Vec<ToolSpec> {
    let probe = crate::tools::all_tools().into_iter();
    let probe = probe.map(|(name, (description, _, parameters))| (name, description, parameters));
    let mut tools: Vec<_> = probe.chain([crate::hub::transcript::RECALL_SPEC]).collect();
    tools.sort_by_key(|(name, ..)| *name);
    tools
}

/// Declare every tool as an OpenAI-style function schema for the chat templates.
pub fn tool_schemas() -> Vec<Value> {
    tool_specs()
        .into_iter()
        .map(|(name, description, parameters)| {
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

/// Word the setting and the conversation: standing instructions first, then per-invocation
/// facts, the conversation itself, and where the answer goes last, so a different redirect
/// leaves everything before it reusable.
pub fn prepare(setting: &Setting, history: &[Message]) -> Result<Vec<ConversationMessage>> {
    let mut out = super::prompting::head(setting);
    for message in history {
        let worded = match message {
            Message::User(text) => ConversationMessage::User(text.clone()),
            Message::Input(data) => ConversationMessage::User(super::prompting::input(data)),
            Message::Note(note @ Note::Restart { chunk, .. }) => {
                let instructions = ConversationMessage::Note(super::prompting::note(note));
                out.push(instructions);
                ConversationMessage::User(super::prompting::transcript(chunk))
            }
            Message::Note(note) => ConversationMessage::Note(super::prompting::note(note)),
            Message::Thinking(text) => ConversationMessage::Thinking(text.clone()),
            Message::Answer(text) => ConversationMessage::Answer(text.clone()),
            Message::Call(call) => ConversationMessage::ToolCall(ToolCall {
                name: call.name.clone(),
                arguments: serde_json::from_str(&call.arguments)
                    .wrap_err_with(|| format!("tool call {} has invalid arguments", call.name))?,
            }),
            Message::Result(result) => ConversationMessage::ToolResult {
                name: result.name.clone(),
                content: result.content.clone(),
            },
        };
        out.push(worded);
    }
    if let Some(target) = &setting.redirection {
        out.push(ConversationMessage::Note(super::prompting::redirection(
            target,
        )));
    }
    Ok(out)
}

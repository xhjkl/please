//! Neutral conversation data shared by the hub-side adapters.
use super::prompting;
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
    ToolCall(FunctionCall),
    ToolResult {
        name: String,
        content: String,
    },
}

/// A dialect-neutral call, shared by decoded output and replayed tool history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

/// Split off the leading entries that dialects fold into their header.
pub fn split_preamble(
    messages: &[ConversationMessage],
) -> (&[ConversationMessage], &[ConversationMessage]) {
    let preamble = messages.iter().take_while(|message| {
        matches!(
            message,
            ConversationMessage::Effort(_)
                | ConversationMessage::Instructions(_)
                | ConversationMessage::Fact(_)
        )
    });
    messages.split_at(preamble.count())
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
        ("stdout", "stdout_bytes_omitted")
    } else {
        ("stderr", "stderr_bytes_omitted")
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
    let exchanges = messages.split_mut(|message| {
        !matches!(
            message,
            ConversationMessage::ToolCall(_) | ConversationMessage::ToolResult { .. }
        )
    });
    for exchange in exchanges {
        let contents: Vec<_> = exchange
            .iter_mut()
            .filter_map(|message| match message {
                ConversationMessage::ToolResult { name, content }
                    if crate::tools::kind_of(name).has_command_output() =>
                {
                    Some(content)
                }
                _ => None,
            })
            .collect();
        if contents.is_empty() {
            continue;
        }
        let per_result = tail_budget / 2 / contents.len();
        for content in contents {
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
    let mut out = prompting::head(setting);
    for message in history {
        let worded = match message {
            Message::User(text) => ConversationMessage::User(text.clone()),
            Message::Input(data) => {
                out.push(ConversationMessage::Note(prompting::INPUT.into()));
                ConversationMessage::User(data.clone())
            }
            // The draft is the model's own message, and the chunk is material, not instructions.
            Message::Note(note @ Note::Summarize { draft, chunk }) => {
                out.push(ConversationMessage::Note(prompting::note(note)));
                if let Some(draft) = draft {
                    out.push(ConversationMessage::Answer(draft.clone()));
                }
                ConversationMessage::User(chunk.clone())
            }
            Message::Note(note) => ConversationMessage::Note(prompting::note(note)),
            Message::Thinking(text) => ConversationMessage::Thinking(text.clone()),
            Message::Answer(text) => ConversationMessage::Answer(text.clone()),
            Message::Call(call) => ConversationMessage::ToolCall(FunctionCall {
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
        out.push(ConversationMessage::Note(prompting::redirection(target)));
    }
    Ok(out)
}

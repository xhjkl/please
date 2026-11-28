//! A prompt before framing: its preamble, and its messages as worded from the protocol's
//! neutral history, with what every dialect needs to prepare them.
use crate::hub::wording;
use crate::protocol::{Effort, Message, Note, Setting, ToolCall};
use crate::tools::spec::{ParamKind, Spec};
use eyre::{Result, WrapErr};
use serde_json::{Map, Value, json};

/// What opens every prompt. The instructions are the same for every invocation, so with the
/// tool declarations they make the head a hub may prefill once and cache. This invocation's
/// facts, the date and the effort, each family states in its own words: after the head,
/// unless the family heeds one only inside it, as gpt-oss does the effort.
#[derive(Debug, Clone, PartialEq)]
pub struct Preamble {
    pub instructions: Vec<&'static str>,
    pub today: String,
    pub effort: Effort,
}

impl Preamble {
    /// Open a prompt for an invocation in `setting`.
    pub fn new(setting: &Setting) -> Self {
        Self {
            instructions: vec![wording::ROLE, wording::TOOL_GUIDANCE.trim()],
            today: setting.today.clone(),
            effort: setting.effort,
        }
    }
}

/// The effort as a family with only three names for it hears it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    Low,
    Medium,
    High,
}

impl Grade {
    /// Name the effort whose level on Inkling's scale is nearest to `effort`'s, so each of
    /// `PLEASE_TRY`'s words comes back as itself.
    pub fn of(effort: Effort) -> Self {
        let level = effort.level();
        let halfway = |one: Effort, other: Effort| (one.level() + other.level()) / 2.0;
        if level < halfway(Effort::LOW, Effort::MEDIUM) {
            return Self::Low;
        }
        if level < halfway(Effort::MEDIUM, Effort::HIGH) {
            return Self::Medium;
        }
        Self::High
    }
}

/// One message of a prompt after its preamble: a `Message` as the model reads it, with notes
/// in words and calls as dialects write them, ready for a dialect to frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptMessage {
    /// The harness's word in the middle of a turn.
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

/// A call as dialects read and write it, in decoded output and replayed history alike: a name
/// and JSON arguments, with no ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

impl FunctionCall {
    /// This call as the protocol carries it and a room keeps it, under `id`.
    pub fn with_id(&self, id: u64) -> ToolCall {
        ToolCall {
            id,
            name: self.name.clone(),
            arguments: self.arguments.to_string(),
        }
    }
}

/// Encoded prompt and how many of its leading tokens are the same for every invocation, at
/// this effort where the family's head holds it: the head a hub may prefill once and cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub tokens: Vec<u32>,
    pub head: usize,
}

/// Newly visible generated text, by the section it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delta {
    Answer(String),
    Thinking(String),
}

/// Every tool the model may call, the probe's and the hub's own, by name.
pub fn tool_specs() -> Vec<Spec> {
    let probe = crate::tools::all_tools().into_values();
    let probe = probe.map(|tool| tool.spec);
    let mut specs: Vec<_> = probe.chain([crate::hub::transcript::RECALL_SPEC]).collect();
    specs.sort_by_key(|spec| spec.name);
    specs
}

/// Declare every tool as an OpenAI-style function schema for the chat templates.
pub fn tool_schemas() -> Vec<Value> {
    tool_specs()
        .into_iter()
        .map(|spec| {
            let mut properties = Map::new();
            let mut required = Vec::new();
            for parameter in spec.params {
                let mut schema = match &parameter.kind {
                    ParamKind::String => json!({"type": "string"}),
                    ParamKind::StringArray => json!({"type": "array", "items": {"type": "string"}}),
                    ParamKind::Number => json!({"type": "number"}),
                    ParamKind::Choice(choices) => json!({"type": "string", "enum": choices}),
                };
                schema["description"] = parameter.description.into();
                if parameter.required {
                    required.push(parameter.name);
                }
                properties.insert(parameter.name.to_string(), schema);
            }
            json!({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
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
    messages: &mut [PromptMessage],
    tail_budget: usize,
    count: impl Fn(&str) -> Result<usize>,
) -> Result<()> {
    let exchanges = messages.split_mut(|message| {
        !matches!(
            message,
            PromptMessage::ToolCall(_) | PromptMessage::ToolResult { .. }
        )
    });
    for exchange in exchanges {
        let contents: Vec<_> = exchange
            .iter_mut()
            .filter_map(|message| match message {
                PromptMessage::ToolResult { name, content }
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

/// Word the history for the model, with where the answer goes last, so a different redirect
/// leaves everything before it reusable.
pub fn prepare(setting: &Setting, history: &[Message]) -> Result<Vec<PromptMessage>> {
    let mut out = Vec::new();
    for message in history {
        let worded = match message {
            Message::User(text) => PromptMessage::User(text.clone()),
            Message::Input(data) => {
                out.push(PromptMessage::Note(wording::INPUT.into()));
                PromptMessage::User(data.clone())
            }
            // The draft is the model's own message, and the chunk is material, not instructions.
            Message::Note(note @ Note::Summarize { draft, chunk }) => {
                out.push(PromptMessage::Note(wording::note(note)));
                if let Some(draft) = draft {
                    out.push(PromptMessage::Answer(draft.clone()));
                }
                PromptMessage::User(chunk.clone())
            }
            Message::Note(note) => PromptMessage::Note(wording::note(note)),
            Message::Thinking(text) => PromptMessage::Thinking(text.clone()),
            Message::Answer(text) => PromptMessage::Answer(text.clone()),
            Message::Call(call) => PromptMessage::ToolCall(FunctionCall {
                name: call.name.clone(),
                arguments: serde_json::from_str(&call.arguments)
                    .wrap_err_with(|| format!("tool call {} has invalid arguments", call.name))?,
            }),
            Message::Result(result) => PromptMessage::ToolResult {
                name: result.name.clone(),
                content: result.content.clone(),
            },
        };
        out.push(worded);
    }
    if let Some(target) = &setting.redirection {
        out.push(PromptMessage::Note(wording::redirection(target)));
    }
    Ok(out)
}

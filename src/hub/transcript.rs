//! Reading a room's transcript back: abbreviated for compacting, listed and quoted for the
//! recall tool, which the hub runs itself.
use super::rooms::{Entry, Room};
use crate::protocol::{Message, ToolCall, ToolResult};
use crate::tools::spec::{Param, ParamKind, Spec};
use serde_json::{Value, json};

/// The hub's own tool: the transcript, restarts included, lives with the hub.
pub const RECALL: &str = "recall";

pub const RECALL_SPEC: Spec = Spec {
    name: RECALL,
    description: "List this conversation's transcript, including entries before a restart, or read one entry in full",
    params: &[
        Param {
            name: "entry",
            description: "Entry number, such as 12; omit to list every entry",
            kind: ParamKind::String,
            required: false,
        },
        Param {
            name: "offset",
            description: "Character to start from, for text longer than one reply; default 0",
            kind: ParamKind::Number,
            required: false,
        },
    ],
};

/// Characters one recall returns at most, however roomy the context.
pub const RECALL_PAGE: usize = 8000;

/// Characters of each kind of body a chunk to summarize shows before clipping it.
const SHOWN_TEXT: usize = 6000;
const SHOWN_REASONING: usize = 800;
const SHOWN_DATA: usize = 800;
const SHOWN_ARGUMENTS: usize = 400;

/// Answer a recall call: the list of entries, or one entry in full, a page at a time.
/// Pages hold at most `page` characters.
pub fn recall(room: &Room, arguments: &Value, page: usize) -> Value {
    let entry = match arguments.get("entry") {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) => Some(name.clone()),
        Some(Value::Number(number)) => Some(number.to_string()),
        Some(_) => return json!({ "error": "entry must be an entry number" }),
    };
    let offset = arguments.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let read = match entry {
        None => room.transcript().map(|entries| index(&entries)),
        Some(name) => room.read(&name),
    };
    let text = match read {
        Err(error) => return json!({ "error": format!("{error:#}") }),
        Ok(text) => text,
    };
    let length = text.chars().count();
    if offset == 0 && length <= page {
        return Value::String(text);
    }
    let page: String = text.chars().skip(offset).take(page).collect();
    let to = offset + page.chars().count();
    json!({ "text": page, "from": offset, "to": to, "length": length })
}

/// One line per entry: its name, kind, size, and opening words.
fn index(entries: &[Entry]) -> String {
    let lines: Vec<_> = entries
        .iter()
        .map(|entry| {
            let (kind, body) = kind_and_body(&entry.message);
            let first = body
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("");
            let restarted = if entry.old { " (before a restart)" } else { "" };
            format!(
                "{} {kind}, {} characters{restarted}: {}",
                entry.number,
                body.chars().count(),
                clip(first, 120)
            )
        })
        .collect();
    lines.join("\n")
}

/// A conversation's entries, each abbreviated the same way every time, grouped into chunks
/// of about `budget` bytes, oldest first; no entry is shown longer than half a chunk.
pub fn chunks(conversation: &[Entry], budget: usize) -> Vec<String> {
    let longest = budget / 2;
    let mut chunks: Vec<String> = Vec::new();
    for (index, entry) in conversation.iter().enumerate() {
        // A conversation that restarted opens with the summary it restarted from.
        // Numbering goes on across restarts, so its number tells even once `old` is erased.
        let summary = index == 0 && entry.number > 1 && matches!(entry.message, Message::Answer(_));
        let block = block(entry, summary, longest);
        match chunks.last_mut() {
            Some(chunk) if chunk.len() + block.len() <= budget => {
                chunk.push_str("\n\n");
                chunk.push_str(&block);
            }
            _ => chunks.push(block),
        }
    }
    chunks
}

/// An entry headed by its number and kind: requests and answers nearly whole, reasoning and
/// data clipped, a call on one line. A `summary` is headed as the message its conversation
/// restarted from rather than as an answer.
fn block(entry: &Entry, summary: bool, longest: usize) -> String {
    let number = entry.number;
    let (kind, body) = kind_and_body(&entry.message);
    let kind = match summary {
        true => "the message the conversation last restarted from".into(),
        false => kind,
    };
    let shown = match &entry.message {
        Message::Call(ToolCall { arguments, .. }) => {
            let more = match arguments.chars().count() > SHOWN_ARGUMENTS {
                true => " ...",
                false => "",
            };
            return format!(
                "[{number}] {kind} {}{more}",
                clip(arguments, SHOWN_ARGUMENTS)
            );
        }
        Message::User(_) | Message::Answer(_) => SHOWN_TEXT,
        Message::Thinking(_) => SHOWN_REASONING,
        _ => SHOWN_DATA,
    };
    let shown = shown.min(longest.max(SHOWN_DATA));
    let length = body.chars().count();
    if length <= shown {
        return format!("[{number}] {kind}\n{body}");
    }
    format!(
        "[{number}] {kind}, abbreviated\n{}\n[... {} more characters; recall {number} for all of it]",
        clip(body, shown),
        length - shown
    )
}

fn kind_and_body(message: &Message) -> (String, &str) {
    match message {
        Message::User(text) => ("user".into(), text),
        Message::Input(data) => ("piped input".into(), data),
        Message::Thinking(text) => ("reasoning".into(), text),
        Message::Answer(text) => ("answer".into(), text),
        Message::Call(ToolCall {
            name, arguments, ..
        }) => (format!("call {name}"), arguments),
        Message::Result(ToolResult { name, content, .. }) => (format!("result of {name}"), content),
        Message::Note(_) => ("note".into(), ""),
    }
}

/// The first `count` characters.
fn clip(text: &str, count: usize) -> &str {
    text.char_indices()
        .nth(count)
        .map_or(text, |(end, _)| &text[..end])
}

#[cfg(test)]
mod tests;

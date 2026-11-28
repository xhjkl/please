//! Reading a room's transcript back: abbreviated for compacting, listed and quoted for the
//! recall tool, which the hub runs itself.
use super::rooms::{Entry, Room};
use crate::protocol::{Message, ToolCall, ToolResult};
use crate::tools::{Param, ParamKind, Spec};
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
        None => room.entries().map(|entries| index(&entries)),
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

/// Entries since the last restart, each abbreviated the same way every time, grouped into
/// chunks of about `budget` bytes, oldest first; no entry is shown longer than half a chunk.
pub fn chunks(entries: &[Entry], budget: usize) -> Vec<String> {
    let longest = budget / 2;
    let mut chunks: Vec<String> = Vec::new();
    for (index, entry) in entries.iter().filter(|entry| !entry.old).enumerate() {
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
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn some_messages() -> Vec<Message> {
        vec![
            Message::User("find me a gc implementation in this codebase".into()),
            Message::Thinking("x".repeat(2000)),
            Message::Call(ToolCall {
                id: 3,
                name: "list_files".into(),
                arguments: r#"{"path":"./core"}"#.into(),
            }),
            Message::Result(ToolResult {
                id: 3,
                name: "list_files".into(),
                content: "alloc.rs\n".repeat(400),
            }),
            Message::Answer("Nothing in ./core.".into()),
        ]
    }

    #[test]
    fn chunks_abbreviate_bulk_and_keep_requests_whole() -> Result<(), eyre::Report> {
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        room.append(&some_messages())?;
        let all = chunks(&room.entries()?, 1 << 20);
        assert_eq!(all.len(), 1);
        let text = &all[0];
        assert!(text.starts_with("[1] user\nfind me a gc implementation in this codebase"));
        assert!(text.contains("[2] reasoning, abbreviated\n"));
        assert!(text.contains("1200 more characters; recall 2 for all of it"));
        assert!(text.contains(r#"[3] call list_files {"path":"./core"}"#));
        assert!(text.contains("[4] result of list_files, abbreviated"));
        assert!(text.ends_with("[5] answer\nNothing in ./core."));
        // A small budget splits between entries, never inside one.
        let small = chunks(&room.entries()?, 900);
        assert!(small.len() > 1 && small.iter().all(|chunk| chunk.starts_with('[')));
        Ok(())
    }

    #[test]
    fn the_message_restarted_from_is_named_even_once_old_is_erased() -> Result<(), eyre::Report> {
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        room.append(&some_messages())?;
        room.restart(&[Message::Answer("You asked me to find a GC.".into())])?;
        room.append(&[Message::User("look in ./gc".into())])?;
        let restarted_from = "[6] the message the conversation last restarted from\n";
        let kept = chunks(&room.entries()?, 1 << 20);
        assert!(kept[0].starts_with(restarted_from));
        std::fs::remove_dir_all(room.dir().join("old"))?;
        let erased = chunks(&room.entries()?, 1 << 20);
        assert_eq!(erased, kept);
        Ok(())
    }

    #[test]
    fn recall_lists_entries_and_reads_them_across_restarts() -> Result<(), eyre::Report> {
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        room.append(&some_messages())?;
        room.restart(&[Message::Answer("You asked me to find a GC.".into())])?;
        let list = recall(&room, &json!({}), RECALL_PAGE);
        let list = list.as_str().unwrap();
        assert!(list.starts_with("1 user, 44 characters (before a restart): find me a gc"));
        assert!(list.ends_with("6 answer, 26 characters: You asked me to find a GC."));
        // A long entry comes back whole, a page at a time.
        let page = recall(&room, &json!({"entry": 4, "offset": 10}), RECALL_PAGE);
        let length = "result list_files 3\n\n".len() + "alloc.rs\n".len() * 400;
        assert_eq!(page["from"].as_u64(), Some(10));
        assert_eq!(page["length"].as_u64(), Some(length as u64));
        let first = recall(&room, &json!({"entry": 1}), RECALL_PAGE);
        assert_eq!(
            first,
            "user\n\nfind me a gc implementation in this codebase"
        );
        let missing = recall(&room, &json!({"entry": "99"}), RECALL_PAGE);
        assert!(missing["error"].as_str().is_some());
        // The next compaction reads the summary as such, not as an ordinary answer.
        let next = chunks(&room.entries()?, 1 << 20);
        assert_eq!(
            next,
            ["[6] the message the conversation last restarted from\nYou asked me to find a GC."]
        );
        Ok(())
    }
}

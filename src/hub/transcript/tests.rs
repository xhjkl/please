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
    let kept = chunks(&room.entries()?, 1 << 20);
    assert!(kept[0].starts_with("[6] ") && !kept[0].starts_with("[6] answer"));
    assert!(kept[0].contains("restarted from"));
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
    assert_eq!(next.len(), 1);
    assert!(next[0].contains("restarted from") && !next[0].contains("answer"));
    assert!(next[0].ends_with("\nYou asked me to find a GC."));
    Ok(())
}

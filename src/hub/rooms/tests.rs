use super::*;
use crate::protocol::Note;
use crate::test_support::TempDir;

fn some_messages() -> Vec<Message> {
    vec![
        Message::Input("diff --git a/x b/x".into()),
        Message::User("find me a gc implementation".into()),
        Message::Note(Note::Pending(vec![3])),
        Message::Thinking("I'll list the files.\n\nThen read.".into()),
        Message::Call(ToolCall {
            id: 4,
            name: "list_files".into(),
            arguments: r#"{"path":"."}"#.into(),
        }),
        Message::Result(ToolResult {
            id: 4,
            name: "list_files".into(),
            content: "core\nlang\ngc\n".repeat(1000),
        }),
        Message::Answer("Nothing in ./core.\n".into()),
    ]
}

#[test]
fn messages_round_trip_through_plain_files() -> Result<()> {
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    room.append(&some_messages())?;
    let kept: Vec<_> = some_messages()
        .into_iter()
        .filter(|message| !matches!(message, Message::Note(_)))
        .collect();
    assert_eq!(room.history()?, kept);
    let user = std::fs::read_to_string(room.dir().join("00000002"))?;
    assert_eq!(user, "user\n\nfind me a gc implementation");
    let call = std::fs::read_to_string(room.dir().join("00000004"))?;
    assert_eq!(call, "call list_files 4\n\n{\"path\":\".\"}");
    // However long, the result lies inside its entry, under its call's ID.
    let result = format!("result list_files 4\n\n{}", "core\nlang\ngc\n".repeat(1000));
    assert_eq!(
        std::fs::read_to_string(room.dir().join("00000005"))?,
        result
    );
    assert_eq!(room.read("[5]")?, result);
    // Text bodies never go to the attachments.
    let listed = |dir: &Path| {
        let names = std::fs::read_dir(dir)?
            .map(|item| item.map(|item| item.file_name().to_string_lossy().into_owned()));
        let mut names = names.collect::<std::io::Result<Vec<_>>>()?;
        names.sort();
        Ok::<_, std::io::Error>(names)
    };
    let entries = [
        "...", "00000001", "00000002", "00000003", "00000004", "00000005", "00000006",
    ];
    assert_eq!(listed(room.dir())?, entries);
    assert!(listed(&room.dir().join(ATTACHMENTS))?.is_empty());
    Ok(())
}

#[test]
fn a_restart_sets_entries_aside_and_numbering_goes_on() -> Result<()> {
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    room.append(&some_messages())?;
    let summary = Message::Answer("You asked me to find a GC; next I'll read ./gc.".into());
    room.restart(std::slice::from_ref(&summary))?;
    assert_eq!(room.history()?, std::slice::from_ref(&summary));
    assert!(room.dir().join("old/00000002").exists());
    assert!(room.dir().join("00000007").exists());
    assert_eq!(room.read("2")?, "user\n\nfind me a gc implementation");
    room.append(&[Message::User("thanks".into())])?;
    let entries = room.transcript()?;
    assert_eq!(entries.len(), 8);
    assert_eq!(entries.iter().filter(|entry| entry.old).count(), 6);
    assert_eq!(entries.last().map(|entry| entry.number), Some(8));
    // Erasing what was set aside leaves the conversation and its numbering be.
    std::fs::remove_dir_all(room.dir().join("old"))?;
    assert_eq!(room.transcript()?.len(), 2);
    assert_eq!(room.next()?, 9);
    Ok(())
}

#[test]
fn an_epoch_lasts_until_the_conversation_restarts() -> Result<()> {
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    room.append(&some_messages())?;
    let first = room.epoch()?;
    room.append(&[Message::User("more".into())])?;
    assert_eq!(room.epoch()?, first);
    room.restart(&[])?;
    let second = room.epoch()?;
    assert_ne!(second, first);
    // Erasing what was set aside lets numbering start over, yet the conversation is the new one.
    std::fs::remove_dir_all(room.dir().join("old"))?;
    room.append(&[Message::User("again".into())])?;
    assert_eq!(room.next()?, 2);
    assert_eq!(room.epoch()?, second);
    // A room emptied by hand is a conversation no hub has seen.
    std::fs::remove_file(room.dir().join(".epoch"))?;
    assert_ne!(room.epoch()?, second);
    Ok(())
}

#[test]
fn foreign_files_and_unreadable_entries_cost_nothing_else() -> Result<()> {
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    std::fs::write(room.dir().join("notes.txt"), "mine")?;
    std::fs::write(room.dir().join(".00000001.partial"), "user\n\nhalf")?;
    assert!(room.conversation()?.is_empty());
    std::fs::write(room.dir().join("00000001"), "letter\n\nhi")?;
    std::fs::create_dir(room.dir().join("old"))?;
    std::fs::write(room.dir().join("old/00000002"), "")?;
    std::fs::write(room.dir().join("00000003"), "call list_files\n\n{}")?;
    room.append(&[Message::User("still here".into())])?;
    assert_eq!(room.history()?, [Message::User("still here".into())]);
    // A restart sets entries aside without reading them.
    room.restart(&[])?;
    assert!(room.conversation()?.is_empty() && room.dir().join("old/00000001").exists());
    Ok(())
}

#[test]
fn odd_names_and_privacy_survive() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    let spaced = Message::Call(ToolCall {
        id: 1,
        name: "list files 2".into(),
        arguments: "{}".into(),
    });
    room.append(std::slice::from_ref(&spaced))?;
    room.append(&[Message::Answer("x".into())])?;
    assert_eq!(room.history()?[0], spaced);
    let nameless = Message::Result(ToolResult {
        id: 1,
        name: String::new(),
        content: "{}".into(),
    });
    assert!(room.append(&[nameless]).is_err());
    let mode = |path: &Path| std::fs::metadata(path).map(|m| m.permissions().mode() & 0o777);
    assert_eq!(mode(room.dir())?, 0o700);
    assert_eq!(mode(&room.dir().join("00000002"))?, 0o600);
    room.set_last_delivery(42)?;
    assert_eq!(room.last_delivery(), Some(42));
    Ok(())
}

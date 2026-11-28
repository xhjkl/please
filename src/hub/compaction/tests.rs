use super::*;
use crate::protocol::{ToolCall, ToolResult};
use crate::test_support::TempDir;

#[test]
fn a_restart_keeps_as_much_of_the_turn_in_progress_as_fits() -> Result<()> {
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    room.append(&[Message::User("old".into()), Message::Answer("done".into())])?;
    room.restart(&[Message::Answer("You asked for the old thing.".into())])?;
    room.append(&[Message::Input("data".into()), Message::User("now".into())])?;
    for step in 0..3 {
        let content = format!("step {step}");
        let id = room.next()? + 1;
        let call = Message::Call(ToolCall {
            id,
            name: "read_file".into(),
            arguments: "{}".into(),
        });
        room.append(&[Message::Thinking(content.clone()), call])?;
        let result = Message::Result(ToolResult {
            id,
            name: "read_file".into(),
            content,
        });
        room.append(&[result])?;
    }
    // Entries 1 and 2 are set aside, 3 is the restart message, and the turn is 4 to 14.
    let split = |most: usize| {
        let fits = |kept: &[Message]| Ok(kept.len() <= most);
        let (folded, kept) = cutover(room.conversation()?, fits)?;
        let last = folded.last().map(|entry| entry.number);
        Ok::<_, eyre::Report>((last, kept))
    };
    let request = [Message::Input("data".into()), Message::User("now".into())];
    // The whole turn fits: only what came before it is folded.
    let (last, kept) = split(11)?;
    assert_eq!((last, kept.len()), (Some(3), 11));
    // The request and the newest step.
    let (last, kept) = split(5)?;
    assert_eq!(last, Some(11));
    assert_eq!(kept[..2], request);
    assert_eq!(kept[2], Message::Thinking("step 2".into()));
    assert_eq!(kept.len(), 5);
    // The request alone, then nothing; the whole turn is folded either way.
    let (last, kept) = split(2)?;
    assert_eq!((last, kept), (Some(14), request.to_vec()));
    let (last, kept) = split(1)?;
    assert_eq!((last, kept), (Some(14), vec![]));
    Ok(())
}

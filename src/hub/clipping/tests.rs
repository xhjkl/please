use super::*;
use crate::protocol::{ToolCall, ToolResult};

/// Every message weighs ten tokens.
fn size(history: &[Message]) -> Result<usize> {
    Ok(history.len() * 10)
}

/// A user turn followed by `steps` tool steps of reasoning, a call, and its result.
fn turn(history: &mut Vec<Message>, request: &str, steps: usize) {
    history.push(Message::User(request.into()));
    for step in 0..steps {
        history.push(Message::Thinking(format!("{request} {step}")));
        history.push(Message::Call(ToolCall {
            id: step as u64,
            name: "read_file".into(),
            arguments: "{}".into(),
        }));
        history.push(Message::Result(ToolResult {
            id: step as u64,
            name: "read_file".into(),
            content: format!("{request} {step}"),
        }));
    }
}

#[test]
fn repeated_cuts_merge_into_one_range() {
    let ranges = [0..4, 0..7, 9..12, 7..8, 3..5, 12..12, 20..22];
    assert_eq!(merged(ranges.into_iter()), [0..8, 9..12, 20..22]);
}

#[test]
fn nothing_is_dropped_while_the_prompt_fits() -> Result<()> {
    let mut history = vec![];
    turn(&mut history, "a", 3);
    assert!(choose(&history, &[], 100, 50, size)?.is_empty());
    Ok(())
}

#[test]
fn older_turns_go_first_and_whole() -> Result<()> {
    let mut history = vec![];
    turn(&mut history, "a", 2);
    turn(&mut history, "b", 2);
    turn(&mut history, "c", 1);
    // 18 messages: over a limit of 120, down to at most 110.
    let dropped = choose(&history, &[], 120, 110, size)?;
    assert_eq!(dropped, std::iter::once(0..7).collect::<Vec<_>>());
    let kept = visible(&history, &dropped);
    assert!(matches!(&kept[0], Message::User(text) if text == "b"));
    Ok(())
}

#[test]
fn a_long_turn_keeps_its_request_and_newest_step() -> Result<()> {
    let mut history = vec![];
    turn(&mut history, "a", 10);
    let dropped = choose(&history, &[], 150, 100, size)?;
    let kept = visible(&history, &dropped);
    assert!(matches!(&kept[0], Message::User(text) if text == "a"));
    assert!(matches!(&kept[1], Message::Thinking(_)));
    assert!(matches!(kept.last(), Some(Message::Result(result)) if result.content == "a 9"));
    assert!(kept.len() * 10 <= 100);
    Ok(())
}

#[test]
fn later_requests_reuse_the_choice_until_they_overflow_again() -> Result<()> {
    let mut history = vec![];
    turn(&mut history, "a", 10);
    let dropped = choose(&history, &[], 150, 100, size)?;
    let first = visible(&history, &dropped);
    history.push(Message::Thinking("next".into()));
    let again = choose(&history, &dropped, 150, 100, size)?;
    assert_eq!(again, dropped);
    // The new prompt extends the previous one, so a kept session only feeds the suffix.
    let next = visible(&history, &again);
    assert_eq!(format!("{:?}", &next[..first.len()]), format!("{first:?}"));
    Ok(())
}

#[test]
fn piped_input_stays_with_its_request() -> Result<()> {
    let mut history = vec![];
    turn(&mut history, "a", 3);
    history.push(Message::Input("data".into()));
    turn(&mut history, "b", 6);
    let dropped = choose(&history, &[], 200, 120, size)?;
    let kept = visible(&history, &dropped);
    assert!(matches!(&kept[0], Message::Input(data) if data == "data"));
    assert!(matches!(&kept[1], Message::User(text) if text == "b"));
    Ok(())
}

#[test]
fn after_a_restart_older_steps_go_and_the_summary_stays() -> Result<()> {
    let mut history = vec![Message::Answer("You asked me to read three files.".into())];
    for step in 0..8 {
        history.push(Message::Thinking(format!("step {step}")));
        history.push(Message::Call(ToolCall {
            id: step as u64,
            name: "read_file".into(),
            arguments: "{}".into(),
        }));
        history.push(Message::Result(ToolResult {
            id: step as u64,
            name: "read_file".into(),
            content: format!("{step}"),
        }));
    }
    let dropped = choose(&history, &[], 150, 100, size)?;
    let kept = visible(&history, &dropped);
    assert!(matches!(&kept[0], Message::Answer(_)));
    assert!(matches!(kept.last(), Some(Message::Result(result)) if result.content == "7"));
    assert!(kept.len() * 10 <= 100);
    Ok(())
}

#[test]
fn an_unfittable_newest_step_is_an_error() {
    let mut history = vec![];
    turn(&mut history, "a", 2);
    assert!(choose(&history, &[], 30, 20, size).is_err());
}

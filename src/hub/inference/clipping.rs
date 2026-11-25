//! Which conversation messages a prompt leaves out once the context fills.
//!
//! Messages are dropped whole, at step boundaries, and only when the rendered prompt no longer
//! fits; then enough go to bring it down to about half the capacity. The choice is kept per
//! session and reapplied to later requests, so their prompts still extend what it consumed.
use crate::protocol::Message;
use eyre::Result;
use std::ops::Range;

/// Copy the probe history without the dropped messages.
pub fn visible(history: &[Message], dropped: &[Range<usize>]) -> Vec<Message> {
    history
        .iter()
        .enumerate()
        .filter(|(index, _)| !dropped.iter().any(|range| range.contains(index)))
        .map(|(_, message)| message.clone())
        .collect()
}

/// List the indices where a user turn or an assistant step begins; cutting elsewhere would
/// separate a tool call from its reasoning or result, or piped input from the request it came with.
pub fn boundaries(history: &[Message]) -> Vec<usize> {
    let tool = |message: &Message| matches!(message, Message::Call(_) | Message::Result(_));
    (1..history.len())
        .filter(|&index| match &history[index] {
            Message::Input(_) => true,
            Message::User(_) => !matches!(history[index - 1], Message::Input(_)),
            message => tool(&history[index - 1]) && !tool(message),
        })
        .collect()
}

/// Find where the turn of the user message at `user` opens: at its piped input, if any.
fn opening(history: &[Message], user: usize) -> usize {
    match user.checked_sub(1).map(|index| &history[index]) {
        Some(Message::Input(_)) => user - 1,
        _ => user,
    }
}

/// Extend `dropped` until the prompt fits. `size` measures a candidate history and `limit`
/// is what must fit; once over it, dropping continues down to `target`. Older user turns go
/// first; within the latest turn its request stays and the oldest steps go, never the last.
pub fn choose(
    history: &[Message],
    dropped: &[Range<usize>],
    limit: usize,
    target: usize,
    mut size: impl FnMut(&[Message]) -> Result<usize>,
) -> Result<Vec<Range<usize>>> {
    if size(&visible(history, dropped))? <= limit {
        return Ok(dropped.to_vec());
    }
    let cuts = boundaries(history);
    // A conversation that restarted may hold no request yet: the message it restarted from
    // anchors it instead.
    let user = history
        .iter()
        .rposition(|message| matches!(message, Message::User(_)))
        .unwrap_or(0);
    let request = opening(history, user);
    // Older turns go whole: only where they open are cut points.
    let older = cuts
        .iter()
        .filter(|&&cut| matches!(history[cut], Message::User(_) | Message::Input(_)))
        .filter(|&&cut| cut <= request)
        .map(|&cut| std::iter::once(0..cut).collect::<Vec<_>>());
    let steps = cuts
        .iter()
        .filter(|&&cut| cut > user + 1)
        .map(|&cut| vec![0..request, user + 1..cut]);
    let mut fitting = None;
    for candidate in older.chain(steps) {
        let candidate: Vec<_> = dropped.iter().cloned().chain(candidate).collect();
        let tokens = size(&visible(history, &candidate))?;
        if tokens <= target {
            return Ok(candidate);
        }
        if tokens <= limit {
            fitting = Some(candidate);
        }
    }
    fitting.ok_or_else(|| eyre::eyre!("the latest request and its newest step exceed the context"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Call, ToolResult};

    /// Every message weighs ten tokens.
    fn size(history: &[Message]) -> Result<usize> {
        Ok(history.len() * 10)
    }

    /// A user turn followed by `steps` tool steps of reasoning, a call, and its result.
    fn turn(history: &mut Vec<Message>, request: &str, steps: usize) {
        history.push(Message::User(request.into()));
        for step in 0..steps {
            history.push(Message::Thinking(format!("{request} {step}")));
            history.push(Message::Call(Call {
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
            history.push(Message::Call(Call {
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
}

//! Which conversation messages a prompt leaves out once the context fills.
//!
//! Messages are dropped whole, at step boundaries, and only when the rendered prompt no longer
//! fits; then enough go to bring it down to about half the capacity. The choice is kept per
//! session and reapplied to later requests, so their prompts still extend what it consumed.
use crate::protocol::Message;
use eyre::Result;
use std::ops::Range;

/// The probe history without the dropped messages.
pub fn visible(history: &[Message], dropped: &[Range<usize>]) -> Vec<Message> {
    history
        .iter()
        .enumerate()
        .filter(|(index, _)| !dropped.iter().any(|range| range.contains(index)))
        .map(|(_, message)| message.clone())
        .collect()
}

/// Indices where a user turn or an assistant step begins; cutting elsewhere would separate a
/// tool call from its reasoning or result.
fn boundaries(history: &[Message], start: usize) -> Vec<usize> {
    (start + 1..history.len())
        .filter(|&index| {
            matches!(history[index], Message::User(_))
                || matches!(history[index - 1], Message::Tool(_))
                    && !matches!(history[index], Message::Tool(_))
        })
        .collect()
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
    let start = usize::from(matches!(history.first(), Some(Message::Context(_))));
    let cuts = boundaries(history, start);
    let request = history
        .iter()
        .rposition(|message| matches!(message, Message::User(_)))
        .filter(|&index| index >= start);
    // Older turns go whole: only their user messages are cut points.
    let older = cuts
        .iter()
        .filter(|&&cut| matches!(history[cut], Message::User(_)))
        .filter(|&&cut| request.is_none_or(|request| cut <= request))
        .map(|&cut| std::iter::once(start..cut).collect::<Vec<_>>());
    let steps = request.into_iter().flat_map(|request| {
        let last = cuts.last().copied().unwrap_or(request);
        cuts.iter()
            .filter(move |&&cut| cut > request + 1 && cut <= last)
            .map(move |&cut| vec![start..request, request + 1..cut])
    });
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
    use crate::protocol::Context;

    fn context() -> Message {
        Message::Context(Context {
            today: "2040-02-03".into(),
            reasoning: "medium".into(),
            stdin: None,
            stdout_redirection: None,
        })
    }

    /// Every message weighs ten tokens.
    fn size(history: &[Message]) -> Result<usize> {
        Ok(history.len() * 10)
    }

    /// A user turn followed by `steps` tool steps of reasoning and one call each.
    fn turn(history: &mut Vec<Message>, request: &str, steps: usize) {
        history.push(Message::User(request.into()));
        for step in 0..steps {
            history.push(Message::Reasoning(format!("{request} {step}")));
            history.push(Message::Tool(format!("{request} {step}")));
        }
    }

    #[test]
    fn nothing_is_dropped_while_the_prompt_fits() -> Result<()> {
        let mut history = vec![context()];
        turn(&mut history, "a", 3);
        assert!(choose(&history, &[], 100, 50, size)?.is_empty());
        Ok(())
    }

    #[test]
    fn older_turns_go_first_and_whole() -> Result<()> {
        let mut history = vec![context()];
        turn(&mut history, "a", 2);
        turn(&mut history, "b", 2);
        turn(&mut history, "c", 1);
        // 14 messages: over a limit of 120, down to at most 90.
        let dropped = choose(&history, &[], 120, 90, size)?;
        assert_eq!(dropped, std::iter::once(1..6).collect::<Vec<_>>());
        let kept = visible(&history, &dropped);
        assert!(matches!(&kept[1], Message::User(text) if text == "b"));
        Ok(())
    }

    #[test]
    fn a_long_turn_keeps_its_request_and_newest_step() -> Result<()> {
        let mut history = vec![context()];
        turn(&mut history, "a", 10);
        let dropped = choose(&history, &[], 150, 100, size)?;
        let kept = visible(&history, &dropped);
        assert!(matches!(&kept[1], Message::User(text) if text == "a"));
        assert!(matches!(&kept[2], Message::Reasoning(_)));
        assert!(matches!(kept.last(), Some(Message::Tool(text)) if text == "a 9"));
        assert!(kept.len() * 10 <= 100);
        Ok(())
    }

    #[test]
    fn later_requests_reuse_the_choice_until_they_overflow_again() -> Result<()> {
        let mut history = vec![context()];
        turn(&mut history, "a", 10);
        let dropped = choose(&history, &[], 150, 100, size)?;
        let first = visible(&history, &dropped);
        history.push(Message::Reasoning("next".into()));
        let again = choose(&history, &dropped, 150, 100, size)?;
        assert_eq!(again, dropped);
        // The new prompt extends the previous one, so a kept session only feeds the suffix.
        let next = visible(&history, &again);
        assert_eq!(format!("{:?}", &next[..first.len()]), format!("{first:?}"));
        Ok(())
    }

    #[test]
    fn an_unfittable_newest_step_is_an_error() {
        let mut history = vec![context()];
        turn(&mut history, "a", 2);
        assert!(choose(&history, &[], 30, 20, size).is_err());
    }
}

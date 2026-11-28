//! Which messages of a history a prompt leaves out once the context fills.
//!
//! Messages are dropped whole, at step boundaries, and only when the rendered prompt no longer
//! fits; then enough go to bring it down to about half the capacity. The choice is kept per
//! session and reapplied to later requests, so their prompts still extend what it consumed.
use crate::protocol::Message;
use eyre::Result;
use std::ops::Range;

/// Copy the history without the dropped messages.
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
pub fn opening(history: &[Message], user: usize) -> usize {
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
    // A conversation that restarted may hold no request yet: the summary it restarted from
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
        let candidate = merged(dropped.iter().cloned().chain(candidate));
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

/// Merge ranges that overlap or touch; each new cut tends to cover the last, so the list would
/// otherwise grow with every clip of a long history.
fn merged(ranges: impl Iterator<Item = Range<usize>>) -> Vec<Range<usize>> {
    let mut ranges: Vec<_> = ranges.filter(|range| !range.is_empty()).collect();
    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

#[cfg(test)]
mod tests;

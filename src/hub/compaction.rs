//! Compacting a conversation: restarting it from a summary of itself, when a prompt nears the
//! context's end in the middle of a turn, or when `/compact` asks.
//!
//! Clipping, by contrast, leaves the conversation as it is and only keeps old messages out of
//! a prompt.
use super::clipping;
use super::inference::Engine;
use super::rooms::{Entry, Room};
use super::step;
use super::transcript;
use crate::protocol::{Destination, Frame, Message, Note, Setting, write_frame};
use eyre::{Result, bail};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// Share of the context a prompt may fill before the conversation is compacted.
const COMPACT_AT: f64 = 0.75;
/// Share of what the instructions leave free that each chunk to summarize may take.
const CHUNK_SHARE: f64 = 0.5;
/// Bytes a token is counted as; code tokens average more, so byte budgets come out smaller
/// than the tokens allow.
pub const BYTES_PER_TOKEN: f64 = 2.5;
/// Tokens kept free around each chunk for the instructions and the drafts, the new one at most
/// `DRAFT_TOKENS` long.
const COMPACT_RESERVE: usize = 2048;
const DRAFT_TOKENS: usize = 1024;
/// Share of the context the turn in progress may keep verbatim after the summary.
const KEEP_SHARE: f64 = 0.25;

/// Whether a prompt of `history` fills so much of the context that the conversation should be
/// compacted first.
pub fn is_due(engine: &Engine, setting: &Setting, history: &[Message]) -> Result<bool> {
    Ok(engine.fill(setting, history)? >= COMPACT_AT)
}

/// Compact the room's conversation in the middle of a turn, keeping after the summary as much
/// of the turn in progress as [`KEEP_SHARE`] of the context holds. Returns the summary and
/// whether any of the turn was kept.
pub async fn compact_mid_turn(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    setting: &Setting,
    room: &Room,
) -> Result<(String, bool)> {
    let base = engine.fill(setting, &[])?;
    let fits = |kept: &[Message]| Ok(engine.fill(setting, kept)? - base <= KEEP_SHARE);
    let conversation = room.conversation()?;
    let (folded, kept) = cutover(conversation, fits)?;
    let continues = !kept.is_empty();
    let summary = compact(sink, engine, setting, room, folded, kept).await?;
    Ok((summary, continues))
}

/// Split a conversation for its restart into the entries its summary is folded from and the
/// messages that follow the summary verbatim. The turn in progress follows whole if it `fits`;
/// otherwise its request and as many of its newest steps as fit, then its request alone, then
/// nothing, with the rest of the turn folded in.
fn cutover(
    mut entries: Vec<Entry>,
    fits: impl Fn(&[Message]) -> Result<bool>,
) -> Result<(Vec<Entry>, Vec<Message>)> {
    let current: Vec<_> = entries.iter().map(|entry| entry.message.clone()).collect();
    let user = current
        .iter()
        .rposition(|message| matches!(message, Message::User(_)));
    let Some(user) = user else {
        return Ok((entries, vec![]));
    };
    let opens = clipping::opening(&current, user);
    let (fold, kept) = if fits(&current[opens..])? {
        (opens, current[opens..].to_vec())
    } else {
        let request = &current[opens..=user];
        let mut kept = (current.len(), vec![]);
        if fits(request)? {
            kept.1 = request.to_vec();
            // Each earlier step boundary keeps more of the turn.
            let steps = clipping::boundaries(&current).into_iter().rev();
            for step in steps.take_while(|&step| step > user + 1) {
                let candidate: Vec<_> = request.iter().chain(&current[step..]).cloned().collect();
                if !fits(&candidate)? {
                    break;
                }
                kept = (step, candidate);
            }
        }
        kept
    };
    entries.truncate(fold);
    Ok((entries, kept))
}

/// Compact the room's conversation: fold the entries in `folded` into one summary, a chunk
/// at a time, and restart from it, followed by `kept`. Progress goes to the probe as log
/// lines. Returns the summary.
pub async fn compact(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    setting: &Setting,
    room: &Room,
    folded: Vec<Entry>,
    kept: Vec<Message>,
) -> Result<String> {
    // The answer's destination does not apply to the summary.
    let setting = Setting {
        destination: Destination::Terminal,
        ..setting.clone()
    };
    let base = engine.fill(&setting, &[])?;
    let free = engine.capacity as f64 * (1.0 - base) - COMPACT_RESERVE as f64;
    let budget = (free * CHUNK_SHARE * BYTES_PER_TOKEN).max(1024.0) as usize;
    let chunks = transcript::chunks(&folded, budget);
    if chunks.is_empty() {
        bail!("nothing to compact before the turn in progress");
    }
    let parts = chunks.len();
    let mut draft: Option<String> = None;
    for (index, chunk) in chunks.into_iter().enumerate() {
        let part = index + 1;
        let progress = format!("compacting the conversation ({part}/{parts})");
        write_frame(sink, &Frame::Log(progress)).await?;
        let note = Note::Summarize {
            draft: draft.take(),
            chunk,
        };
        // A fresh owner: summarizing prompts share no clipping with any conversation.
        let owner = crate::protocol::fresh_id();
        let prompt = [Message::Note(note)];
        let quiet = None::<&mut tokio::io::Sink>;
        // Decoding is greedy, so asking again would write the same message.
        let step =
            step::generate(quiet, engine, owner, &setting, &prompt, Some(DRAFT_TOKENS)).await?;
        if !step.complete {
            bail!("the message for part {part} of the transcript ran past {DRAFT_TOKENS} tokens");
        }
        if !matches!(&step.calls, Ok(calls) if calls.is_empty()) {
            bail!("the model called tools instead of writing part {part}'s message");
        }
        let message = step.answer.trim();
        if message.is_empty() {
            bail!("the model wrote nothing for part {part} of the transcript");
        }
        draft = Some(message.to_string());
    }
    let summary = draft.unwrap_or_default();
    let restarted: Vec<_> = std::iter::once(Message::Answer(summary.clone()))
        .chain(kept)
        .collect();
    room.restart(&restarted)?;
    Ok(summary)
}

#[cfg(test)]
mod tests;

//! The hub's part of a turn, in the place's room: keep what the probe delivers, compact the
//! conversation when it nears the context's end, generate a step, run the hub's own tools,
//! and keep each step. The probe sends only what is new; the conversation so far lives in the
//! room.
use super::compaction;
use super::dialect::FunctionCall;
use super::inference::Engine;
use super::rooms::Room;
use super::step;
use super::transcript;
use super::wording;
use crate::protocol::{
    Frame, Message, Note, Place, Restarted, Setting, ToolCall, ToolResult, write_frame,
};
use eyre::{Result, bail, eyre};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// Consecutive steps the hub may take by itself before handing the turn back to the probe.
const HUB_STEPS: usize = 16;

/// Advance the turn: keep delivery `id`'s `messages` in the place's room and stream the reply.
/// The messages are kept on arrival, once per delivery, so a retry after a dropped connection
/// never repeats them; each step is kept when it completes.
pub async fn advance(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    place: &Place,
    setting: &Setting,
    id: u64,
    messages: Vec<Message>,
) -> Result<()> {
    let room = Room::open(place)?;
    let _lock = lock(&room).await?;
    let (mut notes, incoming): (Vec<_>, Vec<_>) = messages
        .into_iter()
        .partition(|message| matches!(message, Message::Note(_)));
    let own = |note: &Message| matches!(note, Message::Note(Note::Summarize { .. } | Note::Resume));
    if notes.iter().any(own) {
        bail!("notes to summarize or resume are the hub's own");
    }
    // While commands run, an answer without calls is set aside by the probe, so it is not kept.
    let pending = notes.iter().any(|note| {
        matches!(
            note,
            Message::Note(Note::Pending(_) | Note::AnswerSetAside(_))
        )
    });
    keep_delivery(&room, id, incoming)?;
    // A recalled page may take a tenth of the context, so recalling cannot crowd it out.
    let page = transcript::RECALL_PAGE
        .min((engine.capacity as f64 * 0.1 * compaction::BYTES_PER_TOKEN) as usize);
    let mut compacted = false;
    for _ in 0..HUB_STEPS {
        let mut history = room.history()?;
        history.extend(notes.iter().cloned());
        if !compacted && compaction::is_due(engine, setting, &history)? {
            compacted = true;
            let restarted = compaction::compact_mid_turn(sink, engine, setting, &room).await;
            match restarted {
                // Clipping old messages still lets the turn go on.
                Err(error) => {
                    let line = format!("cannot compact the conversation: {error:#}");
                    write_frame(sink, &Frame::Log(line)).await?;
                }
                Ok((summary, continues)) => {
                    let restarted = Restarted::Summarized(summary);
                    write_frame(sink, &Frame::Restarted(restarted)).await?;
                    // Models tend to end their turn after a message of their own.
                    if !continues {
                        notes.push(Message::Note(Note::Resume));
                    }
                    history = room.history()?;
                    history.extend(notes.iter().cloned());
                }
            }
        }
        // The conversation owns its history, so a restart brings another owner.
        let owner = room.epoch()?;
        let step = step::generate(Some(&mut *sink), engine, owner, setting, &history, None).await?;
        let mut kept = step::prose(step.thinking, step.answer);
        // A call's ID is the number of the entry that first keeps it.
        let first = room.next()?;
        let entry = |kept: &[Message]| first + kept.len() as u64;
        let calls = match step.calls {
            Err(error) => {
                // No call is kept for what could not be parsed, so its result has its own ID.
                let content = wording::unparsed(&error);
                kept.push(result(entry(&kept), "tool_call_parse_error", content));
                room.append(&kept)?;
                continue;
            }
            Ok(calls) => calls,
        };
        if calls.is_empty() && pending {
            return write_frame(sink, &Frame::EndOfReply).await;
        }
        // The hub answers its own calls, and refuses one that would only feed a loop.
        let refused = |call: &FunctionCall| repeats(&history, call);
        let (own, probe): (Vec<_>, Vec<_>) = calls
            .into_iter()
            .partition(|call| call.name == transcript::RECALL || refused(call));
        // The hub's calls and their results first, then the calls the probe answers.
        for call in &own {
            let (line, content) = match refused(call) {
                true => (
                    format!("skipped a repeated {} call", call.name),
                    wording::repeated(),
                ),
                false => (
                    format!("recall {}", call.arguments),
                    transcript::recall(&room, &call.arguments, page).to_string(),
                ),
            };
            write_frame(sink, &Frame::Log(line)).await?;
            let id = entry(&kept);
            kept.push(Message::Call(call.with_id(id)));
            kept.push(result(id, &call.name, content));
        }
        let mut sent = Vec::with_capacity(probe.len());
        for call in probe {
            let call = call.with_id(entry(&kept));
            kept.push(Message::Call(call.clone()));
            sent.push(call);
        }
        room.append(&kept)?;
        if !sent.is_empty() || own.is_empty() {
            for call in sent {
                write_frame(sink, &Frame::Call(call)).await?;
            }
            return write_frame(sink, &Frame::EndOfReply).await;
        }
    }
    Err(eyre!(
        "the model took {HUB_STEPS} steps without answering or handing work to the probe"
    ))
}

/// Restart the conversation in the place's room, from a summary of it or from nothing.
pub async fn restart(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    place: &Place,
    setting: &Setting,
    summarize: bool,
) -> Result<()> {
    let room = Room::open(place)?;
    let _lock = lock(&room).await?;
    let conversation = room.conversation()?;
    let restarted = if conversation.is_empty() {
        Restarted::Empty
    } else if summarize {
        let summary =
            compaction::compact(sink, engine, setting, &room, conversation, vec![]).await?;
        Restarted::Summarized(summary)
    } else {
        room.restart(&[])?;
        Restarted::Cleared
    };
    write_frame(sink, &Frame::Restarted(restarted)).await?;
    write_frame(sink, &Frame::EndOfReply).await
}

/// Hold the room for this hub, waiting for another hub to finish with it.
async fn lock(room: &Room) -> Result<std::fs::File> {
    let room = room.clone();
    tokio::task::spawn_blocking(move || room.lock()).await?
}

/// Keep delivery `id`'s messages unless a retry already did, answering first any calls the
/// conversation ends with that the delivery does not answer.
fn keep_delivery(room: &Room, id: u64, incoming: Vec<Message>) -> Result<()> {
    let retried = room.last_delivery() == Some(id);
    let answers = !retried && matches!(incoming.first(), Some(Message::Result(_)));
    let mut arriving = match answers {
        true => vec![],
        false => {
            let history = room.history()?;
            unanswered(&history)
        }
    };
    if !retried {
        arriving.extend(incoming);
    }
    room.append(&arriving)?;
    room.set_last_delivery(id)
}

fn result(id: u64, name: &str, content: String) -> Message {
    Message::Result(ToolResult {
        id,
        name: name.into(),
        content,
    })
}

/// Whether `call` would be the third in a row of the same call, the two before it having
/// come to the same outcome: greedy decoding can repeat such a step forever.
fn repeats(history: &[Message], call: &FunctionCall) -> bool {
    let arguments = call.arguments.to_string();
    let mut made = std::collections::HashMap::new();
    let mut answered = Vec::new();
    for message in history {
        match message {
            Message::Call(call) => {
                made.insert(call.id, call);
            }
            Message::Result(result) => {
                answered.extend(made.get(&result.id).map(|&call| (call, result)))
            }
            _ => {}
        }
    }
    let [.., (first, earlier), (second, latest)] = answered.as_slice() else {
        return false;
    };
    let same = |made: &&ToolCall| made.name == call.name && made.arguments == arguments;
    let outcome = |result: &ToolResult| crate::tools::outcome(&result.content);
    same(first) && same(second) && outcome(earlier).is_some_and(|it| Some(it) == outcome(latest))
}

/// A result for each call `kept` ends with that never got one, as when a turn was cut short
/// after the hub sent the calls; each says nothing was recorded.
fn unanswered(kept: &[Message]) -> Vec<Message> {
    let open = kept
        .iter()
        .rev()
        .take_while(|message| matches!(message, Message::Call(_)))
        .count();
    kept[kept.len() - open..]
        .iter()
        .map(|message| match message {
            Message::Call(call) => result(call.id, &call.name, wording::unrecorded()),
            _ => unreachable!("only calls end the conversation here"),
        })
        .collect()
}

#[cfg(test)]
mod tests;

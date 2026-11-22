//! One request's work on a place's conversation: keep what the request brings, condense the
//! conversation when it nears the context's end, generate, run the hub's own tools, and keep
//! each step. The probe sends only what is new; the conversation itself lives with the hub.
use super::chats::Chat;
use super::dialect::{Delta, ToolCall, prompting};
use super::inference::{self, Engine};
use super::transcript;
use crate::protocol::{
    Call, Frame, Message, Note, Place, Setting, ToolResult, write_frame_to_stream,
};
use eyre::{Result, bail, eyre};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// Share of the context a prompt may fill before the conversation is condensed and restarted.
const CONDENSE_AT: f64 = 0.75;
/// Share of what the instructions leave free that each condensing chunk may take, and the bytes
/// a token is counted as; code tokens average more, so chunks come out smaller than allowed.
const CHUNK_SHARE: f64 = 0.5;
const BYTES_PER_TOKEN: f64 = 2.5;
/// Tokens kept free around each chunk for the instructions and the drafts, the new one at most
/// `DRAFT_TOKENS` long.
const CONDENSE_ROOM: usize = 2048;
const DRAFT_TOKENS: usize = 1024;
/// Share of the context the latest request may take and still follow the summary verbatim.
const KEEP_SHARE: f64 = 0.25;
/// Consecutive steps the hub may take by itself before handing the turn back to the probe.
const HUB_STEPS: usize = 16;

/// What the model produced in one response.
struct Step {
    thinking: String,
    answer: String,
    calls: Result<Vec<ToolCall>>,
}

/// Serve one request: keep `messages` in the place's conversation and stream the reply.
/// The request's messages are kept on arrival, once per request `id`, so a retry after a
/// dropped connection never repeats them; each step is kept when it completes.
pub async fn serve(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    place: &Place,
    setting: &Setting,
    id: u64,
    messages: Vec<Message>,
) -> Result<()> {
    let chat = Chat::open(place)?;
    let _lock = lock(&chat).await?;
    let (mut notes, incoming): (Vec<_>, Vec<_>) = messages
        .into_iter()
        .partition(|message| matches!(message, Message::Note(_)));
    let own = |note: &Message| matches!(note, Message::Note(Note::Restart { .. } | Note::Resume));
    if notes.iter().any(own) {
        bail!("restart notes are the hub's own");
    }
    // While commands run, an answer without calls is set aside by the probe, so it is not kept.
    let pending = notes.iter().any(|note| {
        matches!(
            note,
            Message::Note(Note::Pending(_) | Note::AnswerSetAside(_))
        )
    });
    let mut request = keep_arrivals(&chat, id, incoming)?;
    // A recalled page may take a tenth of the context, so recalling cannot crowd it out.
    let page =
        transcript::RECALLED.min((engine.capacity() as f64 * 0.1 * BYTES_PER_TOKEN) as usize);
    let mut condensed = false;
    for _ in 0..HUB_STEPS {
        let mut history = chat.messages()?;
        history.extend(notes.iter().cloned());
        if !condensed && engine.fill(setting, &history)? >= CONDENSE_AT {
            condensed = true;
            // The latest request stays verbatim after the summary unless it alone would crowd it.
            let entries = chat.entries()?.into_iter();
            let latest: Vec<_> = entries
                .filter(|entry| !entry.backup && entry.number >= request)
                .map(|entry| entry.message)
                .collect();
            let crowded = engine.fill(setting, &latest)? - engine.fill(setting, &[])? > KEEP_SHARE;
            let keep = (!crowded).then_some(request);
            match condense(sink, engine, setting, &chat, keep).await {
                Ok(summary) => {
                    write_frame_to_stream(sink, &Frame::Restarted(summary)).await?;
                    // Models tend to end their turn after a message of their own.
                    notes.push(Message::Note(Note::Resume));
                    request = chat.generation()?;
                    history = chat.messages()?;
                    history.extend(notes.iter().cloned());
                }
                // Clipping old messages still lets the conversation go on.
                Err(error) => {
                    let line = format!("cannot condense the conversation: {error:#}");
                    write_frame_to_stream(sink, &Frame::Log(line)).await?;
                }
            }
        }
        let key = conversation(place, chat.generation()?);
        let step = generate(Some(&mut *sink), engine, key, setting, &history, None).await?;
        let calls = match step.calls {
            Ok(calls) => calls,
            Err(error) => {
                let mut kept = opening(step.thinking, step.answer);
                kept.push(result("tool_call_parse_error", prompting::unparsed(&error)));
                chat.append(&kept)?;
                continue;
            }
        };
        if calls.is_empty() && pending {
            return write_frame_to_stream(sink, &Frame::Stop).await;
        }
        let (own, probe): (Vec<_>, Vec<_>) = calls
            .into_iter()
            .partition(|call| call.name == transcript::RECALL);
        // The hub's calls and their results first, then the calls the probe answers.
        let mut kept = opening(step.thinking, step.answer);
        for call in &own {
            write_frame_to_stream(sink, &Frame::Log(format!("recall {}", call.arguments))).await?;
            kept.push(stored(call));
            let content = transcript::recall(&chat, &call.arguments, page).to_string();
            kept.push(result(&call.name, content));
        }
        kept.extend(probe.iter().map(stored));
        chat.append(&kept)?;
        if !probe.is_empty() || own.is_empty() {
            for call in probe {
                let arguments_json = serde_json::to_string(&call.arguments)?;
                let name = call.name;
                write_frame_to_stream(
                    sink,
                    &Frame::ToolCall {
                        name,
                        arguments_json,
                    },
                )
                .await?;
            }
            return write_frame_to_stream(sink, &Frame::Stop).await;
        }
    }
    Err(eyre!(
        "the model took {HUB_STEPS} steps without answering or handing work to the probe"
    ))
}

/// Restart the place's conversation, from a summary of it or from nothing.
pub async fn restart(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    place: &Place,
    setting: &Setting,
    summarize: bool,
) -> Result<()> {
    let chat = Chat::open(place)?;
    let _lock = lock(&chat).await?;
    if chat.messages()?.is_empty() {
        write_frame_to_stream(sink, &Frame::Log("the conversation is empty".into())).await?;
    } else if summarize {
        let summary = condense(sink, engine, setting, &chat, None).await?;
        write_frame_to_stream(sink, &Frame::Restarted(summary)).await?;
    } else {
        chat.restart(&[])?;
        write_frame_to_stream(sink, &Frame::Log("started a new conversation".into())).await?;
    }
    write_frame_to_stream(sink, &Frame::Stop).await
}

/// Hold the conversation for this hub, waiting for another hub to finish with it.
async fn lock(chat: &Chat) -> Result<std::fs::File> {
    let chat = chat.clone();
    tokio::task::spawn_blocking(move || chat.lock()).await?
}

/// Keep request `id`'s messages unless a retry already did, answering first any calls the
/// conversation ends with that the request does not answer; return where the request starts.
fn keep_arrivals(chat: &Chat, id: u64, incoming: Vec<Message>) -> Result<u64> {
    let applied = chat.applied().filter(|&(applied, _)| applied == id);
    let answers = applied.is_none() && matches!(incoming.first(), Some(Message::Result(_)));
    let mut arriving = Vec::new();
    if !answers {
        arriving = unanswered(&chat.messages()?);
    }
    let start = match applied {
        Some((_, start)) => start,
        None => chat.next()? + arriving.len() as u64,
    };
    if applied.is_none() {
        arriving.extend(incoming);
    }
    chat.append(&arriving)?;
    chat.set_applied(id, start)?;
    Ok(start)
}

/// Who a prompt belongs to, keeping clipping choices with their own conversation; a restart
/// makes it another.
fn conversation(place: &Place, generation: u64) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    (place, generation).hash(&mut hasher);
    hasher.finish()
}

/// A step's reasoning and answer; a response of calls alone still opens its own step.
fn opening(thinking: String, answer: String) -> Vec<Message> {
    let mut kept = Vec::new();
    if !thinking.is_empty() || answer.is_empty() {
        kept.push(Message::Thinking(thinking));
    }
    if !answer.is_empty() {
        kept.push(Message::Answer(answer));
    }
    kept
}

fn stored(call: &ToolCall) -> Message {
    Message::Call(Call {
        name: call.name.clone(),
        arguments: call.arguments.to_string(),
    })
}

fn result(name: &str, content: String) -> Message {
    Message::Result(ToolResult {
        name: name.into(),
        content,
    })
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
            Message::Call(call) => result(&call.name, prompting::unrecorded()),
            _ => unreachable!("only calls end the conversation here"),
        })
        .collect()
}

/// Fold the entries since the last restart into one message, a chunk at a time, and restart
/// the conversation from it. Entries from `keep` on are not folded: they follow the message
/// as they were, so a request is never condensed before the model has read it.
/// Progress goes to the probe as log lines.
async fn condense(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    setting: &Setting,
    chat: &Chat,
    keep: Option<u64>,
) -> Result<String> {
    // The answer's destination does not apply to the summary.
    let setting = Setting {
        redirection: None,
        ..setting.clone()
    };
    let keep = keep.unwrap_or(u64::MAX);
    let (folded, kept): (Vec<_>, Vec<_>) = chat
        .entries()?
        .into_iter()
        .partition(|entry| entry.backup || entry.number < keep);
    let head = engine.fill(&setting, &[])?;
    let free = engine.capacity() as f64 * (1.0 - head) - CONDENSE_ROOM as f64;
    let budget = (free * CHUNK_SHARE * BYTES_PER_TOKEN).max(1024.0) as usize;
    let chunks = transcript::chunks(&folded, budget);
    if chunks.is_empty() {
        bail!("nothing before the latest request to condense");
    }
    let mut draft: Option<String> = None;
    for (index, chunk) in chunks.iter().enumerate() {
        let progress = format!(
            "condensing the conversation ({}/{})",
            index + 1,
            chunks.len()
        );
        write_frame_to_stream(sink, &Frame::Log(progress)).await?;
        let note = Note::Restart {
            draft: draft.clone(),
            chunk: chunk.clone(),
        };
        // A fresh key: condensing prompts share no clipping with any conversation.
        let key = engine.connect();
        let prompt = [Message::Note(note)];
        let mut answer = String::new();
        for _ in 0..2 {
            let quiet = None::<&mut tokio::io::Sink>;
            let step = generate(quiet, engine, key, &setting, &prompt, Some(DRAFT_TOKENS)).await?;
            answer = prompting::unwrap_message(&step.answer).to_string();
            if !answer.is_empty() {
                break;
            }
        }
        if answer.is_empty() {
            bail!(
                "the model wrote nothing for part {} of the transcript",
                index + 1
            );
        }
        draft = Some(answer);
    }
    let summary = draft.unwrap_or_default();
    let restarted = std::iter::once(Message::Answer(summary.clone()));
    let restarted: Vec<_> = restarted
        .chain(kept.into_iter().map(|entry| entry.message))
        .collect();
    chat.restart(&restarted)?;
    Ok(summary)
}

/// Generate one response to `history`, streaming its text to `sink` when given.
async fn generate(
    mut sink: Option<&mut (impl AsyncWriteExt + Unpin)>,
    engine: &Arc<Engine>,
    key: u64,
    setting: &Setting,
    history: &[Message],
    max_tokens: Option<usize>,
) -> Result<Step> {
    let mut parser = engine.output_parser()?;
    let (sender, mut generated) = tokio::sync::mpsc::unbounded_channel();
    let worker = engine.clone();
    let (setting, history) = (setting.clone(), history.to_vec());
    let task = tokio::spawn(async move {
        inference::generate(&worker, key, &setting, &history, sender, max_tokens).await
    });
    let mut step = Step {
        thinking: String::new(),
        answer: String::new(),
        calls: Ok(vec![]),
    };
    while let Some(inference::Generated::Token(token)) = generated.recv().await {
        for delta in parser.push_token(token)? {
            let frame = match delta {
                Delta::Answer(text) => {
                    step.answer.push_str(&text);
                    Frame::Answer(text)
                }
                Delta::Thinking(text) => {
                    step.thinking.push_str(&text);
                    Frame::Thinking(text)
                }
            };
            if let Some(sink) = sink.as_mut() {
                write_frame_to_stream(sink, &frame).await?;
            }
        }
    }
    task.await??;
    step.calls = parser.finish();
    Ok(step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn call(name: &str) -> Message {
        Message::Call(Call {
            name: name.into(),
            arguments: "{}".into(),
        })
    }

    #[test]
    fn calls_left_without_results_get_an_honest_one() {
        let kept = vec![
            Message::User("go".into()),
            Message::Thinking(String::new()),
            call("apply_patch"),
            call("run_command"),
        ];
        let answers = unanswered(&kept);
        let names: Vec<_> = answers
            .iter()
            .map(|message| match message {
                Message::Result(result) => result.name.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(names, ["apply_patch", "run_command"]);
        let Message::Result(first) = &answers[0] else {
            unreachable!()
        };
        assert!(first.content.contains("may or may not have run"));
        assert!(unanswered(&kept[..2]).is_empty());
    }

    #[test]
    fn a_request_is_kept_once_however_often_it_is_retried() -> Result<()> {
        let dir = TempDir::new()?;
        let chat = Chat::at(dir.0.join("chat"))?;
        chat.append(&[
            Message::User("first".into()),
            Message::Thinking(String::new()),
            call("read_file"),
        ])?;
        // Results answer the dangling call and start the request.
        let results = vec![result("read_file", "x".into())];
        assert_eq!(keep_arrivals(&chat, 7, results.clone())?, 4);
        assert_eq!(keep_arrivals(&chat, 7, results)?, 4);
        assert_eq!(chat.messages()?.len(), 4);
        // A new request after a step whose calls never got results answers them first.
        chat.append(&[Message::Thinking(String::new()), call("run_command")])?;
        let request = vec![Message::User("next".into())];
        assert_eq!(keep_arrivals(&chat, 8, request)?, 8);
        let messages = chat.messages()?;
        assert!(matches!(&messages[6], Message::Result(result) if result.name == "run_command"));
        assert_eq!(messages[7], Message::User("next".into()));
        Ok(())
    }

    #[test]
    fn conversations_keep_their_key_until_they_restart() {
        let place = Place {
            host: "h".into(),
            cwd: "/".into(),
            session: "1@2".into(),
        };
        assert_eq!(conversation(&place, 3), conversation(&place, 3));
        assert_ne!(conversation(&place, 3), conversation(&place, 9));
    }
}

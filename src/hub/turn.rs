//! The hub's part of a turn, in the place's room: keep what the probe delivers, compact the
//! room when it nears the context's end, generate, run the hub's own tools, and keep each
//! step. The probe sends only what is new; everything said so far lives in the room.
use super::dialect::{Delta, FunctionCall, prompting};
use super::inference::{self, Engine, clipping};
use super::rooms::{Entry, Room};
use super::transcript;
use crate::protocol::{Frame, Message, Note, Place, Setting, ToolCall, ToolResult, write_frame};
use eyre::{Result, bail, eyre};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// Share of the context a prompt may fill before the room is compacted: restarted from a
/// summary of it.
const COMPACT_AT: f64 = 0.75;
/// Share of what the instructions leave free that each chunk to summarize may take, and the
/// bytes a token is counted as; code tokens average more, so chunks come out smaller than
/// allowed.
const CHUNK_SHARE: f64 = 0.5;
const BYTES_PER_TOKEN: f64 = 2.5;
/// Tokens kept free around each chunk for the instructions and the drafts, the new one at most
/// `DRAFT_TOKENS` long.
const COMPACT_RESERVE: usize = 2048;
const DRAFT_TOKENS: usize = 1024;
/// Share of the context the turn in progress may keep verbatim after the summary.
const KEEP_SHARE: f64 = 0.25;
/// Consecutive steps the hub may take by itself before handing the turn back to the probe.
const HUB_STEPS: usize = 16;

/// What the model produced in one response.
struct Step {
    thinking: String,
    answer: String,
    calls: Result<Vec<FunctionCall>>,
    /// The response ended by itself rather than at its token limit.
    complete: bool,
}

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
    let page =
        transcript::RECALL_PAGE.min((engine.capacity() as f64 * 0.1 * BYTES_PER_TOKEN) as usize);
    let mut compacted = false;
    for _ in 0..HUB_STEPS {
        let mut history = room.messages()?;
        history.extend(notes.iter().cloned());
        if !compacted && engine.fill(setting, &history)? >= COMPACT_AT {
            compacted = true;
            let base = engine.fill(setting, &[])?;
            let fits = |kept: &[Message]| Ok(engine.fill(setting, kept)? - base <= KEEP_SHARE);
            let entries = room.entries()?;
            let (folded, kept) = cutover(entries, fits)?;
            // Models tend to end their turn after a message of their own.
            let resume = kept.is_empty();
            let restarted = compact(sink, engine, setting, &room, folded, kept).await;
            match restarted {
                // Clipping old messages still lets the turn go on.
                Err(error) => {
                    let line = format!("cannot compact the conversation: {error:#}");
                    write_frame(sink, &Frame::Log(line)).await?;
                }
                Ok(summary) => {
                    write_frame(sink, &Frame::Restarted(summary)).await?;
                    if resume {
                        notes.push(Message::Note(Note::Resume));
                    }
                    history = room.messages()?;
                    history.extend(notes.iter().cloned());
                }
            }
        }
        let epoch = room.epoch()?;
        let owner = owner(place, epoch);
        let step = generate(Some(&mut *sink), engine, owner, setting, &history, None).await?;
        let mut kept = prose(step.thinking, step.answer);
        // A call's ID is the number of the entry that first keeps it.
        let first = room.next()?;
        let entry = |kept: &[Message]| first + kept.len() as u64;
        let calls = match step.calls {
            Err(error) => {
                // No call is kept for what could not be parsed, so its result has its own ID.
                let content = prompting::unparsed(&error);
                kept.push(result(entry(&kept), "tool_call_parse_error", content));
                room.append(&kept)?;
                continue;
            }
            Ok(calls) => calls,
        };
        if calls.is_empty() && pending {
            return write_frame(sink, &Frame::Stop).await;
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
                    prompting::repeated(),
                ),
                false => (
                    format!("recall {}", call.arguments),
                    transcript::recall(&room, &call.arguments, page).to_string(),
                ),
            };
            write_frame(sink, &Frame::Log(line)).await?;
            let id = entry(&kept);
            kept.push(stored(call, id));
            kept.push(result(id, &call.name, content));
        }
        let mut sent = Vec::with_capacity(probe.len());
        for call in probe {
            let id = entry(&kept);
            kept.push(stored(&call, id));
            sent.push((id, call));
        }
        room.append(&kept)?;
        if !sent.is_empty() || own.is_empty() {
            for (id, call) in sent {
                let frame = Frame::Call(ToolCall {
                    id,
                    name: call.name,
                    arguments: call.arguments.to_string(),
                });
                write_frame(sink, &frame).await?;
            }
            return write_frame(sink, &Frame::Stop).await;
        }
    }
    Err(eyre!(
        "the model took {HUB_STEPS} steps without answering or handing work to the probe"
    ))
}

/// Restart the place's room, from a summary of it or from nothing.
pub async fn restart(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    place: &Place,
    setting: &Setting,
    summarize: bool,
) -> Result<()> {
    let room = Room::open(place)?;
    let _lock = lock(&room).await?;
    let messages = room.messages()?;
    if messages.is_empty() {
        write_frame(sink, &Frame::Log("the conversation is empty".into())).await?;
    } else if summarize {
        let entries = room.entries()?;
        let summary = compact(sink, engine, setting, &room, entries, vec![]).await?;
        write_frame(sink, &Frame::Restarted(summary)).await?;
    } else {
        room.restart(&[])?;
        write_frame(sink, &Frame::Log("started a new conversation".into())).await?;
    }
    write_frame(sink, &Frame::Stop).await
}

/// Hold the room for this hub, waiting for another hub to finish with it.
async fn lock(room: &Room) -> Result<std::fs::File> {
    let room = room.clone();
    tokio::task::spawn_blocking(move || room.lock()).await?
}

/// Keep delivery `id`'s messages unless a retry already did, answering first any calls the
/// room ends with that the delivery does not answer.
fn keep_delivery(room: &Room, id: u64, incoming: Vec<Message>) -> Result<()> {
    let retried = room.last_delivery() == Some(id);
    let answers = !retried && matches!(incoming.first(), Some(Message::Result(_)));
    let mut arriving = match answers {
        true => vec![],
        false => {
            let kept = room.messages()?;
            unanswered(&kept)
        }
    };
    if !retried {
        arriving.extend(incoming);
    }
    room.append(&arriving)?;
    room.set_last_delivery(id)
}

/// Split a room's entries for its restart into those its summary is folded from and the
/// messages that follow the summary verbatim. The turn in progress follows whole if it `fits`;
/// otherwise its request and as many of its newest steps as fit, then its request alone, then
/// nothing, with the rest of the turn folded in.
fn cutover(
    mut entries: Vec<Entry>,
    fits: impl Fn(&[Message]) -> Result<bool>,
) -> Result<(Vec<Entry>, Vec<Message>)> {
    let since = entries.iter().take_while(|entry| entry.old).count();
    let current: Vec<_> = entries[since..]
        .iter()
        .map(|entry| entry.message.clone())
        .collect();
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
    entries.truncate(since + fold);
    Ok((entries, kept))
}

/// Whose history a prompt renders: the place's room in its current epoch. Clipping choices
/// stay with their owner, and a restart makes the room another.
fn owner(place: &Place, epoch: u64) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    (place, epoch).hash(&mut hasher);
    hasher.finish()
}

/// A step's reasoning and answer; a response of calls alone still opens its own step.
fn prose(thinking: String, answer: String) -> Vec<Message> {
    let mut kept = Vec::new();
    if !thinking.is_empty() || answer.is_empty() {
        kept.push(Message::Thinking(thinking));
    }
    if !answer.is_empty() {
        kept.push(Message::Answer(answer));
    }
    kept
}

fn stored(call: &FunctionCall, id: u64) -> Message {
    Message::Call(ToolCall {
        id,
        name: call.name.clone(),
        arguments: call.arguments.to_string(),
    })
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
            Message::Call(call) => result(call.id, &call.name, prompting::unrecorded()),
            _ => unreachable!("only calls end the room here"),
        })
        .collect()
}

/// Compact the room: fold `folded`'s entries since the last restart into one summary, a chunk
/// at a time, and restart the room from it, followed by `kept`. Progress goes to the probe as
/// log lines.
async fn compact(
    sink: &mut (impl AsyncWriteExt + Unpin),
    engine: &Arc<Engine>,
    setting: &Setting,
    room: &Room,
    folded: Vec<Entry>,
    kept: Vec<Message>,
) -> Result<String> {
    // The answer's destination does not apply to the summary.
    let setting = Setting {
        redirection: None,
        ..setting.clone()
    };
    let head = engine.fill(&setting, &[])?;
    let free = engine.capacity() as f64 * (1.0 - head) - COMPACT_RESERVE as f64;
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
        // A fresh owner: summarizing prompts share no clipping with any room.
        let owner = engine.new_owner();
        let prompt = [Message::Note(note)];
        let quiet = None::<&mut tokio::io::Sink>;
        // Decoding is greedy, so asking again would write the same message.
        let step = generate(quiet, engine, owner, &setting, &prompt, Some(DRAFT_TOKENS)).await?;
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

/// Generate one response to `history`, streaming its text to `sink` when given.
async fn generate(
    mut sink: Option<&mut (impl AsyncWriteExt + Unpin)>,
    engine: &Arc<Engine>,
    owner: u64,
    setting: &Setting,
    history: &[Message],
    max_tokens: Option<usize>,
) -> Result<Step> {
    let mut parser = engine.output_parser()?;
    let (sender, mut generated) = tokio::sync::mpsc::unbounded_channel();
    let (engine, setting, history) = (engine.clone(), setting.clone(), history.to_vec());
    let task = tokio::spawn(async move {
        inference::generate(&engine, owner, &setting, &history, sender, max_tokens).await
    });
    let (mut thinking, mut answer) = (String::new(), String::new());
    while let Some(token) = generated.recv().await {
        for delta in parser.push_token(token)? {
            let frame = match delta {
                Delta::Answer(text) => {
                    answer.push_str(&text);
                    Frame::Answer(text)
                }
                Delta::Thinking(text) => {
                    thinking.push_str(&text);
                    Frame::Thinking(text)
                }
            };
            if let Some(sink) = sink.as_mut() {
                write_frame(sink, &frame).await?;
            }
        }
    }
    let generation = task.await??;
    let calls = parser.finish();
    let complete = generation.is_some_and(|generation| generation.complete);
    Ok(Step {
        thinking,
        answer,
        calls,
        complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn some_call(id: u64, name: &str) -> Message {
        Message::Call(ToolCall {
            id,
            name: name.into(),
            arguments: "{}".into(),
        })
    }

    #[test]
    fn calls_left_without_results_get_an_honest_one() {
        let kept = vec![
            Message::User("go".into()),
            Message::Thinking(String::new()),
            some_call(3, "apply_patch"),
            some_call(4, "run_command"),
        ];
        let answers = unanswered(&kept);
        let answered: Vec<_> = answers
            .iter()
            .map(|message| match message {
                Message::Result(result) => (result.id, result.name.as_str()),
                _ => (0, ""),
            })
            .collect();
        assert_eq!(answered, [(3, "apply_patch"), (4, "run_command")]);
        let Message::Result(first) = &answers[0] else {
            unreachable!()
        };
        assert!(first.content.contains("may or may not have run"));
        assert!(unanswered(&kept[..2]).is_empty());
    }

    #[test]
    fn a_delivery_is_kept_once_however_often_it_is_retried() -> Result<()> {
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        room.append(&[
            Message::User("first".into()),
            Message::Thinking(String::new()),
            some_call(3, "read_file"),
        ])?;
        // Results answer the dangling call.
        let results = vec![result(3, "read_file", "x".into())];
        keep_delivery(&room, 7, results.clone())?;
        keep_delivery(&room, 7, results)?;
        assert_eq!(room.messages()?.len(), 4);
        // A new request after a step whose calls never got results answers them first.
        room.append(&[
            Message::Thinking(String::new()),
            some_call(6, "run_command"),
        ])?;
        keep_delivery(&room, 8, vec![Message::User("next".into())])?;
        let messages = room.messages()?;
        let answer = Message::Result(ToolResult {
            id: 6,
            name: "run_command".into(),
            content: prompting::unrecorded(),
        });
        assert_eq!(messages[6], answer);
        assert_eq!(messages[7], Message::User("next".into()));
        Ok(())
    }

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
            room.append(&[
                Message::Thinking(content.clone()),
                some_call(id, "read_file"),
            ])?;
            room.append(&[result(id, "read_file", content)])?;
        }
        // Entries 1 and 2 are set aside, 3 is the restart message, and the turn is 4 to 14.
        let split = |most: usize| {
            let fits = |kept: &[Message]| Ok(kept.len() <= most);
            let (folded, kept) = cutover(room.entries()?, fits)?;
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

    #[test]
    fn a_third_identical_call_with_an_unchanged_result_is_refused() {
        let call = |name: &str, arguments| FunctionCall {
            name: name.into(),
            arguments,
        };
        let grep = call(
            "run_command",
            serde_json::json!({"argv": ["grep", "x", "f"]}),
        );
        let ids = std::cell::Cell::new(0);
        let step = |made: &FunctionCall, content: &str| {
            let thinking = Message::Thinking(String::new());
            ids.set(ids.get() + 1);
            let id = ids.get();
            vec![
                thinking,
                stored(made, id),
                result(id, &made.name, content.into()),
            ]
        };
        let twice = [step(&grep, "none"), step(&grep, "none")].concat();
        assert!(repeats(&twice, &grep));
        // A changed result, another call between, or other arguments let it run.
        assert!(!repeats(
            &[step(&grep, "none"), step(&grep, "one")].concat(),
            &grep
        ));
        let edit = call("apply_patch", serde_json::json!({"patch": "p"}));
        let between = [step(&grep, "none"), step(&edit, "ok"), step(&grep, "none")].concat();
        assert!(!repeats(&between, &grep));
        assert!(!repeats(&step(&grep, "none"), &grep));
        let other = call(
            "run_command",
            serde_json::json!({"argv": ["grep", "y", "f"]}),
        );
        assert!(!repeats(&twice, &other));
        // A command's handle, process, and running time differ every time; a running one is
        // being waited on.
        let ran = |pid: u32, status: &str| {
            let fields = serde_json::json!({"command_id": pid, "pid": pid, "status": status});
            fields.to_string()
        };
        let finished = [
            step(&grep, &ran(1, "finished")),
            step(&grep, &ran(2, "finished")),
        ];
        assert!(repeats(&finished.concat(), &grep));
        let running = [
            step(&grep, &ran(1, "running")),
            step(&grep, &ran(2, "running")),
        ];
        assert!(!repeats(&running.concat(), &grep));
    }

    #[test]
    fn a_room_keeps_its_owner_until_it_restarts() {
        let place = Place {
            machine: "ssh-ed25519 AAAA".into(),
            cwd: "/".into(),
        };
        assert_eq!(owner(&place, 3), owner(&place, 3));
        assert_ne!(owner(&place, 3), owner(&place, 9));
    }
}

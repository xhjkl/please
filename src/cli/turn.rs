use eyre::{Result, eyre};
use futures_util::future::join_all;
use serde_json::{Value, json};
use std::io;
use std::time::Duration;
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};

use crate::display::{Approval, Display};
use crate::protocol::{
    Frame, Message, Note, Place, ProtocolError, Setting, ToolCall, ToolResult, read_frame,
    write_frame,
};
use crate::tools::{
    Jobs, ToolContext, ToolKind, Tools, all_tools, kind_of, summarize_patch_for_preview,
};

use super::commands::Command;
use super::connect::connect_to_hub;

/// A turn the user cancelled with Ctrl-C.
#[derive(Debug, thiserror::Error)]
#[error("turn cancelled")]
pub struct TurnCancelled;

/// A call the model made, with its arguments parsed into an object.
struct ParsedCall {
    id: u64,
    name: String,
    arguments: Value,
}

/// Where and how a turn happens: `place` names the room the hub keeps it in.
#[derive(Clone, Copy)]
pub struct Scene<'a> {
    pub place: &'a Place,
    pub setting: &'a Setting,
}

/// What the probe adds to the conversation next: a request, or a step's tool results, under an
/// ID that stays the same when a reconnect delivers them again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivery {
    pub id: u64,
    pub messages: Vec<Message>,
}

impl Delivery {
    /// Carry `messages` under a fresh ID.
    pub fn new(messages: Vec<Message>) -> Self {
        use std::hash::BuildHasher;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = std::hash::RandomState::new().hash_one((count, std::process::id()));
        Self { id, messages }
    }
}

/// Take the turn's steps on a connection, sending the hub only what is new: first the turn's
/// request, then each step's tool results. `delivery` stays what the hub has yet to answer
/// until a response completes, so a retry after a reconnect resends it, under the same ID.
/// A complete model response resets the reconnect budget; a partial response does not.
async fn attempt_turn(
    scene: Scene<'_>,
    delivery: &mut Delivery,
    jobs: &Jobs,
    stream: &mut UnixStream,
    reconnect_attempts: &mut usize,
    display: &Display,
) -> Result<String> {
    let tools = all_tools();

    loop {
        let command_ids = jobs.pending_ids();
        let mut messages = delivery.messages.clone();
        let set_aside = messages
            .iter()
            .any(|message| matches!(message, Message::Note(Note::AnswerSetAside(_))));
        if !command_ids.is_empty() && !set_aside {
            messages.push(Message::Note(Note::Pending(command_ids.clone())));
        }
        let mut spinner = Some(display.start_spinning());
        let advance = Frame::Advance {
            place: scene.place.clone(),
            setting: scene.setting.clone(),
            id: delivery.id,
            messages,
        };
        write_frame(stream, &advance).await?;

        let mut store = Vec::with_capacity(4096);
        let mut thinking = None;
        let mut answer = String::new();
        let mut calls = Vec::new();
        let mut malformed = Vec::new();
        // Call IDs in the order the model made the calls, which their results keep.
        let mut order = Vec::new();

        loop {
            let frame = read_frame(stream, &mut store, None, None).await?;
            // Stop the spinner before its line clear can erase streamed output.
            if let Some(spinner) = spinner.take() {
                spinner.stop().await;
            }
            match frame {
                Frame::Log(line) => display.show_log(&line),
                // Shown apart from the answer, which may be going to a file.
                Frame::Restarted(summary) => {
                    thinking = None;
                    eprintln!("[the conversation restarted from this message]\n{summary}\n");
                }
                Frame::Answer(delta) => {
                    if command_ids.is_empty() {
                        thinking = None;
                        display.show_answer(&delta);
                    }
                    answer.push_str(&delta);
                }
                Frame::Thinking(delta) => {
                    if thinking.is_none() {
                        thinking = display.thinking_line();
                    }
                    if let Some(line) = &thinking {
                        line.show(&delta);
                    }
                }
                Frame::Call(ToolCall {
                    id,
                    name,
                    arguments,
                }) => {
                    order.push(id);
                    // Only an object reaches a tool: approval reads the same fields the tool does.
                    match serde_json::from_str::<Value>(&arguments) {
                        Ok(arguments) if arguments.is_object() => calls.push(ParsedCall {
                            id,
                            name,
                            arguments,
                        }),
                        _ => malformed.push((id, name)),
                    }
                }
                Frame::Stop => {
                    *reconnect_attempts = 0;
                    break;
                }
                Frame::Error(error) => return Err(eyre!(error)),
                _ => return Err(eyre!("unexpected hub frame: {frame:?}")),
            }
        }

        drop(thinking);
        if command_ids.is_empty() && !answer.is_empty() {
            display.end_answer();
        }

        let refused: Vec<_> = malformed
            .into_iter()
            .map(|(id, name)| {
                let content = json!({ "error": "tool call arguments must be a JSON object" });
                ToolResult {
                    id,
                    name,
                    content: content.to_string(),
                }
            })
            .collect();
        if calls.is_empty() && refused.is_empty() && !command_ids.is_empty() {
            *delivery = Delivery::new(vec![Message::Note(Note::AnswerSetAside(command_ids))]);
            continue;
        }

        if !command_ids.is_empty() && !answer.is_empty() && !calls.is_empty() {
            // While jobs are pending, visible text alongside calls is progress on stderr.
            eprintln!("{answer}");
        }
        if calls.is_empty() && refused.is_empty() {
            return Ok(answer);
        }

        let mut results = refused;
        results.extend(execute_calls(&calls, &tools, jobs, display).await);
        results.sort_by_key(|result| order.iter().position(|&id| id == result.id));
        *delivery = Delivery::new(results.into_iter().map(Message::Result).collect());
    }
}

/// Collect approvals in call order, then run adjacent command calls together.
/// Other tools stay in order with respect to commands and each other.
async fn execute_calls(
    calls: &[ParsedCall],
    tools: &Tools,
    jobs: &Jobs,
    display: &Display,
) -> Vec<ToolResult> {
    let mut answers = Vec::with_capacity(calls.len());
    for batch in calls.chunk_by(|left, right| {
        kind_of(&left.name).has_command_output() && kind_of(&right.name).has_command_output()
    }) {
        let mut approvals = Vec::with_capacity(batch.len());
        for call in batch {
            display.show_tool_call(&call.name, &call.arguments);
            let approval = approve_call(display, call).await;
            approvals.push(approval);
        }

        // A single command can use the live pane without mixing its bytes with another job.
        let pane = match (batch, approvals.as_slice()) {
            ([call], [Approval::Approved]) if kind_of(&call.name).has_command_output() => {
                display.execution_pane()
            }
            _ => None,
        };
        let results = match pane {
            Some((live, pane)) => {
                let context = ToolContext::new(jobs, Some(&live));
                pane.render(invoke_batch(batch, approvals, tools, context))
                    .await
            }
            None => {
                let context = ToolContext::new(jobs, None);
                let results = invoke_batch(batch, approvals, tools, context).await;
                for (call, result) in batch.iter().zip(&results) {
                    if kind_of(&call.name).has_command_output() {
                        let id = result.get("command_id").and_then(Value::as_u64);
                        let stdout = result.get("stdout").and_then(Value::as_str).unwrap_or("");
                        let stderr = result.get("stderr").and_then(Value::as_str).unwrap_or("");
                        display.show_tool_output(&call.name, id, stdout, stderr);
                    }
                }
                results
            }
        };

        for (call, result) in batch.iter().zip(results) {
            let kind = kind_of(&call.name);
            let command_id = result.get("command_id").and_then(Value::as_u64);
            if kind == ToolKind::RunCommand
                && result["status"] == "running"
                && let Some(id) = command_id
            {
                eprintln!("command {id} running (PID {})", result["pid"]);
            }
            answers.push(ToolResult {
                id: call.id,
                name: call.name.clone(),
                content: result.to_string(),
            });
        }
    }
    answers
}

/// Invoke the batch's approved calls together; a call that was not approved is answered with
/// why. Results come in the calls' order.
async fn invoke_batch(
    calls: &[ParsedCall],
    approvals: Vec<Approval>,
    tools: &Tools,
    context: ToolContext<'_>,
) -> Vec<Value> {
    join_all(
        calls
            .iter()
            .zip(approvals)
            .map(|(call, approval)| async move {
                match approval {
                    Approval::Denied(reason) => {
                        json!({ "error": "user denied", "reason": reason })
                    }
                    Approval::Unavailable(reason) => {
                        json!({ "error": "approval unavailable", "reason": reason })
                    }
                    Approval::Approved => {
                        let arguments = call.arguments.clone();
                        crate::tools::invoke(tools, context, &call.name, arguments).await
                    }
                }
            }),
    )
    .await
}

/// Ask the user before a call that changes their machine: a command to start, or a patch,
/// previewed in full. Any other call is approved as it is.
async fn approve_call(display: &Display, call: &ParsedCall) -> Approval {
    let kind = kind_of(&call.name);
    if kind.starts_command(&call.arguments) {
        return display.confirm(&call.name, "").await;
    }
    if kind != ToolKind::ApplyPatch {
        return Approval::Approved;
    }
    let patch = call.arguments.get("patch").and_then(Value::as_str);
    let preview = patch.and_then(summarize_patch_for_preview);
    display
        .confirm(&call.name, &preview.unwrap_or_default())
        .await
}

/// Run one turn, from a request to its answer. SIGINT cancels all of it: approval, reconnects,
/// and tool waits.
pub async fn run_turn(
    stream: &mut Option<UnixStream>,
    display: &Display,
    scene: Scene<'_>,
    request: Vec<Message>,
) -> Result<String> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let jobs = Jobs::default();
    let mut delivery = Delivery::new(request);
    let result = tokio::select! {
        biased;
        _ = interrupt.recv() => Err(TurnCancelled.into()),
        result = run_with_reconnect(stream, display, scene, &mut delivery, &jobs) => result,
    };
    if result.is_err() {
        // Closing the socket stops the hub's request and leaves the REPL free to reconnect.
        *stream = None;
    }
    jobs.shutdown().await;
    result
}

/// Retry the turn on a new connection after a dropped one, waiting longer each time, up to six
/// times in a row.
async fn run_with_reconnect(
    stream: &mut Option<UnixStream>,
    display: &Display,
    scene: Scene<'_>,
    delivery: &mut Delivery,
    jobs: &Jobs,
) -> Result<String> {
    let mut reconnect_attempts = 0;
    loop {
        let connection = connection(stream).await?;
        let result = attempt_turn(
            scene,
            delivery,
            jobs,
            connection,
            &mut reconnect_attempts,
            display,
        )
        .await;
        let Err(error) = result else { return result };
        if reconnect_attempts >= 6 || !is_disconnect(&error) {
            return Err(error);
        }

        *stream = None;
        tokio::time::sleep(Duration::from_millis(1u64 << reconnect_attempts)).await;
        reconnect_attempts += 1;
    }
}

/// Carry out a command: restart the conversation in the scene's room, from a summary of it or
/// from nothing, showing what the hub reports.
pub async fn restart(
    stream: &mut Option<UnixStream>,
    display: &Display,
    scene: Scene<'_>,
    command: Command,
) -> Result<()> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let restarted = async {
        let connection = connection(stream).await?;
        let restart = Frame::Restart {
            place: scene.place.clone(),
            setting: scene.setting.clone(),
            summarize: command == Command::Compact,
        };
        write_frame(connection, &restart).await?;
        let spinner = display.start_spinning();
        let mut store = Vec::new();
        let mut answered = false;
        loop {
            let frame = read_frame(connection, &mut store, None, None).await?;
            match frame {
                Frame::Log(line) => display.show_log(&line),
                // Asked for, the summary is the answer.
                Frame::Restarted(text) => {
                    answered = true;
                    display.show_answer(&text);
                }
                Frame::Stop => break,
                Frame::Error(error) => return Err(eyre!(error)),
                frame => return Err(eyre!("unexpected hub frame: {frame:?}")),
            }
        }
        spinner.stop().await;
        if answered {
            display.end_answer();
        }
        Ok(())
    };
    let result = tokio::select! {
        biased;
        _ = interrupt.recv() => Err(TurnCancelled.into()),
        result = restarted => result,
    };
    if result.is_err() {
        *stream = None;
    }
    result
}

/// Reuse the open connection to the hub, or open one.
async fn connection(stream: &mut Option<UnixStream>) -> Result<&mut UnixStream> {
    match stream {
        Some(connection) => Ok(connection),
        None => {
            let connection = connect_to_hub().await?;
            Ok(stream.insert(connection))
        }
    }
}

fn is_disconnect(error: &eyre::Report) -> bool {
    let io_error = match error.downcast_ref::<ProtocolError>() {
        Some(ProtocolError::Disconnect) => return true,
        Some(ProtocolError::Io(error)) => Some(error),
        _ => error.downcast_ref::<io::Error>(),
    };
    io_error.is_some_and(|error| {
        use io::ErrorKind::*;
        matches!(
            error.kind(),
            BrokenPipe | ConnectionReset | ConnectionAborted | UnexpectedEof
        )
    })
}

#[cfg(test)]
mod tests;

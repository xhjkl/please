use eyre::{Result, eyre};
use futures_util::future::join_all;
use serde_json::{Value, json};
use std::io;
use std::time::Duration;
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};

use crate::display::{Approval, Display};
use crate::protocol::{
    Frame, Message, Note, Place, ProtocolError, Setting, ToolResult, read_frame_from_stream,
    write_frame_to_stream,
};
use crate::tools::{
    Commands, ExposedTools, ToolContext, ToolKind, all_tools, kind_of, summarize_patch_for_preview,
};

use super::connect::obtain_control_stream;

#[derive(Debug, thiserror::Error)]
#[error("turn cancelled")]
pub struct TurnCancelled;

pub fn is_cancelled(error: &eyre::Report) -> bool {
    error.downcast_ref::<TurnCancelled>().is_some()
}

struct PendingToolCall {
    name: String,
    arguments: Value,
}

/// Where and how a turn happens: the hub keeps the conversation of `place`.
#[derive(Clone, Copy)]
pub struct Scene<'a> {
    pub place: &'a Place,
    pub setting: &'a Setting,
}

/// What the next request carries, under an ID that stays the same when a reconnect retries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    pub id: u64,
    pub messages: Vec<Message>,
}

impl Pending {
    pub fn new(messages: Vec<Message>) -> Self {
        use std::hash::BuildHasher;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = std::hash::RandomState::new().hash_one((count, std::process::id()));
        Self { id, messages }
    }
}

/// Run model and tool subturns on a connection, sending the hub only what is new: first the
/// turn's request, then each step's tool results. `pending` stays what the next request carries
/// until a response completes, so a retry after a reconnect resends it, under the same ID.
/// A complete model response resets the reconnect budget; a partial response does not.
async fn attempt_turn_on_stream(
    scene: Scene<'_>,
    pending: &mut Pending,
    commands: &Commands,
    stream: &mut UnixStream,
    reconnect_attempts: &mut usize,
    display: &Display,
) -> Result<String> {
    let tools = all_tools();

    loop {
        let command_ids = commands.pending_ids();
        let mut messages = pending.messages.clone();
        let set_aside = messages
            .iter()
            .any(|message| matches!(message, Message::Note(Note::AnswerSetAside(_))));
        if !command_ids.is_empty() && !set_aside {
            messages.push(Message::Note(Note::Pending(command_ids.clone())));
        }
        let mut spinner = Some(display.start_spinning().await);
        let req = Frame::Request {
            place: scene.place.clone(),
            setting: scene.setting.clone(),
            id: pending.id,
            messages,
        };
        write_frame_to_stream(stream, &req).await?;

        let mut store = Vec::with_capacity(4096);
        let mut thinking = None;
        let mut answer = String::new();
        let mut calls = Vec::new();
        let mut malformed = Vec::new();

        loop {
            let frame = read_frame_from_stream(stream, &mut store, None, None).await?;
            // Stop the spinner before its line clear can erase streamed output.
            if let Some(spinner) = spinner.take() {
                spinner.stop().await;
            }
            match frame {
                Frame::Log(line) => display.show_log(&line).await,
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
                Frame::ToolCall {
                    name,
                    arguments_json,
                } => {
                    // Only an object reaches a tool: approval reads the same fields the tool does.
                    match serde_json::from_str::<Value>(&arguments_json) {
                        Ok(arguments) if arguments.is_object() => {
                            calls.push(PendingToolCall { name, arguments })
                        }
                        _ => malformed.push(name),
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

        let refused = malformed.into_iter().map(|name| {
            let content = json!({ "error": "tool call arguments must be a JSON object" });
            Message::Result(ToolResult {
                name,
                content: content.to_string(),
            })
        });
        let refused: Vec<_> = refused.collect();
        if calls.is_empty() && refused.is_empty() && !command_ids.is_empty() {
            *pending = Pending::new(vec![Message::Note(Note::AnswerSetAside(command_ids))]);
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
        results.extend(execute_calls(&calls, &tools, commands, display).await);
        *pending = Pending::new(results);
    }
}

/// Collect approvals in call order, then run adjacent command calls together.
/// Other tools stay in order with respect to commands and each other.
async fn execute_calls(
    calls: &[PendingToolCall],
    tools: &ExposedTools,
    commands: &Commands,
    display: &Display,
) -> Vec<Message> {
    let mut messages = Vec::with_capacity(calls.len());
    for batch in calls.chunk_by(|left, right| {
        kind_of(&left.name).has_command_output() && kind_of(&right.name).has_command_output()
    }) {
        let mut approvals = Vec::with_capacity(batch.len());
        for call in batch {
            display.show_tool_call(&call.name, &call.arguments).await;
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
                let context = ToolContext::new(commands, Some(&live));
                pane.render(invoke_batch(batch, approvals, tools, context))
                    .await
            }
            None => {
                let context = ToolContext::new(commands, None);
                let results = invoke_batch(batch, approvals, tools, context).await;
                for (call, result) in batch.iter().zip(&results) {
                    if kind_of(&call.name).has_command_output() {
                        let id = result.get("commandId").and_then(Value::as_u64);
                        let stdout = result.get("stdout").and_then(Value::as_str).unwrap_or("");
                        let stderr = result.get("stderr").and_then(Value::as_str).unwrap_or("");
                        display
                            .show_tool_output(&call.name, id, stdout, stderr)
                            .await;
                    }
                }
                results
            }
        };

        for (call, result) in batch.iter().zip(results) {
            let kind = kind_of(&call.name);
            let command_id = result.get("commandId").and_then(Value::as_u64);
            if kind == ToolKind::RunCommand
                && result["status"] == "running"
                && let Some(id) = command_id
            {
                eprintln!("command {id} running (PID {})", result["pid"]);
            }
            messages.push(Message::Result(ToolResult {
                name: call.name.clone(),
                content: result.to_string(),
            }));
        }
    }
    messages
}

async fn invoke_batch(
    calls: &[PendingToolCall],
    approvals: Vec<Approval>,
    tools: &ExposedTools,
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
                        let result = crate::tools::invoke(
                            tools,
                            context,
                            &call.name,
                            call.arguments.clone(),
                        )
                        .await;
                        result.unwrap_or_else(|error| json!({ "error": error }))
                    }
                }
            }),
    )
    .await
}

async fn approve_call(display: &Display, call: &PendingToolCall) -> Approval {
    let kind = kind_of(&call.name);
    match kind {
        ToolKind::RunCommand if kind.starts_command(&call.arguments) => {
            display.confirm(&call.name, "").await
        }
        ToolKind::ApplyPatch => {
            let preview = call
                .arguments
                .get("patch")
                .and_then(Value::as_str)
                .and_then(summarize_patch_for_preview)
                .unwrap_or_default();
            display.confirm(&call.name, &preview).await
        }
        _ => Approval::Approved,
    }
}

/// Cancel the whole conversation turn on SIGINT, including approval, reconnects, and tool waits.
pub async fn run_turn(
    stream: &mut Option<UnixStream>,
    display: &Display,
    scene: Scene<'_>,
    request: Vec<Message>,
) -> Result<String> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let commands = Commands::default();
    let mut pending = Pending::new(request);
    let result = tokio::select! {
        biased;
        _ = interrupt.recv() => Err(TurnCancelled.into()),
        result = run_with_reconnect(stream, display, scene, &mut pending, &commands) => result,
    };
    if result.is_err() {
        // Closing the socket stops the hub's request and leaves the REPL free to reconnect.
        let connection = stream.take();
        drop(connection);
    }
    commands.shutdown().await;
    result
}

async fn run_with_reconnect(
    stream: &mut Option<UnixStream>,
    display: &Display,
    scene: Scene<'_>,
    pending: &mut Pending,
    commands: &Commands,
) -> Result<String> {
    let mut reconnect_attempts = 0;
    loop {
        let connection = match stream {
            Some(connection) => connection,
            None => stream.insert(obtain_control_stream().await?),
        };
        let result = attempt_turn_on_stream(
            scene,
            pending,
            commands,
            connection,
            &mut reconnect_attempts,
            display,
        )
        .await;
        let Err(error) = result else { return result };
        if reconnect_attempts >= 6 || !is_disconnect(&error) {
            return Err(error);
        }

        let connection = stream.take();
        drop(connection);
        tokio::time::sleep(Duration::from_millis(1u64 << reconnect_attempts)).await;
        reconnect_attempts += 1;
    }
}

/// Restart the scene's conversation on the hub, from a summary of it or from nothing, showing
/// what the hub reports.
pub async fn restart(
    stream: &mut Option<UnixStream>,
    display: &Display,
    scene: Scene<'_>,
    summarize: bool,
) -> Result<()> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let restarted = async {
        let connection = match stream {
            Some(connection) => connection,
            None => stream.insert(obtain_control_stream().await?),
        };
        let request = Frame::Restart {
            place: scene.place.clone(),
            setting: scene.setting.clone(),
            summarize,
        };
        write_frame_to_stream(connection, &request).await?;
        let spinner = display.start_spinning().await;
        let mut store = Vec::new();
        let mut answered = false;
        loop {
            match read_frame_from_stream(connection, &mut store, None, None).await? {
                Frame::Log(line) => display.show_log(&line).await,
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
        drop(stream.take());
    }
    result
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

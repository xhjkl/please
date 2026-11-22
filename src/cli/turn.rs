use eyre::{Result, eyre};
use futures_util::future::join_all;
use serde_json::{Value, json};
use std::io;
use std::time::Duration;
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};

use crate::display::{Approval, Display};
use crate::protocol::{
    Frame, Message, ProtocolError, read_frame_from_stream, write_frame_to_stream,
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

/// Run model and tool subturns on a connection, preserving history and command state.
/// A complete model response resets the reconnect budget; a partial response does not.
async fn attempt_turn_on_stream(
    messages: &mut Vec<Message>,
    commands: &Commands,
    stream: &mut UnixStream,
    reconnect_attempts: &mut usize,
    display: &Display,
) -> Result<String> {
    let tools = all_tools();

    loop {
        let command_ids = commands.pending_ids();
        if !command_ids.is_empty() {
            let reminder = pending_command_prompt(&command_ids);
            // A retried subturn finds the same reminder already recorded in history.
            if !matches!(messages.last(), Some(Message::Developer(text)) if *text == reminder) {
                messages.push(Message::Developer(reminder));
            }
        }
        let mut spinner = Some(display.start_spinning().await);
        let req = Frame::Request {
            messages: messages.clone(),
        };
        write_frame_to_stream(stream, &req).await?;

        let mut store = Vec::with_capacity(4096);
        let mut thinking = None;
        let mut answer = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        let mut tool_parse_error = None;

        loop {
            let frame = read_frame_from_stream(stream, &mut store, None, None).await?;
            // Stop the spinner before its line clear can erase streamed output.
            if let Some(spinner) = spinner.take() {
                spinner.stop().await;
            }
            match frame {
                Frame::Log(line) => display.show_log(&line).await,
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
                    reasoning.push_str(&delta);
                }
                Frame::ToolCall {
                    name,
                    arguments_json,
                } => {
                    let arguments = serde_json::from_str::<Value>(&arguments_json);
                    match arguments {
                        Err(error) => {
                            tool_parse_error =
                                Some(format!("error parsing tool call arguments: {error}"))
                        }
                        Ok(arguments) if !arguments.is_object() => {
                            tool_parse_error = Some("tool call arguments must be an object".into());
                        }
                        Ok(arguments) => calls.push(PendingToolCall { name, arguments }),
                    }
                }
                Frame::ToolCallParseError(error) => tool_parse_error = Some(error),
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

        if tool_parse_error.is_none() && calls.is_empty() && !command_ids.is_empty() {
            messages.push(Message::Developer(format!(
                "Previous answer ignored. {}",
                pending_command_prompt(&command_ids)
            )));
            continue;
        }

        if !command_ids.is_empty() {
            if calls.is_empty() {
                answer.clear();
            } else if !answer.is_empty() {
                // While jobs are pending, visible text alongside calls is progress on stderr.
                eprintln!("{answer}");
            }
        }

        // Recording empty reasoning gives a tool-only response its own assistant step.
        if !reasoning.is_empty() || answer.is_empty() && !calls.is_empty() {
            messages.push(Message::Reasoning(reasoning));
        }
        if !answer.is_empty() {
            messages.push(Message::Assistant(answer.clone()));
        }
        if let Some(error) = tool_parse_error {
            let payload = json!({ "tool": "tool_call_parse_error", "result": { "error": error } });
            messages.push(Message::Tool(payload.to_string()));
            continue;
        }
        if calls.is_empty() {
            return Ok(answer);
        }

        execute_calls(&calls, &tools, commands, display, messages).await;
    }
}

/// Collect approvals in call order, then run adjacent command calls together.
/// Other tools stay in order with respect to commands and each other.
async fn execute_calls(
    calls: &[PendingToolCall],
    tools: &ExposedTools,
    commands: &Commands,
    display: &Display,
    messages: &mut Vec<Message>,
) {
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
            let payload =
                json!({ "tool": call.name, "arguments": call.arguments, "result": result });
            messages.push(Message::Tool(payload.to_string()));
        }
    }
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

fn pending_command_prompt(ids: &[u64]) -> String {
    let ids = ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Pending commandId values: {ids}. You can continue other work. Before answering, collect a final result for each with {} and action=\"wait\" or action=\"kill\".",
        crate::tools::CONTROL_COMMAND_NAME
    )
}

/// Cancel the whole conversation turn on SIGINT, including approval, reconnects, and tool waits.
pub async fn run_turn(
    stream: &mut Option<UnixStream>,
    display: &Display,
    messages: &mut Vec<Message>,
) -> Result<String> {
    let mut interrupt = signal(SignalKind::interrupt())?;
    let commands = Commands::default();
    let result = tokio::select! {
        biased;
        _ = interrupt.recv() => Err(TurnCancelled.into()),
        result = run_with_reconnect(stream, display, messages, &commands) => result,
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
    messages: &mut Vec<Message>,
    commands: &Commands,
) -> Result<String> {
    let mut reconnect_attempts = 0;
    loop {
        let connection = match stream {
            Some(connection) => connection,
            None => stream.insert(obtain_control_stream().await?),
        };
        let result = attempt_turn_on_stream(
            messages,
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

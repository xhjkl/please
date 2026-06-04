use eyre::{Result, eyre};
use std::sync::Arc;
use tokio::net::UnixStream;

use crate::display::Display;
use crate::protocol::{Frame, Message, read_frame_from_stream};
use crate::tools::{Stride, ToolKind, all_tools, kind_of, summarize_patch_for_preview};

use super::connect::obtain_control_stream;

#[derive(Debug)]
pub struct TurnCancelled;

impl std::fmt::Display for TurnCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "turn cancelled")
    }
}

impl std::error::Error for TurnCancelled {}

pub fn is_cancelled(error: &eyre::Report) -> bool {
    error.downcast_ref::<TurnCancelled>().is_some()
}

/// Run a single turn attempt, preserving the full message history across reconnects.
/// Send a prompt to the hub and multiplex streamed frames to display channels.
/// Returns the final answer string.
pub async fn attempt_turn_on_stream(
    stream: &mut UnixStream,
    display: Arc<Display>,
    messages: &mut Vec<Message>,
    stride: Stride,
) -> Result<String> {
    use tokio::io::AsyncWriteExt;

    enum Phase {
        Answering,
        Thinking,
    }

    struct PendingToolCall {
        name: String,
        arguments: serde_json::Value,
    }

    let tools = all_tools();

    loop {
        let mut spinner = Some(display.start_spinning().await);
        let running_command_pids = stride.running_command_pids().await;
        let must_settle_command = !running_command_pids.is_empty();

        // Send full structured message history to the hub for this subturn.
        let mut request_messages = messages.clone();
        if must_settle_command {
            request_messages.push(Message::Developer(settle_command_prompt(
                &running_command_pids,
            )));
        }
        let req = Frame::Request {
            messages: request_messages,
        };
        let body = postcard::to_allocvec(&req).map_err(|e| eyre!(e))?;
        stream.write_all(&body).await?;

        let mut store = Vec::with_capacity(4096);
        let mut phase = Phase::Answering;
        let mut final_answer = String::new();
        let mut answer = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        let mut tool_parse_error = None;

        // Stream frames for this subturn
        loop {
            let frame: Frame = tokio::select! {
                frame = read_frame_from_stream(stream, &mut store, None, None) => {
                    frame.map_err(|error| eyre!(error))?
                }
                _ = tokio::signal::ctrl_c() => {
                    let _ = stream.shutdown().await;
                    stride.kill_running_commands().await;
                    return Err(eyre!(TurnCancelled));
                }
            };
            // Stop spinner before streaming output so its line clear cannot erase the first token.
            if let Some(spinner) = spinner.take() {
                spinner.stop().await;
            }
            match frame {
                Frame::Log(line) => {
                    let _ = display.show_log(&line).await;
                }
                Frame::Answer(delta) => {
                    if must_settle_command {
                        final_answer.push_str(&delta);
                        answer.push_str(&delta);
                        continue;
                    }
                    if matches!(phase, Phase::Thinking) {
                        let _ = display.end_thinking().await;
                    }
                    phase = Phase::Answering;
                    let _ = display.show_delta(&delta).await;
                    final_answer.push_str(&delta);
                    answer.push_str(&delta);
                }
                Frame::Thinking(delta) => {
                    if must_settle_command {
                        reasoning.push_str(&delta);
                        continue;
                    }
                    if !matches!(phase, Phase::Thinking) {
                        let _ = display.start_thinking().await;
                    }
                    phase = Phase::Thinking;
                    let _ = display.show_delta(&delta).await;
                    reasoning.push_str(&delta);
                }
                Frame::ToolCall {
                    name,
                    arguments_json,
                } => match serde_json::from_str(&arguments_json) {
                    Ok(arguments) => calls.push(PendingToolCall { name, arguments }),
                    Err(error) => {
                        tool_parse_error =
                            Some(format!("error parsing tool call arguments: {error}"));
                    }
                },
                Frame::ToolCallParseError(error) => {
                    tool_parse_error = Some(error);
                }
                Frame::Stop => break,
                Frame::Request { .. } => {}
            }
        }

        if matches!(phase, Phase::Thinking) {
            let _ = display.end_thinking().await;
        }
        let _ = display.end_answer().await;

        let missing_required_control = must_settle_command
            && !calls
                .iter()
                .any(|call| kind_of(&call.name).is_control_command());

        // If present, preserve reasoning across subturns without displaying it to the user.
        if !reasoning.is_empty() && !missing_required_control {
            messages.push(Message::Reasoning(reasoning));
        }
        // Preserve assistant-visible content across subturns.
        if !answer.is_empty() && !must_settle_command {
            messages.push(Message::Assistant(answer));
        }
        if let Some(error) = tool_parse_error {
            let payload = serde_json::json!({
                "tool": "tool_call_parse_error",
                "result": { "error": error },
            });
            messages.push(Message::Tool(payload.to_string()));
            continue;
        }
        if missing_required_control {
            messages.push(Message::Developer(format!(
                "Previous response ignored: {}",
                settle_command_prompt(&running_command_pids)
            )));
            continue;
        }
        if calls.is_empty() {
            // The turn is complete, return the final answer.
            stride.kill_running_commands().await;
            return Ok(final_answer);
        }

        // Execute tools and append tool results to history, then continue the loop
        for call in calls {
            let name = call.name;
            let args = call.arguments;
            let kind = kind_of(&name);

            // Show pretty formatted function call
            let _ = display.show_tool_call(&name, &args).await;

            if must_settle_command && !kind.is_control_command() {
                let tool_payload = serde_json::json!({
                    "tool": name,
                    "arguments": args,
                    "result": { "error": format!("{} required while a command is running", crate::tools::CONTROL_COMMAND_NAME) }
                });
                messages.push(Message::Tool(tool_payload.to_string()));
                continue;
            }

            let approved = gate_risky_if_needed(&display, kind, &args).await;
            if !approved {
                let tool_payload = serde_json::json!({
                    "tool": name,
                    "arguments": args,
                    "result": { "error": "user denied" }
                });
                messages.push(Message::Tool(tool_payload.to_string()));
                continue;
            }

            let starts_command = kind.starts_command(&args);
            // Only newly-started commands get a live output pane for streaming stdout/stderr.
            let execution_pane = if starts_command {
                display.start_executing()
            } else {
                None
            };
            let stride = stride.with_live_output(execution_pane.as_ref().map(|pane| pane.sender()));
            let streamed = starts_command && execution_pane.is_some();
            let result = tokio::select! {
                result = crate::tools::invoke(&tools, stride.clone(), &name, args.clone()) => {
                    result.unwrap_or_else(|error| serde_json::json!({ "error": error }))
                }
                _ = tokio::signal::ctrl_c() => {
                    drop(execution_pane);
                    stride.kill_running_commands().await;
                    return Err(eyre!(TurnCancelled));
                }
            };

            drop(execution_pane);

            if !streamed && kind.has_command_output() {
                // For plain display mode, forward stdout/stderr all at once.
                if let Some(obj) = result.as_object() {
                    let stdout = obj.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
                    let stderr = obj.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
                    display.show_tool_output(&name, stdout, stderr).await;
                }
            }

            let tool_payload =
                serde_json::json!({ "tool": name, "arguments": args.clone(), "result": result });
            messages.push(Message::Tool(tool_payload.to_string()));
        }
        // Loop continues: send a new Request with updated history to get the assistant to use the tool results
    }
}

/// Run a single turn while tapping the answer stream to collect a full string.
/// Send a prompt to the hub and multiplex streamed frames to display channels.
/// This may reconnect to the hub if the connection is lost.
/// Returns the final answer string.
pub async fn run_turn(
    stream: &mut UnixStream,
    display: Arc<Display>,
    messages: Vec<Message>,
) -> Result<String> {
    use std::time::Duration;
    fn is_disconnect(e: &eyre::Report) -> bool {
        if let Some(pe) = e.downcast_ref::<crate::protocol::ProtocolError>() {
            return matches!(pe, crate::protocol::ProtocolError::Disconnect);
        }
        if let Some(ioe) = e.downcast_ref::<std::io::Error>() {
            use std::io::ErrorKind::*;
            return matches!(
                ioe.kind(),
                BrokenPipe | ConnectionReset | ConnectionAborted | UnexpectedEof
            );
        }
        false
    }

    let max_attempts = 6;
    let mut attempt = 0;
    let mut messages = messages;
    let stride = Stride::default();

    loop {
        match attempt_turn_on_stream(stream, display.clone(), &mut messages, stride.clone()).await {
            Ok(s) => return Ok(s),
            Err(e) => {
                if !is_disconnect(&e) {
                    stride.kill_running_commands().await;
                    return Err(e);
                }
                if attempt >= max_attempts {
                    stride.kill_running_commands().await;
                    return Err(e);
                }

                tokio::time::sleep(Duration::from_millis(1u64 << attempt.min(6))).await;

                let mut new_stream = obtain_control_stream().await?;
                std::mem::swap(stream, &mut new_stream);

                attempt += 1;
                continue;
            }
        }
    }
}

fn settle_command_prompt(pids: &[u32]) -> String {
    let pids = pids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Command pid(s) {pids} are still running. Call {} now with action=\"wait\" or action=\"kill\"; do not answer final.",
        crate::tools::CONTROL_COMMAND_NAME
    )
}

async fn gate_risky_if_needed(display: &Display, kind: ToolKind, args: &serde_json::Value) -> bool {
    match kind {
        ToolKind::RunCommand => {
            let argv: Vec<String> = args
                .get("argv")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(|t| t.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            if argv.is_empty() {
                return true;
            }
            display.confirm_run_command_execution(&argv).await
        }
        ToolKind::ApplyPatch => {
            let preview = match args.get("patch").and_then(|v| v.as_str()) {
                Some(patch) => summarize_patch_for_preview(patch).unwrap_or_default(),
                None => String::new(),
            };
            display.confirm_apply_patch_edits(&preview).await
        }
        ToolKind::ControlCommand | ToolKind::Other => true,
    }
}

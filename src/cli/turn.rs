use eyre::{Result, eyre};
use std::sync::Arc;
use tokio::net::UnixStream;

use crate::display::Display;
use crate::harmony::{HarmonyEvent, HarmonyMessageHandler, HarmonyParser};
use crate::protocol::{Frame, Message, read_frame_from_stream};
use crate::tools::{all_tools, invoke, summarize_patch_for_preview, to_harmony};

use super::connect::obtain_control_stream;

/// Run a single turn attempt, preserving the full message history across reconnects.
/// Send a prompt to the hub and multiplex streamed frames to display channels.
/// Returns the final answer string.
pub async fn attempt_turn_on_stream(
    stream: &mut UnixStream,
    display: Arc<Display>,
    messages: &mut Vec<Message>,
) -> Result<String> {
    use tokio::io::AsyncWriteExt;

    enum Phase {
        Answering,
        Thinking,
        ToolCalling,
    }

    let tools = all_tools();
    let tool_specs = to_harmony(&tools);

    loop {
        let mut spinner = Some(display.start_spinning().await);

        // Send full structured message history to the hub for this subturn.
        let req = Frame::Request {
            messages: messages.clone(),
        };
        let body = postcard::to_allocvec(&req).map_err(|e| eyre!(e))?;
        stream.write_all(&body).await?;

        let mut store = Vec::with_capacity(4096);
        let mut parser = HarmonyParser::new();
        parser.add_implicit_start();
        let mut handler = HarmonyMessageHandler::new();
        let _ = handler.init(&tool_specs, None);

        let mut phase = Phase::Answering;
        let mut final_answer = String::new();
        let mut answer = String::new();
        let mut reasoning = String::new();
        let mut did_send_answer_end = false;

        // Stream frames for this subturn
        loop {
            let frame: Frame = read_frame_from_stream(stream, &mut store, None, None)
                .await
                .map_err(|e| eyre!(e))?;
            // Stop spinner on first received frame if not dropped already
            let _ = spinner.take().map(drop);
            match frame {
                Frame::Log(line) => {
                    let _ = display.show_log(&line).await;
                }
                Frame::Answer(delta) => {
                    // For display: drive UI phases with the event parser
                    for ev in parser.add_content(&delta) {
                        match ev {
                            HarmonyEvent::HeaderComplete { header } => {
                                if !header.recipient.is_empty() {
                                    phase = Phase::ToolCalling;
                                } else {
                                    match header.channel.as_str() {
                                        "analysis" => {
                                            let _ = display.start_thinking().await;
                                            phase = Phase::Thinking;
                                        }
                                        "commentary" | "final" => {
                                            let _ = display.end_thinking().await;
                                            phase = Phase::Answering;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            HarmonyEvent::ContentEmitted { content } => {
                                match phase {
                                    Phase::ToolCalling => {
                                        // Suppress raw tool-call JSON streaming; we'll render a pretty call later.
                                    }
                                    Phase::Answering | Phase::Thinking => {
                                        let _ = display.show_delta(&content).await;
                                        if matches!(phase, Phase::Answering) {
                                            final_answer.push_str(&content);
                                        }
                                    }
                                }
                            }
                            HarmonyEvent::MessageEnd => {
                                match phase {
                                    Phase::Thinking => {
                                        let _ = display.end_thinking().await;
                                    }
                                    Phase::Answering => {
                                        let _ = display.end_answer().await;
                                        did_send_answer_end = true;
                                    }
                                    Phase::ToolCalling => {}
                                }
                                phase = Phase::Answering;
                            }
                            HarmonyEvent::MessageStart => {}
                        }
                    }
                    // For tools: feed the same delta into the handler and accumulate parsed content
                    if let Ok((a, t, _)) = handler.add(&delta, false) {
                        answer.push_str(&a);
                        reasoning.push_str(&t);
                    }
                }
                Frame::Stop => break,
                Frame::Request { .. } => {}
            }
        }

        if !did_send_answer_end {
            let _ = display.end_answer().await;
        }

        // Finalize tool calls for this subturn
        let mut had_tool_parse_error = false;
        let calls = match handler.add("", true) {
            Ok((_, _, calls)) => calls,
            Err(e) => {
                // Surface parse errors to history so the model can self-correct
                let payload = serde_json::json!({
                    "tool": "tool_call_parse_error",
                    "result": { "error": e },
                });
                messages.push(Message::Tool(payload.to_string()));
                had_tool_parse_error = true;
                Vec::new()
            }
        };

        // If present, preserve reasoning across subturns without displaying it to the user.
        if !reasoning.is_empty() {
            messages.push(Message::Reasoning(reasoning));
        }
        // Preserve assistant-visible content across subturns.
        if !answer.is_empty() {
            messages.push(Message::Assistant(answer));
        }
        if calls.is_empty() {
            if had_tool_parse_error {
                // No tool executed, but we surfaced an error → continue the loop
                continue;
            }
            // The turn is complete, return the final answer.
            return Ok(final_answer);
        }

        // Execute tools and append tool results to history, then continue the loop
        for call in calls {
            let name = call.function.name.clone();
            let args = call.function.arguments.clone();

            // Show pretty formatted function call
            let _ = display.show_tool_call(&name, &args).await;

            let approved = gate_risky_if_needed(&*display, &name, &args).await;
            if !approved {
                let tool_payload = serde_json::json!({
                    "tool": name,
                    "arguments": args,
                    "result": { "error": "user denied" }
                });
                messages.push(Message::Tool(tool_payload.to_string()));
                continue;
            }

            let result = invoke(&tools, &name, args.clone())
                .await
                .unwrap_or_else(|e| serde_json::json!({ "error": e }));

            forward_output_if_needed(&*display, &name, &result).await;

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

    loop {
        match attempt_turn_on_stream(stream, display.clone(), &mut messages).await {
            Ok(s) => return Ok(s),
            Err(e) => {
                if !is_disconnect(&e) {
                    return Err(e);
                }
                if attempt >= max_attempts {
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

async fn gate_risky_if_needed(display: &Display, name: &str, args: &serde_json::Value) -> bool {
    if name == "run_command" {
        let argv: Vec<String> = args
            .get("argv")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(|t| t.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        return display.confirm_run_command_execution(&argv).await;
    }
    if name == "apply_patch" {
        let preview = match args.get("patch").and_then(|v| v.as_str()) {
            Some(patch) => summarize_patch_for_preview(patch).unwrap_or_default(),
            None => String::new(),
        };
        return display.confirm_apply_patch_edits(&preview).await;
    }
    true
}

async fn forward_output_if_needed(display: &Display, name: &str, result: &serde_json::Value) {
    if name == "run_command"
        && let Some(obj) = result.as_object()
        && obj.get("ok").and_then(|v| v.as_bool()) == Some(true)
    {
        if let Some(stdout) = obj.get("stdout").and_then(|v| v.as_str()) {
            let _ = display.show_log(&format!("stdout:\n{}", stdout)).await;
        }
        if let Some(stderr) = obj.get("stderr").and_then(|v| v.as_str()) {
            let _ = display.show_log(&format!("stderr:\n{}", stderr)).await;
        }
    }
}

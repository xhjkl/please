use eyre::Result;
use serde_json::Value;

use crate::protocol::Message;

/// Render a Harmony prompt from a history of messages with correct tool **call** and **result** shapes.
/// - Assistant tool-call:
///   `<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{ARGS_JSON}<|call|>`
/// - Tool result:
///   `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>{PAYLOAD}<|end|>`
pub fn render_prompt_from_history(
    messages: &[Message],
    append_assistant_start: bool,
) -> Result<String> {
    fn push_segment(buf: &mut String, head: &str, body: &str) {
        buf.push_str("<|start|>");
        buf.push_str(head);
        buf.push_str("<|message|>");
        buf.push_str(body);
        buf.push_str("<|end|>");
    }

    fn push_assistant(buf: &mut String, channel: &str, body: &str) {
        buf.push_str("<|start|>assistant<|channel|>");
        buf.push_str(channel);
        buf.push_str("<|message|>");
        buf.push_str(body);
        buf.push_str("<|end|>");
    }

    fn push_tool_call(buf: &mut String, name: &str, args_json: &str) {
        // assistant tool-call line (JSON-only constraint, ends with <|call|>)
        buf.push_str("<|start|>assistant<|channel|>commentary to=functions.");
        buf.push_str(name);
        buf.push_str(" <|constrain|>json<|message|>");
        buf.push_str(args_json);
        buf.push_str("<|call|>");
    }

    fn push_tool_result(buf: &mut String, name: &str, payload: &str) {
        // tool result line
        buf.push_str("<|start|>functions.");
        buf.push_str(name);
        buf.push_str(" to=assistant<|channel|>commentary<|message|>");
        buf.push_str(payload);
        buf.push_str("<|end|>");
    }

    let mut out = String::new();

    for m in messages {
        match m {
            Message::System(s) => push_segment(&mut out, "system", s),
            Message::Developer(s) => push_segment(&mut out, "developer", s),
            Message::User(s) => push_segment(&mut out, "user", s),

            Message::Assistant(s) => push_assistant(&mut out, "final", s),
            Message::Reasoning(s) => push_assistant(&mut out, "analysis", s),

            // Tool message can contain arguments or result. Support both.
            // Expected JSON shapes (any of them):
            // { "tool":"list_files", "arguments":{...}, "result": ... }
            // { "tool":"list_files", "arguments":{...} }                // call only
            // { "tool":"list_files", "result": ... }                    // result only
            // If malformed, fallback to a plain assistant commentary block.
            Message::Tool(s) => match serde_json::from_str::<Value>(s) {
                Ok(val) => {
                    let tool_name = val.get("tool").and_then(Value::as_str).unwrap_or_default();

                    // emit call if we have arguments
                    if let Some(args) = val.get("arguments") {
                        let args_json = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
                        push_tool_call(&mut out, tool_name, &args_json);
                    }

                    // emit result if we have it (string or any JSON)
                    if let Some(res) = val.get("result") {
                        let payload = if let Some(s) = res.as_str() {
                            s.to_owned()
                        } else {
                            serde_json::to_string(res).unwrap_or_else(|_| "null".into())
                        };
                        push_tool_result(&mut out, tool_name, &payload);
                    }

                    // if neither present, treat whole blob as the payload (result-only)
                    if val.get("arguments").is_none() && val.get("result").is_none() {
                        push_tool_result(&mut out, tool_name, s);
                    }
                }
                // malformed Tool message → make it visible but harmless
                Err(_) => push_assistant(&mut out, "commentary", s),
            },
        }
    }

    if append_assistant_start {
        // Cue next assistant turn; model chooses the channel.
        out.push_str("<|start|>assistant");
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_and_result_are_wired() {
        let msgs = &[
            Message::User("size?".into()),
            Message::Tool(r#"{ "tool":"run_command", "arguments":{"argv":["bash","-lc","du -sh target src"]}, "result":{"ok":true}}"#.into()),
        ];
        let p = render_prompt_from_history(msgs, true).unwrap();
        assert!(p.contains("assistant<|channel|>commentary to=functions.run_command"));
        assert!(p.contains("<|constrain|>json<|message|>{\"argv\":[\"bash\",\"-lc\",\"du -sh target src\"]}<|call|>"));
        assert!(p.contains("<|start|>functions.run_command to=assistant<|channel|>commentary<|message|>{\"ok\":true}<|end|>"));
        assert!(p.ends_with("<|start|>assistant"));
    }
}

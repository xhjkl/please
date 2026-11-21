//! Probe-local context and conversation history; model prompts belong to the hub.
use crate::protocol::{Context, Message};

/// Capture session facts without selecting a model or formatting its prompt.
pub fn make_history(
    stdin_content: Option<String>,
    stdout_redirection_path: Option<String>,
) -> Vec<Message> {
    let now = time::OffsetDateTime::now_local();
    let now = match now {
        Err(_) => time::OffsetDateTime::now_utc(),
        Ok(now) => now,
    };
    let now = now.date().to_string();
    let reasoning = std::env::var("PLEASE_TRY");
    let reasoning = reasoning
        .ok()
        .map(|v| v.trim().to_lowercase())
        .and_then(|v| match v.as_str() {
            _ if v.starts_with("h") => Some("high".to_string()),
            _ if v.starts_with("m") => Some("medium".to_string()),
            _ if v.starts_with("l") => Some("low".to_string()),
            _ if v.starts_with("e") => Some("low".to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "medium".to_string());
    vec![Message::Context(Context {
        today: now,
        reasoning,
        stdin: stdin_content,
        stdout_redirection: stdout_redirection_path,
    })]
}

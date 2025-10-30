//! Extensions to handle lists of messages.
use crate::prompting::SYSTEM_PREAMBLE;
use crate::protocol::Message;

/// Compose a full session history from the default preamble
/// and optional stdin/extra contexts in the canonical order.
pub fn make_history(
    stdin_content: Option<String>,
    stdout_redirection_path: Option<String>,
) -> Vec<Message> {
    let now = time::OffsetDateTime::now_local().unwrap_or_else(|_| time::OffsetDateTime::now_utc());
    let now = now.date().to_string();
    let reasoning = std::env::var("PLEASE_TRY")
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
    let mut history = vec![Message::System(
        SYSTEM_PREAMBLE
            .replace("¶cutoff", "2024-06")
            .replace("¶today", &now)
            .replace("¶reasoning", &reasoning),
    )];
    let guidance = crate::prompting::TOOL_GUIDANCE.trim();
    if !guidance.is_empty() {
        history.push(Message::Developer(guidance.to_string()));
    }
    if let Some(s) = stdin_content {
        let s = s.trim();
        if !s.is_empty() {
            history.push(Message::Developer(
                "The next message is the full stdin content.".to_string(),
            ));
            history.push(Message::Developer(s.to_string()));
        }
    }
    match stdout_redirection_path {
        Some(path) if path.is_empty() => {
            history.push(Message::Developer(
                "Your final answer output is redirected to a file, so do not fence anything and produce the content directly without any extra prose.".to_string(),
            ));
        }
        Some(path) => {
            history.push(Message::Developer(format!(
                "Your final answer output is redirected to file named `{path}`, so do not fence anything and produce the content directly without any extra prose."
            )));
        }
        None => {}
    }
    history
}

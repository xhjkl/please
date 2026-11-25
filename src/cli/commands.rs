//! Commands on the conversation, typed at the prompt or given as the whole one-shot request.

/// What a command asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Restart the conversation from a message summing it up.
    Compact,
    /// Set the conversation aside and start a new one.
    New,
}

/// The command a prompt line spells, `None` for an ordinary request, or an error naming the
/// commands. Only a lone `/word` is a command, so requests about paths like `/etc/hosts` stay
/// requests, and so does a bare `new` answering the model's question.
pub fn parse(line: &str) -> Option<Result<Command, String>> {
    let name = line.trim().strip_prefix('/')?;
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    Some(match name.to_lowercase().as_str() {
        "compact" | "compress" | "condense" | "squash" | "summarize" | "summarise" => {
            Ok(Command::Compact)
        }
        "new" | "clear" | "reset" | "fresh" | "forget" => Ok(Command::New),
        other => Err(format!(
            "unknown command /{other}; try /compact to condense the conversation or /new to start over"
        )),
    })
}

/// The command a whole one-shot request spells, as `parse` reads it, except that a command's
/// own bare name counts too: `please compact` can only mean the command. Synonyms still need
/// the slash, since `please clear` or `please summarize` are everyday requests.
pub fn parse_request(prompt: &str) -> Option<Result<Command, String>> {
    match prompt.trim().to_lowercase().as_str() {
        "compact" => Some(Ok(Command::Compact)),
        "new" => Some(Ok(Command::New)),
        _ => parse(prompt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_accept_synonyms_and_ordinary_text_is_no_command() {
        assert_eq!(parse("/compact"), Some(Ok(Command::Compact)));
        assert_eq!(parse("  /Condense "), Some(Ok(Command::Compact)));
        assert_eq!(parse("/squash"), Some(Ok(Command::Compact)));
        assert_eq!(parse("/clear"), Some(Ok(Command::New)));
        assert_eq!(parse("/new"), Some(Ok(Command::New)));
        assert_eq!(parse("new"), None);
        assert_eq!(parse_request("compact"), Some(Ok(Command::Compact)));
        assert_eq!(parse_request(" New\n"), Some(Ok(Command::New)));
        assert_eq!(parse_request("/clear"), Some(Ok(Command::New)));
        assert_eq!(parse_request("summarize"), None);
        assert_eq!(parse_request("clear"), None);
        assert!(matches!(parse("/frobnicate"), Some(Err(_))));
        assert_eq!(parse("compact the logs"), None);
        assert_eq!(parse("summarize ./a/b"), None);
        assert_eq!(parse("/etc/hosts looks wrong"), None);
        assert_eq!(parse("/"), None);
    }
}

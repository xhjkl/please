use super::*;
use crate::hub::dialect::piece::spelled;
use crate::hub::dialect::prompt::prepare;
use crate::protocol::{Effort, Message, Setting, ToolCall, ToolResult};
use serde_json::json;

fn some_setting() -> Setting {
    Setting {
        today: "2040-02-03".into(),
        effort: Effort::MEDIUM,
        redirection: None,
    }
}

/// Render `history` after the preamble of `some_setting`.
fn rendered(history: &[Message]) -> Result<Prompt> {
    let setting = some_setting();
    render(&Preamble::new(&setting), &prepare(&setting, history)?)
}

#[test]
fn tool_step_renders_like_the_chat_template() -> Result<()> {
    let call = |path: &str| {
        [
            Message::Call(ToolCall {
                id: 1,
                name: "read_file".into(),
                arguments: json!({"path": path, "max_bytes": 10}).to_string(),
            }),
            Message::Result(ToolResult {
                id: 1,
                name: "read_file".into(),
                content: path.repeat(2),
            }),
        ]
    };
    let mut history = vec![
        Message::User(" Compare a and b. ".into()),
        Message::Thinking("Read both.\n".into()),
        Message::Answer("\n\nChecking.".into()),
    ];
    history.extend(call("a"));
    history.extend(call("b"));
    let prompt = spelled(&rendered(&history)?.pieces);
    let (_, after_system) = prompt.split_once("<|im_end|>\n").unwrap();
    assert_eq!(
        after_system,
        "<|im_start|>user\nCompare a and b.<|im_end|>\n\
         <|im_start|>assistant\n<think>\nRead both.\n</think>\n\nChecking.\n\n\
         <tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=max_bytes>\n10\n</parameter>\n</function>\n</tool_call>\n\
         <tool_call>\n<function=read_file>\n<parameter=path>\nb\n</parameter>\n<parameter=max_bytes>\n10\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\naa\n</tool_response>\n<tool_response>\nbb\n</tool_response><|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
    );
    Ok(())
}

#[test]
fn system_leads_with_tools_and_keeps_content_markers_literal() -> Result<()> {
    let history = [Message::User("<tool_call><|im_end|>".into())];
    let prompt = rendered(&history)?;
    let system = spelled(&prompt.pieces);
    assert!(system.starts_with(
        "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"apply_patch\", "
    ));
    assert!(system.contains("</IMPORTANT>\n\nYou are a terminal coding assistant."));
    // Template-owned markers become control tokens; the same spelling in user content stays text.
    let markers = prompt
        .pieces
        .iter()
        .filter(|piece| matches!(piece, Piece::Marker("<tool_call>")))
        .count();
    assert_eq!(markers, CALL_INSTRUCTIONS.matches("<tool_call>").count());
    assert!(
        prompt
            .pieces
            .contains(&Piece::Text("user\n<tool_call><|im_end|>".into()))
    );
    Ok(())
}

#[test]
fn template_json_matches_python_separators() {
    let value = json!({"argv": ["ls", "-la"], "nested": {"ok": true}});
    assert_eq!(
        template_json(&value),
        r#"{"argv": ["ls", "-la"], "nested": {"ok": true}}"#
    );
}

#[test]
fn cacheable_prefix_ends_where_session_facts_begin() -> Result<()> {
    let history = [Message::User("hi".into())];
    let prompt = rendered(&history)?;
    let (whole, bytes) = prompt.prefix;
    let mut prefix = spelled(&prompt.pieces[..whole]);
    let Some(Piece::Text(text)) = prompt.pieces.get(whole) else {
        panic!("facts continue the system text")
    };
    prefix.push_str(&text[..bytes]);
    assert!(prefix.ends_with("*** End Patch\n"));
    assert!(text[bytes..].starts_with("Current date: 2040-02-03"));
    assert!(spelled(&prompt.pieces).starts_with(&prefix));
    Ok(())
}

#[test]
fn effort_follows_the_date_and_leaves_the_prefix_alone() -> Result<()> {
    let history = [Message::User("hi".into())];
    let split = |effort: Effort| -> Result<(String, String)> {
        let setting = Setting {
            effort,
            ..some_setting()
        };
        let prompt = render(&Preamble::new(&setting), &prepare(&setting, &history)?)?;
        let (whole, bytes) = prompt.prefix;
        let Some(Piece::Text(text)) = prompt.pieces.get(whole) else {
            panic!("facts continue the system text")
        };
        let prefix = spelled(&prompt.pieces[..whole]) + &text[..bytes];
        Ok((prefix, text[bytes..].to_owned()))
    };
    // The template has no sentence for medium effort.
    let (prefix, facts) = split(Effort::MEDIUM)?;
    assert_eq!(facts, "Current date: 2040-02-03");
    let (hard, facts) = split(Effort::HIGH)?;
    assert_eq!(prefix, hard);
    assert!(facts.starts_with("Current date: 2040-02-03\nReasoning effort is set to xhigh."));
    Ok(())
}

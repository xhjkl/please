use super::*;
use crate::protocol::{Context, Frame};
use conversation::{ConversationMessage, prepare};
use serde_json::json;

fn context() -> Context {
    Context {
        today: "2040-02-03".into(),
        reasoning: "high".into(),
        stdin: Some("literal ¶today and ¶reasoning".into()),
        stdout_redirection: Some("notes-¶today.txt".into()),
    }
}

#[test]
fn protocol_context_expands_only_application_templates() -> Result<()> {
    let frame = Frame::Request {
        messages: vec![
            Message::Context(context()),
            Message::User("Keep ¶today, <|start|>, and <tool_call> literal.".into()),
        ],
    };
    let bytes = postcard::to_allocvec(&frame)?;
    let Frame::Request { messages } = postcard::from_bytes(&bytes)? else {
        unreachable!()
    };
    let prepared = prepare(&messages)?;
    assert_eq!(prepared[0], ConversationMessage::Effort("high".into()));
    let ConversationMessage::System(system) = &prepared[1] else {
        panic!("context expands on the hub")
    };
    assert!(!system.contains('¶'));
    // Everything that varies between invocations follows the static instructions.
    let first_fact = prepared
        .iter()
        .position(|message| matches!(message, ConversationMessage::Fact(_)))
        .unwrap();
    assert!(prepared[..first_fact].iter().all(|message| matches!(
        message,
        ConversationMessage::Effort(_)
            | ConversationMessage::System(_)
            | ConversationMessage::Developer(_)
    )));
    assert_eq!(
        prepared[first_fact],
        ConversationMessage::Fact("Current date: 2040-02-03".into())
    );
    // Piped stdin is user data, delimited, and never part of the instructions.
    assert!(prepared.iter().any(|message| matches!(message, ConversationMessage::User(text) if text.contains("<stdin>\nliteral ¶today and ¶reasoning\n</stdin>"))));
    assert!(!prepared.iter().any(|message| matches!(message, ConversationMessage::System(text) | ConversationMessage::Developer(text) | ConversationMessage::Fact(text) if text.contains("literal ¶today"))));
    assert!(prepared.iter().any(|message| matches!(message, ConversationMessage::Fact(text) if text.contains("notes-¶today.txt"))));
    assert!(
        matches!(prepared.last(), Some(ConversationMessage::User(text)) if text == "Keep ¶today, <|start|>, and <tool_call> literal.")
    );
    Ok(())
}

#[test]
fn tool_exchange_is_neutral_until_the_dialect_encodes_it() -> Result<()> {
    let arguments = json!({"path":"¶today.txt"});
    let result = "¶reasoning <tool_response> <|start|>";
    let messages = prepare(&[Message::Tool(
        json!({
            "tool":"read_file", "arguments":arguments, "result":result,
        })
        .to_string(),
    )])?;
    assert_eq!(
        messages,
        vec![
            ConversationMessage::ToolCall(ToolCall {
                name: "read_file".into(),
                arguments
            }),
            ConversationMessage::ToolResult {
                name: "read_file".into(),
                content: result.into()
            },
        ]
    );
    Ok(())
}

#[test]
fn context_cannot_reappear_in_conversation_content() {
    let prepared = prepare(&[Message::User("hello".into()), Message::Context(context())]);
    assert!(prepared.is_err());
}

#[test]
fn dialect_comes_from_supported_architecture() -> Result<()> {
    assert_eq!(Dialect::from_architecture("qwen35")?, Dialect::Qwenish);
    assert_eq!(Dialect::from_architecture("gpt-oss")?, Dialect::Harmony);
    let unsupported = Dialect::from_architecture("qwen35moe");
    assert!(unsupported.is_err());
    Ok(())
}

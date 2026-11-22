use super::*;
use ConversationMessage::*;

#[test]
fn replays_parallel_calls_and_a_tool_only_step_as_the_checkpoint_does() -> Result<()> {
    let call = |name: &str, arguments| {
        ToolCall(super::ToolCall {
            name: name.to_owned(),
            arguments,
        })
    };
    let result = |name: &str, content: &str| ToolResult {
        name: name.to_owned(),
        content: content.to_owned(),
    };
    let messages = [
        Effort(crate::protocol::Effort::Low),
        Instructions("Static instructions.".into()),
        Fact("Current date: 2040-02-03".into()),
        User("  Run both.  ".into()),
        Thinking(" plan ".into()),
        call(
            "run_command",
            json!({"waitSeconds": 0.1, "argv": ["printf", "hello"]}),
        ),
        result("run_command", r#"{"status":"running","commandId":1}"#),
        call("read_file", json!({"path": "data.txt"})),
        result("read_file", "file content"),
        Thinking(String::new()),
        call("control_command", json!({"commandId": 1, "action": "wait"})),
        result("control_command", r#"{"status":"finished"}"#),
    ];
    let (pieces, preamble) = render(&messages)?;
    assert_eq!(
        &pieces[..3],
        &[
            Piece::Marker(SYSTEM),
            Piece::Text("tool_declare".into()),
            Piece::Marker("<|content_xml|>"),
        ]
    );
    let Piece::Text(tools) = &pieces[3] else {
        panic!("expected Inkling tool declarations");
    };
    let tools: Vec<Value> = serde_json::from_str(tools)?;
    assert!(tools.iter().any(|tool| {
        tool["name"] == "run_command"
            && tool["type"] == "function"
            && tool["parameters"]["properties"]["argv"]["type"] == "array"
    }));
    assert_eq!(
        pieces[preamble + 2],
        Piece::Text("Current date: 2040-02-03".into())
    );

    let transcript: String = pieces[5..]
        .iter()
        .map(|piece| match piece {
            Piece::Marker(marker) => *marker,
            Piece::Text(text) => text.as_str(),
        })
        .collect();
    // Captured from the pinned GGUF's `tokenizer.chat_template` with these messages
    // and no declarations.
    assert_eq!(
        transcript,
        concat!(
            "<|message_system|><|content_text|>Static instructions.<|end_message|>",
            "<|message_system|><|content_text|>Current date: 2040-02-03<|end_message|>",
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>",
            "<|message_user|><|content_text|>  Run both.  <|end_message|>",
            "<|message_model|><|content_thinking|> plan <|end_message|>",
            r#"<|message_model|>run_command<|content_invoke_tool_json|>{"name":"run_command","args":{"argv":["printf","hello"],"waitSeconds":0.1}}<|end_message|>"#,
            r#"<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"data.txt"}}<|end_message|>"#,
            "<|content_model_end_sampling|>",
            r#"<|message_tool|>run_command<|content_text|>{"status":"running","commandId":1}<|end_message|>"#,
            "<|message_tool|>read_file<|content_text|>file content<|end_message|>",
            r#"<|message_model|>control_command<|content_invoke_tool_json|>{"name":"control_command","args":{"action":"wait","commandId":1}}<|end_message|>"#,
            "<|content_model_end_sampling|>",
            r#"<|message_tool|>control_command<|content_text|>{"status":"finished"}<|end_message|>"#,
        )
    );
    Ok(())
}

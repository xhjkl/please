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
            json!({"wait_seconds": 0.1, "argv": ["printf", "hello"]}),
        ),
        result("run_command", r#"{"status":"running","command_id":1}"#),
        call("read_file", json!({"path": "data.txt"})),
        result("read_file", "file content"),
        Thinking(String::new()),
        call(
            "control_command",
            json!({"command_id": 1, "action": "wait"}),
        ),
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
            r#"<|message_model|>run_command<|content_invoke_tool_json|>{"name":"run_command","args":{"argv":["printf","hello"],"wait_seconds":0.1}}<|end_message|>"#,
            r#"<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"data.txt"}}<|end_message|>"#,
            "<|content_model_end_sampling|>",
            r#"<|message_tool|>run_command<|content_text|>{"status":"running","command_id":1}<|end_message|>"#,
            "<|message_tool|>read_file<|content_text|>file content<|end_message|>",
            r#"<|message_model|>control_command<|content_invoke_tool_json|>{"name":"control_command","args":{"action":"wait","command_id":1}}<|end_message|>"#,
            "<|content_model_end_sampling|>",
            r#"<|message_tool|>control_command<|content_text|>{"status":"finished"}<|end_message|>"#,
        )
    );
    Ok(())
}

/// Every control token the framing uses, spelled out as ordinary text.
const SPELLED: [&str; 9] = [SYSTEM, USER, MODEL, TOOL, TEXT, THINKING, CALL, END, STOP];

/// A user request, a call, and its result, all carrying `text`.
fn carrying(text: &str) -> Vec<ConversationMessage> {
    vec![
        User(text.into()),
        ToolCall(super::ToolCall {
            name: "read_file".into(),
            arguments: json!({ "path": text }),
        }),
        ToolResult {
            name: "read_file".into(),
            content: text.into(),
        },
    ]
}

#[test]
fn literal_markers_in_content_stay_text() -> Result<()> {
    let literal = format!(r#"{{"stdin":"{}"}}"#, SPELLED.concat());
    let markers = |text: &str| -> Result<usize> {
        let (pieces, _) = render(&carrying(text))?;
        let markers = pieces
            .iter()
            .filter(|piece| matches!(piece, Piece::Marker(_)));
        Ok(markers.count())
    };
    assert_eq!(markers(&literal)?, markers("x")?);
    let (pieces, _) = render(&carrying(&literal))?;
    assert!(pieces.contains(&Piece::Text(literal.clone())));
    Ok(())
}

/// The checkpoint's tokenizer must encode control spellings in content as text, and the parser
/// must read them back as text.
#[test]
#[ignore = "requires Metal and an Inkling checkpoint at INKLING_PATH"]
fn literal_markers_stay_text_on_the_checkpoint_tokenizer() -> Result<()> {
    use crate::hub::dialect::piece::special;
    let path = std::env::var_os("INKLING_PATH");
    let path = path.ok_or_else(|| eyre!("set INKLING_PATH to the checkpoint path"))?;
    let weights = oco::load(path).map_err(|error| eyre!(error))?;
    let tokenizer = weights.tokenizer();
    let ids = SPELLED
        .iter()
        .map(|spelling| special(tokenizer, spelling))
        .collect::<Result<Vec<_>>>()?;
    let controls = |tokens: &[u32]| tokens.iter().filter(|token| ids.contains(token)).count();
    let spelled = SPELLED.concat();
    assert_eq!(controls(&encode_text(tokenizer, &spelled)?), 0);
    let prompt = |text: &str| Adapter.encode(tokenizer, carrying(text), 8_192);
    let literal = format!(r#"{{"stdin":"{spelled}"}}"#);
    assert_eq!(
        controls(&prompt(&literal)?.tokens),
        controls(&prompt("x")?.tokens)
    );

    let mut parser = super::output::Parser::new(tokenizer)?;
    let mut tokens = vec![special(tokenizer, MODEL)?, special(tokenizer, TEXT)?];
    tokens.extend(encode_text(tokenizer, &spelled)?);
    tokens.push(special(tokenizer, END)?);
    let mut answer = String::new();
    for token in tokens {
        for delta in parser.push_token(token)? {
            match delta {
                crate::hub::dialect::Delta::Answer(text) => answer.push_str(&text),
                crate::hub::dialect::Delta::Thinking(_) => {}
            }
        }
    }
    assert_eq!(answer, spelled);
    assert!(parser.finish()?.is_empty());
    Ok(())
}

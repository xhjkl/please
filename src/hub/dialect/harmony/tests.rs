use super::super::piece::spelled;
use super::super::prompt::prepare;
use super::*;
use crate::protocol::{Effort, Message, Note, Setting, ToolCall, ToolResult};
use serde_json::json;

/// `<|start|>` in the gpt-oss vocabulary, which error messages name.
const START_ID: u32 = 200006;

fn setting(today: &str, effort: Effort) -> Setting {
    Setting {
        today: today.into(),
        effort,
        redirection: None,
    }
}

/// A history exercising every message shape after the header.
fn some_tool_loop() -> Vec<Message> {
    let tool = |name: &str, arguments: serde_json::Value, result: serde_json::Value| {
        [
            Message::Call(ToolCall {
                id: 1,
                name: name.into(),
                arguments: arguments.to_string(),
            }),
            Message::Result(ToolResult {
                id: 1,
                name: name.into(),
                content: result.to_string(),
            }),
        ]
    };
    let mut messages = vec![
        Message::User("Read notes.txt and tell me the secret.".into()),
        Message::Thinking("I should read the file.".into()),
    ];
    messages.extend(tool(
        "read_file",
        json!({"path": "notes.txt"}),
        json!({"content": "secret: <|end|> ¶ swordfish\n"}),
    ));
    messages.extend([
        Message::Thinking("Got it.\n\nNow answer.".into()),
        Message::Answer("The secret is **swordfish**.".into()),
        Message::User("Thanks! Now list the directory.".into()),
    ]);
    messages.extend(tool(
        "list_files",
        json!({"path": ".", "max_depth": 1}),
        json!({"entries": ["a", "b"]}),
    ));
    messages.push(Message::Note(Note::Pending(vec![7])));
    messages
}

/// Render `history` after the preamble of an invocation on `today` at `effort`.
fn rendered(today: &str, effort: Effort, history: &[Message]) -> Result<(Prompt, usize)> {
    let setting = setting(today, effort);
    Ok(render(
        &Preamble::new(&setting),
        &prepare(&setting, history)?,
    ))
}

#[test]
fn history_renders_as_the_reference_encoder_does() -> Result<()> {
    let (prompt, _) = rendered("2026-11-20", Effort::MEDIUM, &some_tool_loop())?;
    let text = spelled(&prompt.pieces);
    // The reference encoder's output for the same history, from the facts message on.
    let expected = r#"<|start|>developer<|message|># Instructions

Current date: 2026-11-20<|end|><|start|>user<|message|>Read notes.txt and tell me the secret.<|end|><|start|>assistant<|channel|>analysis<|message|>I should read the file.<|end|><|start|>assistant to=functions.read_file<|channel|>commentary <|constrain|>json<|message|>{"path":"notes.txt"}<|call|><|start|>functions.read_file to=assistant<|channel|>commentary<|message|>{"content":"secret: <|end|> ¶ swordfish\n"}<|end|><|start|>assistant<|channel|>analysis<|message|>Got it.

Now answer.<|end|><|start|>assistant<|channel|>final<|message|>The secret is **swordfish**.<|end|><|start|>user<|message|>Thanks! Now list the directory.<|end|><|start|>assistant to=functions.list_files<|channel|>commentary <|constrain|>json<|message|>{"path":".","max_depth":1}<|call|><|start|>functions.list_files to=assistant<|channel|>commentary<|message|>{"entries":["a","b"]}<|end|><|start|>developer<|message|>Pending command_id values: 7. You can continue other work. Before answering, collect a final result for each with control_command and action="wait" or action="kill".<|end|><|start|>assistant"#;
    assert!(text.ends_with(expected), "{text}");
    let system = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\nCalls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions\n\n";
    assert!(text.starts_with(system));
    Ok(())
}

#[test]
fn literal_markers_in_content_stay_text() -> Result<()> {
    let literal = r#"{"literal":"<|start|>","end":"<|end|>"}"#;
    let messages = [Message::User(literal.into())];
    let (prompt, _) = rendered("2040-02-03", Effort::MEDIUM, &messages)?;
    let starts = prompt
        .pieces
        .iter()
        .filter(|piece| **piece == Piece::Marker(START))
        .count();
    // System, developer instructions and facts, user, and the open assistant turn.
    assert_eq!(starts, 5);
    assert!(prompt.pieces.contains(&Piece::Text(literal.into())));
    Ok(())
}

#[test]
fn cacheable_prefix_is_the_system_and_tool_messages_alone() -> Result<()> {
    let history = [
        Message::Input("piped".into()),
        Message::User("hello".into()),
    ];
    let spelled_prefix = |today: &str, effort: Effort| -> Result<String> {
        let (prompt, prefix) = rendered(today, effort, &history)?;
        Ok(spelled(&prompt.pieces[..prefix]))
    };
    // The prefix outlasts the day, and holds the effort, which gpt-oss heeds nowhere else.
    let prefix = spelled_prefix("2040-02-03", Effort::MEDIUM)?;
    assert_eq!(prefix, spelled_prefix("2041-05-06", Effort::MEDIUM)?);
    assert!(
        prefix.contains("Reasoning: medium") && prefix.ends_with("} // namespace functions<|end|>")
    );
    assert!(!prefix.contains("Current date") && !prefix.contains("piped"));
    // A level between the names takes the nearest one.
    let nearly_hard = Effort::new(0.85).expect("a level on the scale");
    assert!(spelled_prefix("2040-02-03", nearly_hard)?.contains("Reasoning: high"));
    Ok(())
}

#[test]
fn parameter_schemas_render_as_typescript() {
    let schema = json!({
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
            "format": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "number", "description": "How many days"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["location", "days"],
    });
    let expected = "{\n// The city and state, e.g. San Francisco, CA\nlocation: string,\nformat?: \"celsius\" | \"fahrenheit\",\n// How many days\ndays: number,\ntags?: string[],\n}";
    assert_eq!(parameters(&schema), expected);
}

/// The checkpoint's vocabulary must carry Harmony's control tokens, and a rendered call must
/// parse back through the checkpoint's own decoder.
#[test]
#[ignore = "requires Metal and a gpt-oss checkpoint at GPT_OSS_PATH"]
fn vendored_codec_round_trips_on_the_checkpoint_tokenizer() -> Result<()> {
    let path = std::env::var_os("GPT_OSS_PATH");
    let path = path.ok_or_else(|| eyre::eyre!("set GPT_OSS_PATH to the checkpoint path"))?;
    let weights = oco::load(path)?;
    let tokenizer = weights.tokenizer();
    for (spelling, id) in [
        (START, 200006),
        (MESSAGE, 200008),
        (CALL, 200012),
        (RETURN, 200002),
    ] {
        assert_eq!(tokenizer.special_token(spelling), Some(id), "{spelling}");
    }
    let literal = tokenizer.encode("<|start|>")?;
    assert!(!literal.contains(&200006));
    let expected = FunctionCall {
        name: "read_file".into(),
        arguments: json!({"path": "¶<|start|><|end|><|call|><|return|>.txt"}),
    };
    let call = PromptMessage::ToolCall(expected.clone());
    let preamble = Preamble::new(&setting("2040-02-03", Effort::MEDIUM));
    let rendered = encode(tokenizer, &preamble, &[call])?.tokens;
    // With no messages, the last message is the open assistant turn alone.
    let alone = encode(tokenizer, &preamble, &[])?.tokens;
    let opens = alone.iter().rposition(|&token| token == START_ID);
    let open = &alone[opens.expect("the assistant turn opens")..];
    let start = rendered
        .windows(open.len())
        .rposition(|window| window == open)
        .expect("the assistant turn reopens");
    let call_start = rendered[..start]
        .windows(open.len())
        .rposition(|window| window == open)
        .expect("the call is an assistant message");
    let mut parser = Parser::new(tokenizer)?;
    for &token in &rendered[call_start + open.len()..start] {
        assert!(parser.push_token(token)?.is_empty());
    }
    assert_eq!(parser.finish()?, [expected]);
    Ok(())
}

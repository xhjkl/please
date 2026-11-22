use super::super::conversation::Delta;
use super::super::conversation::prepare;
use super::output::{Event, Marker, OutputStream};
use super::*;
use crate::protocol::{Call, Effort, Message, Note, Setting, ToolResult};
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

/// A conversation exercising every message shape after the header.
fn tool_loop() -> Vec<ConversationMessage> {
    let tool = |name: &str, arguments: serde_json::Value, result: serde_json::Value| {
        [
            Message::Call(Call {
                name: name.into(),
                arguments: arguments.to_string(),
            }),
            Message::Result(ToolResult {
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
    prepare(&setting("2026-11-20", Effort::Medium), &messages).unwrap()
}

#[test]
fn conversation_renders_as_the_reference_encoder_does() -> Result<()> {
    let (prompt, _) = render(&tool_loop());
    let text = prompt.spelled();
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
    let (prompt, _) = render(&prepare(&setting("2040-02-03", Effort::Medium), &messages)?);
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
fn static_head_is_the_system_and_tool_messages_alone() -> Result<()> {
    let history = |today: &str, effort: Effort| {
        let messages = [
            Message::Input("piped".into()),
            Message::User("hello".into()),
        ];
        prepare(&setting(today, effort), &messages)
    };
    let spelled_head = |messages: &[ConversationMessage]| {
        let (prompt, head) = render(messages);
        let pieces = Prompt {
            pieces: prompt.pieces[..head].to_vec(),
        };
        pieces.spelled()
    };
    let head = spelled_head(&history("2040-02-03", Effort::Medium)?);
    assert_eq!(head, spelled_head(&history("2041-05-06", Effort::Medium)?));
    assert_ne!(head, spelled_head(&history("2040-02-03", Effort::High)?));
    assert!(
        head.contains("Reasoning: medium") && head.ends_with("} // namespace functions<|end|>")
    );
    assert!(!head.contains("Current date") && !head.contains("piped"));
    // An empty history is only the open assistant turn, with nothing to cache.
    let (prompt, head) = render(&[]);
    assert_eq!((prompt.spelled().as_str(), head), ("<|start|>assistant", 0));
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

/// Feed a spelled-out continuation, splitting markers from text, into a fresh stream.
fn stream(spelled: &str) -> (OutputStream, Vec<Delta>, Option<String>) {
    let mut output = OutputStream::new(START_ID);
    let mut deltas = Vec::new();
    let mut rest = spelled;
    while !rest.is_empty() {
        let marker = Marker::ALL
            .into_iter()
            .find(|marker| rest.starts_with(marker.spelling()));
        let pushed = match marker {
            Some(marker) => {
                rest = &rest[marker.spelling().len()..];
                output.push(0, Event::Marker(marker))
            }
            None => {
                let length = rest.chars().next().map_or(1, char::len_utf8);
                let (text, after) = rest.split_at(length);
                rest = after;
                output.push(0, Event::Text(text))
            }
        };
        match pushed {
            Ok(pushed) => deltas.extend(pushed),
            Err(error) => return (output, deltas, Some(error.to_string())),
        }
    }
    (output, deltas, None)
}

fn joined(deltas: &[Delta]) -> (String, String) {
    let mut thinking = String::new();
    let mut answer = String::new();
    for delta in deltas {
        match delta {
            Delta::Thinking(text) => thinking.push_str(text),
            Delta::Answer(text) => answer.push_str(text),
        }
    }
    (thinking, answer)
}

#[test]
fn analysis_and_final_stream_as_thinking_and_answer() -> Result<()> {
    let (mut output, deltas, error) = stream(
        "<|channel|>analysis<|message|>first<|end|><|start|>assistant<|channel|>analysis<|message|>second<|end|><|start|>assistant<|channel|>final<|message|>Hi there<|return|>",
    );
    assert_eq!(error, None);
    // Separate analysis messages become paragraphs of one reasoning string.
    assert_eq!(
        joined(&deltas),
        ("first\n\nsecond".into(), "Hi there".into())
    );
    assert!(output.finish()?.is_empty());
    Ok(())
}

#[test]
fn tool_calls_parse_with_the_recipient_before_or_after_the_channel() -> Result<()> {
    for spelled in [
        r#"<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{"path":"¶"}<|call|>"#,
        r#" to=functions.read_file<|channel|>commentary <|constrain|>json<|message|>{"path":"¶"}<|call|>"#,
        r#"<|channel|>commentary to=functions.read_file json<|message|>{"path":"¶"}<|call|>"#,
        r#"<|channel|>analysis<|message|>look<|end|><|start|>assistant to=functions.read_file<|channel|>commentary<|message|>{"path":"¶"}<|call|>"#,
    ] {
        let (mut output, deltas, error) = stream(spelled);
        assert_eq!(error, None, "{spelled}");
        // Arguments are never shown as an answer.
        assert!(joined(&deltas).1.is_empty(), "{spelled}");
        let calls = output.finish()?;
        let expected = ToolCall {
            name: "read_file".into(),
            arguments: json!({"path": "¶"}),
        };
        assert_eq!(calls, [expected], "{spelled}");
    }
    Ok(())
}

#[test]
fn json_arguments_keep_literal_control_token_spellings() -> Result<()> {
    let (mut output, _, error) = stream("<|channel|>commentary to=functions.read_file<|message|>");
    assert_eq!(error, None);
    let path = "<|start|><|end|><|call|><|return|><|channel|><|message|><|constrain|>";
    let arguments = json!({"path": path});
    let text = arguments.to_string();
    assert!(output.push(0, Event::Text(&text))?.is_empty());
    assert!(output.push(0, Event::Marker(Marker::Call))?.is_empty());
    assert_eq!(
        output.finish()?,
        [ToolCall {
            name: "read_file".into(),
            arguments
        }]
    );
    Ok(())
}

#[test]
fn commentary_without_a_recipient_is_answer_text() -> Result<()> {
    let (mut output, deltas, error) = stream(
        "<|channel|>commentary<|message|>Looking first. <|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>",
    );
    assert_eq!(error, None);
    assert_eq!(joined(&deltas).1, "Looking first. Done.");
    assert!(output.finish()?.is_empty());
    Ok(())
}

#[test]
fn malformed_output_fails_or_is_dropped_like_the_reference() -> Result<()> {
    // Text after a message, where only a new message may start.
    let (_, deltas, error) = stream("<|channel|>final<|message|>answer<|end|>junk");
    assert_eq!(joined(&deltas).1, "answer");
    let expected = "Unexpected token 0 while expecting start token 200006";
    assert_eq!(error.as_deref(), Some(expected));
    // A first header that never reaches its content calls nothing.
    let (mut output, _, error) = stream("<|channel|>final stray words<|return|>");
    assert_eq!(error, None);
    assert!(output.finish()?.is_empty());
    // A response cut off inside a header.
    let (mut output, _, _) = stream("<|channel|>analysis");
    let message = output.finish().unwrap_err().to_string();
    assert_eq!(
        message,
        "Unexpected EOS while waiting for message header to complete"
    );
    // Arguments that are not JSON.
    let (mut output, _, _) =
        stream(r#"<|channel|>commentary to=functions.read_file<|message|>{"path": oops<|call|>"#);
    let message = output.finish().unwrap_err().to_string();
    assert!(message.starts_with(r#"error parsing tool call: raw='{"path": oops'"#));
    Ok(())
}

#[test]
fn a_response_cut_off_mid_answer_keeps_its_text() -> Result<()> {
    let (mut output, deltas, error) = stream(r#"<|channel|>final<|message|>cut"#);
    assert_eq!((joined(&deltas).1.as_str(), error), ("cut", None));
    assert!(output.finish()?.is_empty());
    Ok(())
}

/// The checkpoint's vocabulary must carry Harmony's control tokens, and a rendered call must
/// parse back through the checkpoint's own decoder.
#[test]
#[ignore = "requires Metal and a gpt-oss checkpoint at GPT_OSS_PATH"]
fn vendored_codec_round_trips_on_the_checkpoint_tokenizer() -> Result<()> {
    let path = std::env::var_os("GPT_OSS_PATH");
    let path = path.ok_or_else(|| eyre::eyre!("set GPT_OSS_PATH to the checkpoint path"))?;
    let weights = oco::load(path).map_err(|error| eyre::eyre!(error))?;
    let tokenizer = weights.tokenizer();
    for (spelling, id) in [
        (START, 200006),
        (MESSAGE, 200008),
        (CALL, 200012),
        (RETURN, 200002),
    ] {
        assert_eq!(tokenizer.special_token(spelling), Some(id), "{spelling}");
    }
    let literal = encode_text(tokenizer, "<|start|>")?;
    assert!(!literal.contains(&200006));
    let expected = ToolCall {
        name: "read_file".into(),
        arguments: json!({"path": "¶<|start|><|end|><|call|><|return|>.txt"}),
    };
    let call = ConversationMessage::ToolCall(expected.clone());
    let rendered = Adapter.encode(tokenizer, vec![call], 8_192)?.tokens;
    let open = Adapter.encode(tokenizer, vec![], 8_192)?.tokens;
    let start = rendered
        .windows(open.len())
        .rposition(|window| window == open)
        .expect("the assistant turn reopens");
    let call_start = rendered[..start]
        .windows(open.len())
        .rposition(|window| window == open)
        .expect("the call is an assistant message");
    let mut parser = Adapter.output_parser(tokenizer)?;
    for &token in &rendered[call_start + open.len()..start] {
        assert!(parser.push_token(token)?.is_empty());
    }
    assert_eq!(parser.finish()?, [expected]);
    Ok(())
}

#[test]
fn messages_the_assistant_did_not_author_neither_stream_nor_call() -> Result<()> {
    // An unknown role followed by more words is a tool's message, as in the reference.
    let (mut output, deltas, error) = stream(
        r#"<|channel|>analysis<|message|>a<|end|><|start|>functions.read_file to=assistant<|channel|>commentary<|message|>{"x":1}<|end|><|start|>assistant<|channel|>analysis<|message|>b<|return|>"#,
    );
    assert_eq!(error, None);
    // Addressed to the assistant, it stays unshown, yet it separates the two analyses.
    assert_eq!(joined(&deltas), ("a\n\nb".into(), String::new()));
    assert!(output.finish()?.is_empty());
    let (_, _, error) = stream("<|channel|>final<|end|><|start|>stranger<|message|>x");
    assert_eq!(error.as_deref(), Some("Unknown role: stranger"));
    Ok(())
}

#[test]
fn a_first_header_cut_off_by_a_stop_still_counts_as_a_message() -> Result<()> {
    let (mut output, deltas, error) = stream(
        "<|channel|>analysis stray<|end|><|start|>assistant<|channel|>analysis<|message|>next<|return|>",
    );
    assert_eq!(error, None);
    // No thinking streamed before, so nothing separates; the message was still counted.
    assert_eq!(joined(&deltas).0, "next");
    assert!(output.finish()?.is_empty());
    let (_, _, error) = stream("<|channel|> x<|end|>");
    let expected = "channel marker present but no channel value found in header";
    assert_eq!(error.as_deref(), Some(expected));
    Ok(())
}

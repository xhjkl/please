use super::*;
use serde_json::json;

/// `<|start|>` in the gpt-oss vocabulary, which error messages name.
const START_ID: u32 = 200006;

/// Feed a spelled-out continuation, splitting markers from text, into a fresh stream.
fn stream(spelled: &str) -> (Stream, Vec<Delta>, Option<String>) {
    let mut output = Stream::new(START_ID);
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
        let expected = FunctionCall {
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
        [FunctionCall {
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
    // The error names the `<|start|>` it expected instead.
    assert!(error.is_some_and(|error| error.contains(&START_ID.to_string())));
    // A first header that never reaches its content calls nothing.
    let (mut output, _, error) = stream("<|channel|>final stray words<|return|>");
    assert_eq!(error, None);
    assert!(output.finish()?.is_empty());
    // A response cut off inside a header.
    let (mut output, _, _) = stream("<|channel|>analysis");
    assert!(output.finish().is_err());
    // Arguments that are not JSON.
    let (mut output, _, _) =
        stream(r#"<|channel|>commentary to=functions.read_file<|message|>{"path": oops<|call|>"#);
    let message = output.finish().unwrap_err().to_string();
    assert!(message.contains(r#"{"path": oops"#));
    Ok(())
}

#[test]
fn a_response_cut_off_mid_answer_keeps_its_text() -> Result<()> {
    let (mut output, deltas, error) = stream(r#"<|channel|>final<|message|>cut"#);
    assert_eq!((joined(&deltas).1.as_str(), error), ("cut", None));
    assert!(output.finish()?.is_empty());
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
    assert!(error.is_some_and(|error| error.contains("stranger")));
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
    assert!(error.is_some_and(|error| error.contains("channel")));
    Ok(())
}

use super::*;

#[test]
fn streams_thinking_and_answer_and_collects_multiple_native_calls() -> Result<()> {
    let mut stream = Stream::default();
    stream.marker(Marker::Model);
    stream.marker(Marker::Thinking);
    let text = stream.text(" plan");
    assert_eq!(text, Some(Delta::Thinking(" plan".into())));
    stream.marker(Marker::End);
    stream.marker(Marker::Model);
    stream.marker(Marker::Text);
    let text = stream.text("Working <|end_message|>");
    assert_eq!(text, Some(Delta::Answer("Working <|end_message|>".into())));
    stream.marker(Marker::End);
    for path in ["one.txt", "two.txt"] {
        stream.marker(Marker::Model);
        stream.text("read_");
        stream.text("file");
        stream.marker(Marker::Invoke);
        let text = stream.text(r#"{"name":"read_file","args":{"path":""#);
        assert!(text.is_none());
        stream.text(path);
        stream.text(r#""}}"#);
        stream.marker(Marker::End);
    }
    let calls = stream.finish()?;
    let paths: Vec<_> = calls.iter().map(|call| &call.arguments["path"]).collect();
    assert_eq!(paths, ["one.txt", "two.txt"]);
    Ok(())
}

#[test]
fn fails_the_step_if_a_call_is_truncated_or_misframed() {
    for (payload, end) in [
        (r#"{"name":"read_file","args":{}}"#, Marker::Reset),
        (r#"{"name":"run_command","args":{}}"#, Marker::End),
        (r#"{"name":"read_file","args":[]}"#, Marker::End),
    ] {
        let mut stream = Stream::default();
        stream.marker(Marker::Model);
        stream.text("read_file");
        stream.marker(Marker::Invoke);
        stream.text(payload);
        stream.marker(end);
        let result = stream.finish();
        assert!(result.is_err(), "{payload}");
    }
}

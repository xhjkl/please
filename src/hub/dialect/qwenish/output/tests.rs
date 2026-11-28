use super::*;
use serde_json::json;

/// Feed generated output where `¦</think>¦`-style spans stand for the structure tokens the
/// model emits; everything else arrives one character per token, as plain decoded text.
fn stream(output: &str) -> Result<(String, String, Vec<FunctionCall>)> {
    let mut stream = Stream::default();
    let mut deltas = Vec::new();
    for (index, part) in output.split('¦').enumerate() {
        let event = match part {
            _ if index % 2 == 0 => None,
            "</think>" => Some(Event::ThinkEnd),
            "<tool_call>" => Some(Event::CallStart),
            "</tool_call>" => Some(Event::CallEnd),
            other => panic!("unknown structure token {other}"),
        };
        match event {
            Some(event) => deltas.extend(stream.push(event)),
            None => {
                for character in part.chars() {
                    deltas.extend(stream.push(Event::Text(&character.to_string())));
                }
            }
        }
    }
    deltas.extend(stream.push(Event::End));
    let (mut thinking, mut answer) = (String::new(), String::new());
    for delta in deltas {
        match delta {
            Delta::Thinking(text) => thinking.push_str(&text),
            Delta::Answer(text) => answer.push_str(&text),
        }
    }
    Ok((thinking, answer, stream.finish()?))
}

#[test]
fn native_calls_preserve_strings_and_decode_arrays() -> Result<()> {
    let call = parse_call(
        "<function=run_command>\n<parameter=argv>\n[\"printf\",\"hello ✓\"]\n</parameter>\n<parameter=wait_seconds>\n0\n</parameter>\n</function>",
    )?;
    assert_eq!(
        call.arguments,
        json!({"argv":["printf","hello ✓"],"wait_seconds":0})
    );
    let call = parse_call(
        "<function=apply_patch>\n<parameter=path>\n123\n</parameter>\n<parameter=patch>\n  text\n\n</parameter>\n</function>",
    )?;
    assert_eq!(call.arguments, json!({"path":"123","patch":"  text\n"}));
    Ok(())
}

#[test]
fn quoted_json_arguments_keep_tag_spellings_and_control_tokens() -> Result<()> {
    let output = r#"¦</think>¦¦<tool_call>¦<function=run_command><parameter=argv>["printf","\"</parameter><parameter=wait_seconds></function>¦</tool_call>¦ <tool_call>"]</parameter><parameter=wait_seconds>0</parameter></function>¦</tool_call>¦"#;
    let (_, answer, calls) = stream(output)?;
    assert!(answer.is_empty());
    assert_eq!(
        calls,
        [FunctionCall {
            name: "run_command".into(),
            arguments: json!({
                "argv": ["printf", "\"</parameter><parameter=wait_seconds></function></tool_call> <tool_call>"],
                "wait_seconds": 0,
            }),
        }]
    );
    Ok(())
}

#[test]
fn invalid_calls_never_execute() {
    for source in [
        "",
        "<function=unknown></function>",
        "<function=run_command></function>",
        "<function=run_command><parameter=argv>oops</parameter></function>",
        "<function=read_file><parameter=path>x</parameter><parameter=path>y</parameter></function>",
    ] {
        assert!(parse_call(source).is_err(), "{source}");
    }
}

#[test]
fn structure_tokens_split_thinking_answer_and_calls() -> Result<()> {
    let output = "Think ✓\n¦</think>¦\n\nDone.\n\n¦<tool_call>¦\n<function=read_file>\n<parameter=path>\ninput.txt\n</parameter>\n</function>\n¦</tool_call>¦";
    let (thinking, answer, calls) = stream(output)?;
    assert_eq!(thinking, "Think ✓");
    assert_eq!(answer, "Done.");
    assert_eq!(
        calls,
        vec![FunctionCall {
            name: "read_file".into(),
            arguments: json!({"path":"input.txt"})
        }]
    );
    Ok(())
}

/// Observed on Qwen3.8-27B: asked to quote the tags, it emits the structure tokens themselves.
#[test]
fn quoted_tags_in_the_answer_stay_content() -> Result<()> {
    let output = "Explain.¦</think>¦Calls look like `¦<tool_call>¦` … `¦</tool_call>¦`; thinking ends at `¦</think>¦`.";
    let (_, answer, calls) = stream(output)?;
    assert_eq!(
        answer,
        "Calls look like `<tool_call>` … `</tool_call>`; thinking ends at `</think>`."
    );
    assert!(calls.is_empty());
    let (_, answer, _) = stream("¦</think>¦Opened with `¦<tool_call>¦` and never closed.")?;
    assert_eq!(answer, "Opened with `<tool_call>` and never closed.");
    let (_, answer, calls) = stream("¦</think>¦¦<tool_call>¦")?;
    assert_eq!(answer, "<tool_call>");
    assert!(calls.is_empty());
    Ok(())
}

#[test]
fn spelled_out_markers_stay_content() -> Result<()> {
    let output = "Quote </think> here¦</think>¦Write <tool_call> and </tool_call> literally.";
    let (thinking, answer, calls) = stream(output)?;
    assert_eq!(thinking, "Quote </think> here");
    assert_eq!(answer, "Write <tool_call> and </tool_call> literally.");
    assert!(calls.is_empty());
    Ok(())
}

#[test]
fn inner_whitespace_survives_and_structural_whitespace_does_not() -> Result<()> {
    let (thinking, answer, calls) = stream("a\n\nb \n¦</think>¦\n\n  x\n\n  y \n")?;
    assert_eq!((thinking.as_str(), answer.as_str()), ("a\n\nb", "x\n\n  y"));
    assert!(calls.is_empty());
    Ok(())
}

#[test]
fn incomplete_call_and_closing_token_inside_a_patch() -> Result<()> {
    assert!(stream("¦</think>¦¦<tool_call>¦<function=read_file>").is_err());
    let output = "¦</think>¦¦<tool_call>¦<function=apply_patch><parameter=patch>\n¦</tool_call>¦\n</parameter></function>¦</tool_call>¦";
    let (_, _, calls) = stream(output)?;
    assert_eq!(calls[0].arguments, json!({"patch":"</tool_call>"}));
    Ok(())
}

#[test]
fn values_end_only_at_the_real_closing_structure() -> Result<()> {
    let patch = "<a>\n</parameter> <parameter=x> </function> </tool_call>\n</b>";
    let call = parse_call(&format!(
        "<function=apply_patch>\n<parameter=patch>\n{patch}\n</parameter>\n<parameter=path>\nindex.html\n</parameter>\n</function>"
    ))?;
    assert_eq!(
        call.arguments,
        json!({"patch": patch, "path": "index.html"})
    );
    let call = parse_call(&format!(
        "<function=apply_patch>\n<parameter=path>\nx.xml\n</parameter>\n<parameter=patch>\n{patch}\n</parameter>\n</function>"
    ))?;
    assert_eq!(call.arguments, json!({"path": "x.xml", "patch": patch}));
    Ok(())
}

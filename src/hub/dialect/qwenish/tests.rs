use super::output::{Event, OutputStream, parse_call};
use super::*;
use crate::hub::dialect::conversation::{Delta, prepare};
use crate::protocol::{Call, Effort, Message, Setting, ToolResult};
use serde_json::json;

/// Spell markers out, as the template's own string output would.
fn spelled(prompt: &Prompt) -> String {
    prompt
        .pieces
        .iter()
        .map(|piece| match piece {
            Piece::Text(text) => text.as_str(),
            Piece::Marker(spelling) => spelling,
        })
        .collect()
}

fn setting() -> Setting {
    Setting {
        today: "2040-02-03".into(),
        effort: Effort::Medium,
        redirection: None,
    }
}

/// Feed generated output where `¦</think>¦`-style spans stand for the structure tokens the
/// model emits; everything else arrives one character per token, as plain decoded text.
fn stream(output: &str) -> Result<(String, String, Vec<ToolCall>)> {
    let mut stream = OutputStream::default();
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
fn tool_step_renders_like_the_chat_template() -> Result<()> {
    let call = |path: &str| {
        [
            Message::Call(Call {
                name: "read_file".into(),
                arguments: json!({"path": path, "max_bytes": 10}).to_string(),
            }),
            Message::Result(ToolResult {
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
    let prompt = spelled(&render(&prepare(&setting(), &history)?)?);
    let (_, conversation) = prompt.split_once("<|im_end|>\n").unwrap();
    assert_eq!(
        conversation,
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
    let prompt = render(&prepare(&setting(), &history)?)?;
    let system = spelled(&prompt);
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
fn template_json_matches_python_separators() -> Result<()> {
    let value = json!({"argv": ["ls", "-la"], "nested": {"ok": true}});
    assert_eq!(
        template_json(&value)?,
        r#"{"argv": ["ls", "-la"], "nested": {"ok": true}}"#
    );
    Ok(())
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
        [ToolCall {
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
        vec![ToolCall {
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

#[test]
fn static_head_ends_where_session_facts_begin() -> Result<()> {
    let history = [Message::User("hi".into())];
    let prompt = render(&prepare(&setting(), &history)?)?;
    let (whole, bytes) = prompt.preamble.unwrap();
    let mut head = spelled(&Prompt {
        pieces: prompt.pieces[..whole].to_vec(),
        preamble: None,
    });
    let Some(Piece::Text(text)) = prompt.pieces.get(whole) else {
        panic!("facts continue the system text")
    };
    head.push_str(&text[..bytes]);
    assert!(head.ends_with("*** End Patch\n"));
    assert!(text[bytes..].starts_with("Current date: 2040-02-03"));
    assert!(spelled(&prompt).starts_with(&head));
    Ok(())
}

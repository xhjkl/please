use super::*;
use crate::protocol::{Effort, Frame, Message, Note, Place, Setting, ToolCall, ToolResult};
use eyre::Result;
use prompt::{PromptMessage, prepare};
use serde_json::json;

fn some_setting() -> Setting {
    Setting {
        today: "2040-02-03".into(),
        effort: Effort::High,
        redirection: Some("notes-¶today.txt".into()),
    }
}

/// The preamble `prepare` puts before any history: effort, instructions, and the date.
const PREAMBLE: usize = 4;

fn call(name: &str, arguments: serde_json::Value) -> Message {
    Message::Call(ToolCall {
        id: 1,
        name: name.into(),
        arguments: arguments.to_string(),
    })
}

fn result(name: &str, content: serde_json::Value) -> Message {
    Message::Result(ToolResult {
        id: 1,
        name: name.into(),
        content: content.to_string(),
    })
}

#[test]
fn the_hub_words_the_setting_and_keeps_content_literal() -> Result<()> {
    let frame = Frame::Advance {
        place: Place {
            machine: "ssh-ed25519 AAAA".into(),
            cwd: "/srv".into(),
        },
        setting: some_setting(),
        id: 9,
        messages: vec![
            Message::Input("literal ¶today and ¶reasoning".into()),
            Message::User("Keep ¶today, <|start|>, and <tool_call> literal.".into()),
        ],
    };
    let bytes = postcard::to_allocvec(&frame)?;
    let Frame::Advance {
        setting, messages, ..
    } = postcard::from_bytes(&bytes)?
    else {
        unreachable!()
    };
    let prepared = prepare(&setting, &messages)?;
    assert_eq!(prepared[0], PromptMessage::Effort(Effort::High));
    let PromptMessage::Instructions(system) = &prepared[1] else {
        panic!("instructions open the conversation")
    };
    assert!(!system.contains('¶'));
    // Everything that varies between invocations follows the standing instructions.
    let first_fact = prepared
        .iter()
        .position(|message| matches!(message, PromptMessage::Fact(_)))
        .unwrap();
    assert!(prepared[..first_fact].iter().all(|message| matches!(
        message,
        PromptMessage::Effort(_) | PromptMessage::Instructions(_)
    )));
    assert_eq!(
        prepared[first_fact],
        PromptMessage::Fact("Current date: 2040-02-03".into())
    );
    // Piped stdin is user data in a message of its own, announced, and never instructions.
    assert_eq!(
        prepared[PREAMBLE],
        PromptMessage::Note(prompting::INPUT.into())
    );
    assert_eq!(
        prepared[PREAMBLE + 1],
        PromptMessage::User("literal ¶today and ¶reasoning".into())
    );
    assert_eq!(
        prepared[PREAMBLE + 2],
        PromptMessage::User("Keep ¶today, <|start|>, and <tool_call> literal.".into())
    );
    // Where the answer goes comes last, so another redirect keeps everything before it.
    assert!(
        matches!(prepared.last(), Some(PromptMessage::Note(text)) if text.contains("notes-¶today.txt"))
    );
    Ok(())
}

#[test]
fn tool_exchange_is_neutral_until_the_dialect_encodes_it() -> Result<()> {
    let arguments = json!({"path":"¶today.txt"});
    let text = "¶reasoning <tool_response> <|start|>";
    let setting = Setting {
        redirection: None,
        ..some_setting()
    };
    let messages = prepare(
        &setting,
        &[
            call("read_file", arguments.clone()),
            Message::Result(ToolResult {
                id: 1,
                name: "read_file".into(),
                content: text.into(),
            }),
        ],
    )?;
    assert_eq!(
        messages[PREAMBLE..],
        [
            PromptMessage::ToolCall(FunctionCall {
                name: "read_file".into(),
                arguments
            }),
            PromptMessage::ToolResult {
                name: "read_file".into(),
                content: text.into()
            },
        ]
    );
    Ok(())
}

#[test]
fn notes_are_worded_by_the_hub() -> Result<()> {
    let setting = Setting {
        redirection: None,
        ..some_setting()
    };
    let restart = Note::Summarize {
        draft: Some("You asked me to find a GC.".into()),
        chunk: "[7] user\nfind it".into(),
    };
    let messages = prepare(
        &setting,
        &[
            Message::Note(Note::Pending(vec![3, 4])),
            Message::Note(Note::AnswerSetAside(vec![3])),
            Message::Note(restart),
        ],
    )?;
    let [
        PromptMessage::Note(pending),
        PromptMessage::Note(aside),
        PromptMessage::Note(restart),
        PromptMessage::Answer(draft),
        PromptMessage::User(transcript),
    ] = &messages[PREAMBLE..]
    else {
        panic!("notes stay notes: {messages:?}");
    };
    assert!(pending.starts_with("Pending command_id values: 3, 4."));
    assert!(aside.starts_with("Previous answer ignored. Pending command_id values: 3."));
    // The draft is the model's own message, and transcript data the user's material: neither
    // speaks with the harness's authority.
    assert_eq!(draft, "You asked me to find a GC.");
    assert!(!restart.contains("find a GC") && !restart.contains("find it"));
    assert_eq!(transcript, "[7] user\nfind it");
    Ok(())
}

#[test]
fn stored_calls_need_json_arguments() {
    let broken = Message::Call(ToolCall {
        id: 1,
        name: "read_file".into(),
        arguments: "{path:".into(),
    });
    assert!(prepare(&some_setting(), &[broken]).is_err());
}

#[test]
fn parallel_command_results_share_the_available_context() -> Result<()> {
    let mut history = vec![Message::User("run both".into())];
    for (id, name) in [(1, "run_command"), (2, "control_command")] {
        history.push(call(name, json!({})));
        history.push(result(
            name,
            json!({ "command_id": id, "stdout": format!("{}end {id}", "x".repeat(900)) }),
        ));
    }

    let mut messages = prepare(&some_setting(), &history)?;
    prompt::fit_command_results(&mut messages, 1024, |content| Ok(content.len()))?;

    let mut size = 0;
    for message in messages {
        if let PromptMessage::ToolResult { content, .. } = message {
            size += content.len();
            let result: serde_json::Value = serde_json::from_str(&content)?;
            let id = result["command_id"].as_u64().unwrap();
            let stdout = result["stdout"].as_str().unwrap();
            assert!(stdout.ends_with(&format!("end {id}")));
            assert_eq!(result["stdout_bytes_omitted"], 905 - stdout.len());
        }
    }
    assert!(size <= 512);
    Ok(())
}

#[test]
fn a_following_tool_only_step_keeps_earlier_output_budget() -> Result<()> {
    let command = |id| {
        [
            call("run_command", json!({ "argv": ["printf", "output"] })),
            result(
                "run_command",
                json!({ "command_id": id, "stdout": "x".repeat(900) }),
            ),
        ]
    };
    let setting = Setting {
        redirection: None,
        ..some_setting()
    };
    let mut history = vec![
        Message::User("Run both in sequence.".into()),
        Message::Thinking(String::new()),
    ];
    history.extend(command(1));
    let mut before = prepare(&setting, &history)?;
    prompt::fit_command_results(&mut before, 1024, |content| Ok(content.len()))?;

    history.push(Message::Thinking(String::new()));
    history.extend(command(2));
    let mut after = prepare(&setting, &history)?;
    prompt::fit_command_results(&mut after, 1024, |content| Ok(content.len()))?;

    assert_eq!(&after[..before.len()], before);
    Ok(())
}

#[test]
fn dialect_comes_from_supported_architecture() -> Result<()> {
    assert_eq!(Dialect::from_architecture("qwen35")?, Dialect::Qwenish);
    assert_eq!(Dialect::from_architecture("gpt-oss")?, Dialect::Harmony);
    let unsupported = Dialect::from_architecture("qwen35moe");
    assert!(unsupported.is_err());
    Ok(())
}

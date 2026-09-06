use super::*;
use openai_harmony::chat::{Content, Role};
use openai_harmony::{HarmonyEncodingName, ParseOptions, load_harmony_encoding};
use serde_json::{Value, json};

#[test]
fn command_tail_keeps_metadata_and_framing_after_context_clipping() -> Result<()> {
    let harmony = HarmonyAdapter::gpt_oss()?;
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
        .map_err(|error| eyre!(error.to_string()))?;
    let stdout_marker = "LATEST STDOUT: ✓\n";
    let stderr_marker = "LATEST STDERR: ошибка\n";
    let stdout = (0..4_000)
        .map(|line| format!("src/module.rs:{line}: warning: unused variable `value`\n"))
        .collect::<String>()
        + stdout_marker;
    let stderr = "error: \"ошибка\" at src/module.rs:42\t\n".repeat(5_000) + stderr_marker;

    for (name, stdout, stderr) in [
        ("run_command", stdout.as_str(), stderr.as_str()),
        ("control_command", stdout.as_str(), stderr.as_str()),
        ("run_command", stdout.as_str(), stderr_marker),
        ("control_command", stdout_marker, stderr.as_str()),
    ] {
        let arguments = match name {
            "run_command" => json!({ "argv": ["sh", "-c", "make"], "waitSeconds": 40 }),
            _ => json!({ "pid": 123, "action": "wait" }),
        };
        for ctx_cap in [8_192, 32_768] {
            let history = [
                Message::System("Supervise commands using their output and runtime.".into()),
                Message::User("Previous discussion. ".repeat(20_000)),
                Message::Tool(
                    json!({
                        "tool": name,
                        "arguments": arguments,
                        "result": {
                            "status": "running",
                            "runningFor": "123.4s",
                            "pid": 123,
                            "next": "Wait longer or kill the command.",
                            "stdout": stdout,
                            "stdoutBytesOmitted": 17,
                            "stderr": stderr,
                            "stderrBytesOmitted": 23,
                        }
                    })
                    .to_string(),
                ),
                Message::Developer("Command pid 123 is still running; wait or kill.".into()),
            ];
            let original = harmony.render_protocol_tokens(&history)?;
            assert!(original.len() > ctx_cap);
            let preamble_len = compute_preamble_len(&harmony, &history, ctx_cap)?;
            let tokens = harmony
                .render_protocol_tokens_with_tail_budget(&history, ctx_cap - 1 - preamble_len)?;
            let mut tokens = clip_to_ctx(tokens, preamble_len, ctx_cap);
            assert!(tokens.len() < ctx_cap);
            let completion_prefix = harmony.render_completion_tokens(&[])?;
            assert!(tokens.ends_with(&completion_prefix));
            tokens.truncate(tokens.len() - completion_prefix.len());

            // Parsing the real encoded/clipped conversation verifies that
            // retained JSON is still enclosed in a complete tool message.
            let messages = encoding
                .parse_messages_from_completion_tokens_with_options(
                    tokens,
                    None,
                    ParseOptions { strict: false },
                )
                .map_err(|error| eyre!(error.to_string()))?;
            let command = messages
                .iter()
                .find(|message| {
                    message.author.role == Role::Tool
                        && message.author.name.as_deref() == Some(name)
                })
                .expect("latest command result survives as a tool message");
            let [Content::Text(content)] = command.content.as_slice() else {
                panic!("command result has one text body");
            };
            let result: Value = serde_json::from_str(&content.text)?;
            assert_eq!(result["status"], "running");
            assert_eq!(result["runningFor"], "123.4s");
            assert_eq!(result["pid"], 123);
            assert_eq!(result["next"], "Wait longer or kill the command.");
            for (stream, original, omitted, marker) in [
                ("stdout", stdout, 17, stdout_marker),
                ("stderr", stderr, 23, stderr_marker),
            ] {
                let output = result[stream].as_str().unwrap();
                assert!(output.ends_with(marker));
                assert!(original.ends_with(output));
                if original == marker {
                    assert_eq!(output, original);
                } else {
                    assert!(output.len() < original.len());
                }
                assert_eq!(
                    result[format!("{stream}BytesOmitted")],
                    original.len() - output.len() + omitted,
                );
            }
        }
    }
    Ok(())
}

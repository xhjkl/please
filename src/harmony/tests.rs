use super::*;

const START_TOKEN: u32 = 200006;

#[test]
fn literal_start_marker_in_user_json_is_content_not_structure() -> Result<()> {
    let harmony = HarmonyAdapter::gpt_oss()?;
    let messages = [HarmonyMessage::User(
        r#"{"literal":"<|start|>","nested":{"still":"content"}}"#.to_string(),
    )];

    let tokens = harmony.render_completion_tokens(&messages)?;
    assert_eq!(count_token(&tokens, START_TOKEN), 2);
    assert!(
        !harmony
            .encoding
            .tokenizer()
            .encode_ordinary("<|start|>")
            .contains(&START_TOKEN)
    );
    Ok(())
}

#[test]
fn literal_start_marker_in_tool_call_json_is_content_not_structure() -> Result<()> {
    let harmony = HarmonyAdapter::gpt_oss()?;
    let messages = [HarmonyMessage::AssistantToolCall {
        recipient: "functions.echo".to_string(),
        arguments_json: r#"{"argument":"<|start|>"}"#.to_string(),
    }];

    let tokens = harmony.render_completion_tokens(&messages)?;
    assert_eq!(count_token(&tokens, START_TOKEN), 2);
    Ok(())
}

#[test]
fn small_command_result_is_unchanged_by_context_budget() -> Result<()> {
    let harmony = HarmonyAdapter::gpt_oss()?;
    let history = [Message::Tool(
        serde_json::json!({
            "tool": "run_command",
            "arguments": { "argv": ["printf", "hello"] },
            "result": {
                "status": "finished", "exitCode": 0, "runningFor": "0.1s",
                "stdout": "hello", "stdoutBytesOmitted": 0,
                "stderr": "", "stderrBytesOmitted": 0,
            }
        })
        .to_string(),
    )];
    let original = harmony.render_protocol_tokens(&history)?;
    let budgeted = harmony.render_protocol_tokens_with_tail_budget(&history, 8_192)?;
    assert_eq!(original, budgeted);
    Ok(())
}

fn count_token(tokens: &[u32], needle: u32) -> usize {
    tokens.iter().filter(|token| **token == needle).count()
}

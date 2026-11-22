use super::*;
use crate::protocol::{ProtocolError, write_frame_to_stream};
use std::time::Duration;

#[tokio::test]
async fn completed_subturn_resets_budget_before_later_disconnect() {
    let display = crate::display::make_display();
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let hub = tokio::spawn(async move {
        let mut store = Vec::new();
        let request: Frame = read_frame_from_stream(&mut server, &mut store, None, None)
            .await
            .unwrap();
        assert!(matches!(request, Frame::Request { .. }));
        write_frame_to_stream(
            &mut server,
            &Frame::ToolCall {
                name: "list_files".into(),
                arguments_json: serde_json::json!({ "path": ".", "max_depth": 0 }).to_string(),
            },
        )
        .await
        .unwrap();
        write_frame_to_stream(&mut server, &Frame::Stop)
            .await
            .unwrap();

        let request: Frame = read_frame_from_stream(&mut server, &mut store, None, None)
            .await
            .unwrap();
        let Frame::Request { messages } = request else {
            panic!("tool result is sent back in the next request");
        };
        assert!(
            matches!(&messages[messages.len() - 2], Message::Reasoning(text) if text.is_empty())
        );
        let Some(Message::Tool(payload)) = messages.last() else {
            panic!("completed tool exchange preserves the result in history");
        };
        let payload: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(payload["tool"], "list_files");
        assert!(payload["result"].is_array());
    });
    let mut messages = vec![Message::User("hello".into())];
    let mut reconnect_attempts = 6;
    let commands = Commands::default();
    let attempt = attempt_turn_on_stream(
        &mut messages,
        &commands,
        &mut client,
        &mut reconnect_attempts,
        &display,
    );
    let result = tokio::time::timeout(Duration::from_secs(5), attempt)
        .await
        .expect("completed tool exchange reaches the disconnect");
    let error = result.unwrap_err();
    assert!(matches!(
        error.downcast_ref::<ProtocolError>(),
        Some(ProtocolError::Disconnect)
    ));
    assert_eq!(reconnect_attempts, 0);
    hub.await.unwrap();
}

#[tokio::test]
async fn incomplete_subturn_preserves_reconnect_budget() {
    let display = crate::display::make_display();
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let hub = tokio::spawn(async move {
        let mut store = Vec::new();
        let request: Frame = read_frame_from_stream(&mut server, &mut store, None, None)
            .await
            .unwrap();
        assert!(matches!(request, Frame::Request { .. }));
        write_frame_to_stream(&mut server, &Frame::Thinking("partial response".into()))
            .await
            .unwrap();
    });
    let mut messages = vec![Message::User("hello".into())];
    let mut reconnect_attempts = 6;
    let commands = Commands::default();
    let attempt = attempt_turn_on_stream(
        &mut messages,
        &commands,
        &mut client,
        &mut reconnect_attempts,
        &display,
    );
    let result = tokio::time::timeout(Duration::from_secs(5), attempt)
        .await
        .expect("partial response reaches the disconnect");
    let error = result.unwrap_err();
    assert!(matches!(
        error.downcast_ref::<ProtocolError>(),
        Some(ProtocolError::Disconnect)
    ));
    assert_eq!(reconnect_attempts, 6);
    hub.await.unwrap();
}

#[tokio::test]
async fn pending_command_allows_other_tools_and_requires_collection_before_answer() -> Result<()> {
    let commands = Commands::default();
    let start = crate::tools::invoke(
        &all_tools(),
        ToolContext::new(&commands, None),
        "run_command",
        json!({ "argv": ["sleep", "999"], "waitSeconds": 0 }),
    )
    .await
    .unwrap();
    assert_eq!(start["status"], "running");
    let id = start["commandId"].as_u64().unwrap();

    let (mut client, mut server) = UnixStream::pair()?;
    let hub = tokio::spawn(async move {
        let mut store = Vec::new();
        let _: Frame = read_frame_from_stream(&mut server, &mut store, None, None).await?;
        write_frame_to_stream(
            &mut server,
            &Frame::Thinking("the job can run in the background".into()),
        )
        .await?;
        write_frame_to_stream(
            &mut server,
            &Frame::Answer("I'll list the files while it runs.".into()),
        )
        .await?;
        write_frame_to_stream(
            &mut server,
            &Frame::ToolCall {
                name: "list_files".into(),
                arguments_json: json!({ "path": ".", "max_depth": 0 }).to_string(),
            },
        )
        .await?;
        write_frame_to_stream(&mut server, &Frame::Stop).await?;

        let request = read_frame_from_stream(&mut server, &mut store, None, None).await?;
        let Frame::Request { messages } = request else {
            panic!("expected a follow-up request")
        };
        assert!(
            messages
                .iter()
                .any(|message| matches!(message, Message::Reasoning(text)
            if text == "the job can run in the background"))
        );
        assert!(
            messages
                .iter()
                .any(|message| matches!(message, Message::Assistant(text)
            if text == "I'll list the files while it runs."))
        );
        assert!(
            messages
                .iter()
                .any(|message| matches!(message, Message::Tool(payload)
            if serde_json::from_str::<Value>(payload).unwrap()["result"].is_array()))
        );
        write_frame_to_stream(&mut server, &Frame::Answer("too early".into())).await?;
        write_frame_to_stream(&mut server, &Frame::Stop).await?;

        let request = read_frame_from_stream(&mut server, &mut store, None, None).await?;
        let Frame::Request { messages } = request else {
            panic!("expected a follow-up request")
        };
        assert!(
            !messages
                .iter()
                .any(|message| matches!(message, Message::Assistant(text) if text == "too early"))
        );
        write_frame_to_stream(
            &mut server,
            &Frame::ToolCall {
                name: "control_command".into(),
                arguments_json: json!({ "commandId": id, "action": "kill" }).to_string(),
            },
        )
        .await?;
        write_frame_to_stream(&mut server, &Frame::Stop).await?;

        let request = read_frame_from_stream(&mut server, &mut store, None, None).await?;
        let Frame::Request { messages } = request else {
            panic!("expected a follow-up request")
        };
        let Some(Message::Tool(payload)) = messages.last() else {
            panic!("missing stop result")
        };
        let payload: Value = serde_json::from_str(payload)?;
        assert_eq!(payload["result"]["status"], "killed");
        write_frame_to_stream(&mut server, &Frame::Answer("settled".into())).await?;
        write_frame_to_stream(&mut server, &Frame::Stop).await
    });

    let mut messages = vec![Message::User("start a command".into())];
    let mut reconnect_attempts = 0;
    let display = crate::display::make_display();
    let attempt = attempt_turn_on_stream(
        &mut messages,
        &commands,
        &mut client,
        &mut reconnect_attempts,
        &display,
    );
    let result = tokio::time::timeout(Duration::from_secs(6), attempt).await??;
    hub.await??;
    assert_eq!(result, "settled");
    Ok(())
}

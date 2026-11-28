use super::*;
use crate::protocol::{Effort, ProtocolError, write_frame};
use std::time::Duration;

fn some_place() -> Place {
    Place {
        machine: "ssh-ed25519 AAAA".into(),
        cwd: "/".into(),
    }
}

fn some_setting() -> Setting {
    Setting {
        today: "2040-02-03".into(),
        effort: Effort::MEDIUM,
        redirection: None,
    }
}

/// The messages the probe delivers next.
async fn next_delivery(server: &mut UnixStream, store: &mut Vec<u8>) -> Result<Vec<Message>> {
    match read_frame(server, store, None, None).await? {
        Frame::Advance { messages, .. } => Ok(messages),
        frame => Err(eyre!("expected a request, got {frame:?}")),
    }
}

fn result_of(messages: &[Message], tool: &str) -> Option<Value> {
    messages.iter().find_map(|message| match message {
        Message::Result(ToolResult { name, content, .. }) if name == tool => {
            serde_json::from_str(content).ok()
        }
        _ => None,
    })
}

#[tokio::test]
async fn completed_step_resets_budget_before_later_disconnect() {
    let display = Display::new(false);
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let hub = tokio::spawn(async move {
        let mut store = Vec::new();
        let first = next_delivery(&mut server, &mut store).await.unwrap();
        assert_eq!(first, [Message::User("hello".into())]);
        write_frame(
            &mut server,
            &Frame::Call(ToolCall {
                id: 2,
                name: "list_files".into(),
                arguments: serde_json::json!({ "path": ".", "max_depth": 0 }).to_string(),
            }),
        )
        .await
        .unwrap();
        write_frame(&mut server, &Frame::Stop).await.unwrap();

        // The hub keeps the conversation, so the probe sends only the new result.
        let messages = next_delivery(&mut server, &mut store).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert!(result_of(&messages, "list_files").unwrap().is_array());
    });
    let (place, setting) = (some_place(), some_setting());
    let scene = Scene {
        place: &place,
        setting: &setting,
    };
    let mut delivery = Delivery::new(vec![Message::User("hello".into())]);
    let mut reconnect_attempts = 6;
    let jobs = Jobs::default();
    let attempt = attempt_turn(
        scene,
        &mut delivery,
        &jobs,
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
    // A retry after reconnecting resends the result the hub never answered, under one ID.
    assert!(result_of(&delivery.messages, "list_files").is_some());
    hub.await.unwrap();
}

#[tokio::test]
async fn results_answer_their_calls_by_id_in_call_order() {
    let display = Display::new(false);
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let hub = tokio::spawn(async move {
        let mut store = Vec::new();
        next_delivery(&mut server, &mut store).await.unwrap();
        let listing = json!({ "path": ".", "max_depth": 0 }).to_string();
        let calls = [(5, "list_files", listing), (6, "read_file", "[]".into())];
        for (id, name, arguments) in calls {
            let name = name.into();
            let call = Frame::Call(ToolCall {
                id,
                name,
                arguments,
            });
            write_frame(&mut server, &call).await.unwrap();
        }
        write_frame(&mut server, &Frame::Stop).await.unwrap();
        // The refused call's result still follows the one before it.
        let messages = next_delivery(&mut server, &mut store).await.unwrap();
        let answered: Vec<_> = messages
            .iter()
            .map(|message| match message {
                Message::Result(result) => (result.id, result.name.as_str()),
                _ => (0, ""),
            })
            .collect();
        assert_eq!(answered, [(5, "list_files"), (6, "read_file")]);
        assert!(result_of(&messages, "read_file").unwrap()["error"].is_string());
    });
    let (place, setting) = (some_place(), some_setting());
    let scene = Scene {
        place: &place,
        setting: &setting,
    };
    let mut delivery = Delivery::new(vec![Message::User("hello".into())]);
    let mut reconnect_attempts = 0;
    let jobs = Jobs::default();
    let attempt = attempt_turn(
        scene,
        &mut delivery,
        &jobs,
        &mut client,
        &mut reconnect_attempts,
        &display,
    );
    let result = tokio::time::timeout(Duration::from_secs(5), attempt).await;
    assert!(result.expect("the hub hears the results").is_err());
    hub.await.unwrap();
}

#[tokio::test]
async fn incomplete_step_preserves_reconnect_budget_and_request() {
    let display = Display::new(false);
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let hub = tokio::spawn(async move {
        let mut store = Vec::new();
        next_delivery(&mut server, &mut store).await.unwrap();
        write_frame(&mut server, &Frame::Thinking("partial response".into()))
            .await
            .unwrap();
    });
    let (place, setting) = (some_place(), some_setting());
    let scene = Scene {
        place: &place,
        setting: &setting,
    };
    let mut delivery = Delivery::new(vec![Message::User("hello".into())]);
    let mut reconnect_attempts = 6;
    let jobs = Jobs::default();
    let attempt = attempt_turn(
        scene,
        &mut delivery,
        &jobs,
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
    assert_eq!(delivery.messages, [Message::User("hello".into())]);
    hub.await.unwrap();
}

#[tokio::test]
async fn pending_command_allows_other_tools_and_requires_collection_before_answer() -> Result<()> {
    let jobs = Jobs::default();
    let start = crate::tools::invoke(
        &all_tools(),
        ToolContext::new(&jobs, None),
        "run_command",
        json!({ "argv": ["sleep", "999"], "wait_seconds": 0 }),
    )
    .await
    .unwrap();
    assert_eq!(start["status"], "running");
    let id = start["command_id"].as_u64().unwrap();

    let (mut client, mut server) = UnixStream::pair()?;
    let hub = tokio::spawn(async move {
        let mut store = Vec::new();
        let first = next_delivery(&mut server, &mut store).await?;
        assert!(first.contains(&Message::Note(Note::Pending(vec![id]))));
        write_frame(
            &mut server,
            &Frame::Thinking("the job can run in the background".into()),
        )
        .await?;
        write_frame(
            &mut server,
            &Frame::Answer("I'll list the files while it runs.".into()),
        )
        .await?;
        write_frame(
            &mut server,
            &Frame::Call(ToolCall {
                id: 3,
                name: "list_files".into(),
                arguments: json!({ "path": ".", "max_depth": 0 }).to_string(),
            }),
        )
        .await?;
        write_frame(&mut server, &Frame::Stop).await?;

        let messages = next_delivery(&mut server, &mut store).await?;
        assert!(result_of(&messages, "list_files").unwrap().is_array());
        assert_eq!(
            messages.last(),
            Some(&Message::Note(Note::Pending(vec![id])))
        );
        write_frame(&mut server, &Frame::Answer("too early".into())).await?;
        write_frame(&mut server, &Frame::Stop).await?;

        // An answer while the command runs is set aside, and the hub hears why.
        let messages = next_delivery(&mut server, &mut store).await?;
        assert_eq!(messages, [Message::Note(Note::AnswerSetAside(vec![id]))]);
        write_frame(
            &mut server,
            &Frame::Call(ToolCall {
                id: 6,
                name: "control_command".into(),
                arguments: json!({ "command_id": id, "action": "kill" }).to_string(),
            }),
        )
        .await?;
        write_frame(&mut server, &Frame::Stop).await?;

        let messages = next_delivery(&mut server, &mut store).await?;
        let killed = result_of(&messages, "control_command").unwrap();
        assert_eq!(killed["status"], "killed");
        assert!(
            !messages
                .iter()
                .any(|message| matches!(message, Message::Note(_)))
        );
        write_frame(&mut server, &Frame::Answer("settled".into())).await?;
        write_frame(&mut server, &Frame::Stop).await
    });

    let (place, setting) = (some_place(), some_setting());
    let scene = Scene {
        place: &place,
        setting: &setting,
    };
    let mut delivery = Delivery::new(vec![Message::User("start a command".into())]);
    let mut reconnect_attempts = 0;
    let display = Display::new(false);
    let attempt = attempt_turn(
        scene,
        &mut delivery,
        &jobs,
        &mut client,
        &mut reconnect_attempts,
        &display,
    );
    let result = tokio::time::timeout(Duration::from_secs(6), attempt).await??;
    hub.await??;
    assert_eq!(result, "settled");
    Ok(())
}

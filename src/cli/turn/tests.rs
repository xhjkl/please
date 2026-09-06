use super::*;
use crate::protocol::{ProtocolError, write_frame_to_stream};
use std::time::Duration;

#[tokio::test]
async fn completed_subturn_resets_budget_before_later_disconnect() {
    let display = crate::display::make_display();
    let display = Arc::new(display);
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
        let Some(Message::Tool(payload)) = messages.last() else {
            panic!("completed tool exchange preserves the result in history");
        };
        let payload: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(payload["tool"], "list_files");
        assert!(payload["result"].is_array());
    });
    let mut messages = vec![Message::User("hello".into())];
    let mut reconnect_attempts = 6;
    let context = ToolContext::default();
    let attempt = attempt_turn_on_stream(
        &mut messages,
        context,
        &mut client,
        &mut reconnect_attempts,
        display,
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
    let display = Arc::new(display);
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
    let context = ToolContext::default();
    let attempt = attempt_turn_on_stream(
        &mut messages,
        context,
        &mut client,
        &mut reconnect_attempts,
        display,
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

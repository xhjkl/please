use super::*;
use tokio::io::AsyncWriteExt;

#[tokio::test(start_paused = true)]
async fn established_connection_survives_long_tool_waits() {
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let reader = tokio::spawn(async move {
        let mut store = Vec::new();
        let request = read_request(&mut server, &mut store).await.unwrap();
        assert!(matches!(request, Frame::Request { .. }));
        write_frame_to_stream(&mut server, &Frame::Stop)
            .await
            .unwrap();

        for _ in 0..8 {
            let request = wait_for_request(&mut server, &mut store).await.unwrap();
            assert!(matches!(request, Frame::Request { .. }));
            write_frame_to_stream(&mut server, &Frame::Stop)
                .await
                .unwrap();
        }
    });

    let request = Frame::Request { messages: vec![] };
    let mut store = Vec::new();
    write_frame_to_stream(&mut client, &request).await.unwrap();
    let reply: Frame = read_frame_from_stream(&mut client, &mut store, None, None)
        .await
        .unwrap();
    assert!(matches!(reply, Frame::Stop));
    for _ in 0..8 {
        tokio::time::sleep(Duration::from_secs(40)).await;
        write_frame_to_stream(&mut client, &request).await.unwrap();
        let reply: Frame = read_frame_from_stream(&mut client, &mut store, None, None)
            .await
            .unwrap();
        assert!(matches!(reply, Frame::Stop));
    }
    reader.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn idle_initial_connection_times_out() {
    let (_client, mut server) = UnixStream::pair().unwrap();
    let mut store = Vec::new();
    let started = tokio::time::Instant::now();
    let read = read_request(&mut server, &mut store);
    let result = tokio::time::timeout(Duration::from_secs(31), read)
        .await
        .expect("an idle probe has a bounded lifetime");
    let elapsed = started.elapsed();
    assert!(matches!(result, Err(ProtocolError::Timeout)));
    assert!(elapsed >= Duration::from_secs(30));
}

#[tokio::test(start_paused = true)]
async fn incomplete_request_on_established_connection_times_out() {
    let (mut client, mut server) = UnixStream::pair().unwrap();
    let request = postcard::to_allocvec(&Frame::Request { messages: vec![] }).unwrap();
    client.write_all(&request[..1]).await.unwrap();

    // Observing socket readiness before starting the virtual frame deadline.
    server.readable().await.unwrap();
    let mut store = Vec::new();
    let started = tokio::time::Instant::now();
    let read = wait_for_request(&mut server, &mut store);
    let result = tokio::time::timeout(Duration::from_secs(31), read)
        .await
        .expect("an incomplete request has a bounded lifetime");
    let elapsed = started.elapsed();
    assert!(matches!(result, Err(ProtocolError::Timeout)));
    assert!(elapsed >= Duration::from_secs(30));
}

#[tokio::test]
async fn buffered_request_is_delivered_before_disconnect() {
    let (client, mut server) = UnixStream::pair().unwrap();
    let mut store = postcard::to_allocvec(&Frame::Request {
        messages: vec![Message::User("already received".into())],
    })
    .unwrap();
    drop(client);

    let request = wait_for_request(&mut server, &mut store).await.unwrap();
    let Frame::Request { messages } = request else {
        panic!("buffered request is delivered even after the peer closes");
    };
    assert!(matches!(messages.as_slice(), [Message::User(text)] if text == "already received"));

    let request = wait_for_request(&mut server, &mut store).await;
    assert!(matches!(request, Err(ProtocolError::Disconnect)));
}

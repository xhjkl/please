use super::*;
use crate::protocol::{Frame, read_frame, write_frame};
use crate::test_support::TempDir;

#[tokio::test]
async fn existing_hub_is_used_without_loading_local_weights() -> Result<()> {
    let directory = TempDir::new()?;
    let path = directory.0.join("socket");
    let listener = tokio::net::UnixListener::bind(&path)?;
    let mut probe = connect_to_hub_at(&path).await?;
    let (mut hub, _) = listener.accept().await?;
    write_frame(&mut hub, &Frame::Answer("forwarded hub".into())).await?;
    let mut store = Vec::new();
    let response = read_frame(&mut probe, &mut store, None).await?;
    assert!(matches!(response, Frame::Answer(text) if text == "forwarded hub"));
    Ok(())
}

#[tokio::test]
async fn a_busy_hub_turns_the_probe_away() -> Result<()> {
    for (reply, admitted) in [(Frame::Welcome, true), (Frame::Busy, false)] {
        let (mut probe, mut hub) = UnixStream::pair()?;
        let serving = tokio::spawn(async move {
            let mut store = Vec::new();
            let hello = read_frame(&mut hub, &mut store, None).await?;
            assert!(matches!(hello, Frame::Hello { .. }));
            write_frame(&mut hub, &reply).await
        });
        let greeted = crate::protocol::greet(&mut probe).await;
        serving.await??;
        match greeted {
            Ok(()) => assert!(admitted),
            Err(error) => {
                assert!(!admitted);
                assert!(error.is::<crate::protocol::HubBusy>());
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn invalid_socket_does_not_fall_back_to_local_inference() -> Result<()> {
    let directory = TempDir::new()?;
    let path = directory.0.join("socket");
    std::fs::write(&path, "keep this file")?;
    let connected = connect_to_hub_at(&path).await;
    assert!(connected.is_err());
    let content = std::fs::read_to_string(path)?;
    assert_eq!(content, "keep this file");
    Ok(())
}

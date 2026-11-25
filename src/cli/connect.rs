use eyre::{Result, eyre};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::net::UnixStream;

/// Why there is no hub to talk to at a socket path.
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("missing socket: {}", .0.display())]
    Missing(PathBuf),
    #[error("permission denied: {}", .0.display())]
    PermissionDenied(PathBuf),
    #[error("not a socket: {}", .0.display())]
    NotSocket(PathBuf),
    #[error("no listener at: {}", .0.display())]
    NoListener(PathBuf),
}

impl ConnectError {
    /// Whether no hub is there yet, as opposed to something in the way of one.
    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Missing(_) | Self::NoListener(_))
    }
}

/// Connect to a hub already listening at `path`.
pub async fn try_connect_to_hub(path: &Path) -> std::result::Result<UnixStream, ConnectError> {
    use std::io::ErrorKind;
    use std::os::unix::fs::FileTypeExt;
    let metadata = std::fs::metadata(path);
    if metadata.is_ok_and(|metadata| !metadata.file_type().is_socket()) {
        return Err(ConnectError::NotSocket(path.to_path_buf()));
    }
    let connect = UnixStream::connect(path);
    let connected = tokio::time::timeout(Duration::from_millis(64), connect).await;
    let error: fn(PathBuf) -> ConnectError = match connected {
        Err(_elapsed) => ConnectError::NoListener,
        Ok(Err(error)) => match error.kind() {
            ErrorKind::NotFound => ConnectError::Missing,
            ErrorKind::PermissionDenied => ConnectError::PermissionDenied,
            ErrorKind::ConnectionRefused | ErrorKind::WouldBlock | ErrorKind::TimedOut => {
                ConnectError::NoListener
            }
            _ => ConnectError::NotSocket,
        },
        Ok(Ok(stream)) => return Ok(stream),
    };
    Err(error(path.to_path_buf()))
}

/// Start a detached hub and connect once it has bound its socket.
async fn start_hub(path: &Path) -> Result<UnixStream> {
    let exe = std::env::current_exe()?;
    let mut hub = tokio::process::Command::new(exe);
    hub.arg("run")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0);
    let mut child = hub.spawn()?;
    let connected = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let connected = try_connect_to_hub(path).await;
            match connected {
                Err(error) if !error.is_absent() => return Err(eyre!(error)),
                Err(_absent) => {}
                Ok(stream) => return Ok(stream),
            }
            let status = child.try_wait()?;
            if let Some(status) = status {
                return Err(eyre!(
                    "hub exited with {status}; run `please run` to see its startup error"
                ));
            }
            tokio::time::sleep(Duration::from_millis(128)).await;
        }
    })
    .await;
    let connected = connected.unwrap_or_else(|_| {
        Err(eyre!(
            "hub did not bind {} within 30 seconds",
            path.display()
        ))
    });
    if connected.is_err() {
        // Reaping a failed startup; an embedded fallback could load a second model.
        let _ = child.kill().await;
        let _ = child.wait().await;
    }
    connected
}

/// Connect to an existing or forwarded hub before considering a local embedded hub, and be
/// admitted by it; a hub serving another probe fails this with [`crate::protocol::HubBusy`].
pub async fn obtain_control_stream() -> Result<UnixStream> {
    let path = crate::hub::socket_path();
    let mut stream = obtain_control_stream_at(&path).await?;
    crate::protocol::greet(&mut stream).await?;
    Ok(stream)
}

async fn obtain_control_stream_at(path: &Path) -> Result<UnixStream> {
    let connected = try_connect_to_hub(path).await;
    match connected {
        Err(error) if !error.is_absent() => return Err(eyre!(error)),
        Ok(stream) => {
            tracing::info!("probe: connected to existing hub at {}", path.display());
            return Ok(stream);
        }
        Err(_absent) => {}
    }

    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        eyre::bail!("no hub at {}; {}", path.display(), crate::hub::UNSUPPORTED);
    }
    let prefers_daemon = std::env::var_os("PLEASE_SPAWN_HUB").is_some();
    if prefers_daemon {
        return start_hub(path).await;
    }

    let stream = crate::hub::spawn(None).await?;
    tracing::info!("probe: connected to the embedded hub");
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Frame, read_frame_from_stream, write_frame_to_stream};
    use crate::test_support::TempDir;

    #[tokio::test]
    async fn existing_hub_is_used_without_loading_local_weights() -> Result<()> {
        let directory = TempDir::new()?;
        let path = directory.0.join("socket");
        let listener = tokio::net::UnixListener::bind(&path)?;
        let mut probe = obtain_control_stream_at(&path).await?;
        let (mut hub, _) = listener.accept().await?;
        write_frame_to_stream(&mut hub, &Frame::Answer("forwarded hub".into())).await?;
        let mut store = Vec::new();
        let response: Frame = read_frame_from_stream(&mut probe, &mut store, None, None).await?;
        assert!(matches!(response, Frame::Answer(text) if text == "forwarded hub"));
        Ok(())
    }

    #[tokio::test]
    async fn a_busy_hub_turns_the_probe_away() -> Result<()> {
        for (reply, admitted) in [(Frame::Welcome, true), (Frame::Busy, false)] {
            let (mut probe, mut hub) = UnixStream::pair()?;
            let serving = tokio::spawn(async move {
                let mut store = Vec::new();
                let hello: Frame = read_frame_from_stream(&mut hub, &mut store, None, None).await?;
                assert!(matches!(hello, Frame::Hello { .. }));
                write_frame_to_stream(&mut hub, &reply).await
            });
            let greeted = crate::protocol::greet(&mut probe).await;
            serving.await??;
            match greeted {
                Ok(()) => assert!(admitted),
                Err(error) => {
                    assert!(!admitted);
                    assert!(error.downcast_ref::<crate::protocol::HubBusy>().is_some());
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
        let connected = obtain_control_stream_at(&path).await;
        assert!(connected.is_err());
        let content = std::fs::read_to_string(path)?;
        assert_eq!(content, "keep this file");
        Ok(())
    }
}

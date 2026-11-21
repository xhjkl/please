use eyre::{Result, eyre};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::net::UnixStream;

#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("missing socket: {}", path.display())]
    Missing { path: PathBuf },
    #[error("permission denied: {}", path.display())]
    PermissionDenied { path: PathBuf },
    #[error("not a socket: {}", path.display())]
    NotSocket { path: PathBuf },
    #[error("no listener at: {}", path.display())]
    NoListener { path: PathBuf },
}

/// Try to connect to an existing hub via Unix socket.
pub async fn try_connect_to_hub(path: &Path) -> std::result::Result<UnixStream, ConnectError> {
    let metadata = std::fs::metadata(path);
    if let Ok(metadata) = metadata {
        use std::os::unix::fs::FileTypeExt;
        if !metadata.file_type().is_socket() {
            return Err(ConnectError::NotSocket {
                path: path.to_path_buf(),
            });
        }
    }
    let connect = UnixStream::connect(path);
    let connected = tokio::time::timeout(Duration::from_millis(64), connect).await;
    match connected {
        Err(_elapsed) => Err(ConnectError::NoListener {
            path: path.to_path_buf(),
        }),
        Ok(Err(error)) => match error.kind() {
            std::io::ErrorKind::NotFound => Err(ConnectError::Missing {
                path: path.to_path_buf(),
            }),
            std::io::ErrorKind::PermissionDenied => Err(ConnectError::PermissionDenied {
                path: path.to_path_buf(),
            }),
            std::io::ErrorKind::ConnectionRefused => Err(ConnectError::NoListener {
                path: path.to_path_buf(),
            }),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
                Err(ConnectError::NoListener {
                    path: path.to_path_buf(),
                })
            }
            _ => Err(ConnectError::NotSocket {
                path: path.to_path_buf(),
            }),
        },
        Ok(Ok(stream)) => Ok(stream),
    }
}

/// Start a detached hub and connect once it has bound its socket.
async fn start_hub(path: &Path) -> Result<UnixStream> {
    let exe = std::env::current_exe()?;
    let mut cmd = tokio::process::Command::new(exe);
    cmd.arg("run");
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());
    let mut child = cmd.spawn()?;
    let connected = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let connected = try_connect_to_hub(path).await;
            match connected {
                Err(
                    error
                    @ (ConnectError::NotSocket { .. } | ConnectError::PermissionDenied { .. }),
                ) => {
                    return Err(eyre!(error));
                }
                Err(ConnectError::NoListener { .. } | ConnectError::Missing { .. }) => {}
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
        Err(error @ (ConnectError::NotSocket { .. } | ConnectError::PermissionDenied { .. })) => {
            return Err(eyre!(error));
        }
        Err(ConnectError::NoListener { .. } | ConnectError::Missing { .. }) => {}
        Ok(stream) => {
            tracing::info!("probe: connected to existing hub at {}", path.display());
            return Ok(stream);
        }
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

//! Host Oco and serve streamed requests from local or forwarded clients.
use super::socket_path;
use eyre::{Result, eyre};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

use super::inference::{self, Engine};
use crate::protocol::{Frame, ProtocolError, read_frame_from_stream, write_frame_to_stream};

/// Load the hub's explicit checkpoint or discover the best installed model on this machine.
async fn load(model_path: Option<&str>) -> Result<Arc<Engine>> {
    let model = match model_path {
        Some(path) => super::discovery::inspect(path.into())?,
        None => {
            let model = super::discovery::selected_model()?;
            model.ok_or_else(|| eyre!(super::NoWeights))?
        }
    };
    tracing::info!(model_path=%model.path.display(), dialect=?model.dialect, "hub: selected model");
    let model = inference::load_model(&model.path, model.dialect).await?;
    tracing::info!("hub: model loaded");
    Ok(Arc::new(model))
}

/// Ensure the socket directory exists and is private to its owner.
fn ensure_socket_dir(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Bind a hub without displacing another listener; retain the lock for the listener's lifetime.
async fn bind_listener(path: &std::path::Path) -> Result<(UnixListener, std::fs::File)> {
    use std::os::unix::fs::OpenOptionsExt;
    ensure_socket_dir(path)?;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path.with_file_name(".socket.lock"))?;
    lock.try_lock()
        .map_err(|error| eyre!("hub already owns {}: {error}", path.display()))?;
    cleanup_stale_socket(path).await?;
    let listener = UnixListener::bind(path)?;
    Ok((listener, lock))
}

/// Remove only a socket with no listener, preserving active hubs and forwarding endpoints.
async fn cleanup_stale_socket(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::FileTypeExt;
    let metadata = std::fs::symlink_metadata(path);
    let metadata = match metadata {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
        Ok(metadata) => metadata,
    };
    if !metadata.file_type().is_socket() {
        return Err(eyre!(
            "hub: path exists but is not a socket: {}",
            path.display()
        ));
    }
    let connected =
        tokio::time::timeout(Duration::from_millis(250), UnixStream::connect(path)).await;
    match connected {
        Err(_) => return Err(eyre!("hub: cannot verify that {} is stale", path.display())),
        Ok(Ok(_stream)) => return Err(eyre!("hub already listening at {}", path.display())),
        Ok(Err(error)) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Ok(Err(error)) if error.kind() == std::io::ErrorKind::ConnectionRefused => {}
        Ok(Err(error)) => return Err(error.into()),
    }
    std::fs::remove_file(path)?;
    Ok(())
}

/// Read a complete request within the frame deadline.
async fn read_request(
    stream: &mut UnixStream,
    store: &mut Vec<u8>,
) -> std::result::Result<Frame, ProtocolError> {
    let per_read_timeout = Some(Duration::from_millis(250));
    let total_timeout = Some(Duration::from_secs(30));
    read_frame_from_stream(stream, store, per_read_timeout, total_timeout).await
}

/// Wait for another request, allowing the client to run tools before sending it.
async fn wait_for_request(
    stream: &mut UnixStream,
    store: &mut Vec<u8>,
) -> std::result::Result<Frame, ProtocolError> {
    if store.is_empty() {
        // Starting the frame deadline only after data arrives; tool waits can be arbitrarily long.
        let mut byte = [0];
        let read = stream.read(&mut byte).await.map_err(ProtocolError::Io)?;
        if read == 0 {
            return Err(ProtocolError::Disconnect);
        }
        store.extend_from_slice(&byte);
    }

    read_request(stream, store).await
}

/// Serve a long-lived client connection, handling multiple turns per session.
async fn accept_and_serve_request(stream: &mut UnixStream, hub: Arc<Engine>) -> Result<()> {
    tracing::info!("hub: connection accepted");

    let mut store = Vec::with_capacity(4096);
    let mut request = read_request(stream, &mut store).await;
    // Only an introduced probe may use the session, and only one probe at a time.
    let mut admission = None;

    loop {
        let frame = match request {
            Err(ProtocolError::Disconnect) => return Ok(()),
            Err(e) => return Err(eyre!(e)),
            Ok(frame) => frame,
        };

        let served = match frame {
            Frame::ModelInfoRequest => {
                let response = Frame::ModelInfo {
                    engine: "Oco".into(),
                    path: hub.path.display().to_string(),
                };
                write_frame_to_stream(stream, &response).await
            }
            Frame::Hello { probe } => {
                let Some(admitted) = hub.occupancy.admit(probe) else {
                    tracing::info!("hub: turned away probe {probe:016x} while serving another");
                    return write_frame_to_stream(stream, &Frame::Busy).await;
                };
                admission = Some(admitted);
                write_frame_to_stream(stream, &Frame::Welcome).await
            }
            Frame::Request { .. } | Frame::Restart { .. } | Frame::BenchmarkRequest { .. }
                if admission.is_none() =>
            {
                let refusal = Frame::Error(
                    "the probe did not say hello; probe and hub are probably different versions of please"
                        .into(),
                );
                return write_frame_to_stream(stream, &refusal).await;
            }
            Frame::Request {
                place,
                setting,
                id,
                messages,
            } => {
                tracing::info!("hub: received inference request");
                let (mut reader, mut writer) = stream.split();
                let turn = super::turn::serve(&mut writer, &hub, &place, &setting, id, messages);
                tokio::pin!(turn);
                let mut byte = [0];
                tokio::select! {
                    biased;
                    read = reader.read(&mut byte) => {
                        let read = read?;
                        // Dropping the receiver stops generation even when there are no deltas.
                        if read == 0 {
                            return Ok(());
                        }
                        store.extend_from_slice(&byte[..read]);
                        turn.await
                    }
                    result = &mut turn => result,
                }
            }
            Frame::Restart {
                place,
                setting,
                summarize,
            } => {
                tracing::info!("hub: received restart request");
                let (mut reader, mut writer) = stream.split();
                let restart = super::turn::restart(&mut writer, &hub, &place, &setting, summarize);
                tokio::pin!(restart);
                let mut byte = [0];
                // A probe that goes away cancels the restart before it is kept.
                tokio::select! {
                    biased;
                    read = reader.read(&mut byte) => {
                        if read? == 0 {
                            return Ok(());
                        }
                        store.extend_from_slice(&byte[..1]);
                        restart.await
                    }
                    result = &mut restart => result,
                }
            }
            Frame::BenchmarkRequest {
                setting,
                messages,
                max_tokens,
            } => {
                tracing::info!("hub: received benchmark request");
                let report =
                    super::benchmark::measure(hub.clone(), setting, messages, max_tokens).await;
                match report {
                    Err(error) => Err(error),
                    Ok(report) => {
                        write_frame_to_stream(stream, &Frame::BenchmarkResult(report)).await
                    }
                }
            }
            _ => return Err(eyre!("bad request: {frame:?}")),
        };

        if let Err(error) = served {
            write_frame_to_stream(
                stream,
                &Frame::Error(format!("inference failed: {error:#}")),
            )
            .await?;
            return Ok(());
        }
        request = wait_for_request(stream, &mut store).await;
    }
}

/// Bind the shared socket and serve all clients with one loaded model.
pub async fn run() -> Result<()> {
    let path = socket_path();
    let (listener, _lock) = bind_listener(&path).await?;
    tracing::info!("hub: listening at {}", path.display());
    let hub = load(None).await?;

    loop {
        let (stream, _addr) = listener.accept().await?;
        serve_connection(stream, hub.clone());
    }
}

/// Start a colocated hub and return the probe's end of its Unix socket pair.
/// An explicit checkpoint is used only by the local benchmark harness.
/// The probe's embedded engine lives as long as the process: reconnecting after Ctrl-C only
/// opens a new socket pair onto it, keeping the model and its session loaded.
pub async fn spawn(model_path: Option<&str>) -> Result<UnixStream> {
    static EMBEDDED: tokio::sync::OnceCell<Arc<Engine>> = tokio::sync::OnceCell::const_new();
    let hub = match model_path {
        Some(_) => load(model_path).await?,
        None => EMBEDDED.get_or_try_init(|| load(None)).await?.clone(),
    };
    let (probe_end, hub_end) = UnixStream::pair()?;
    serve_connection(hub_end, hub);
    Ok(probe_end)
}

/// Serve the same protocol for embedded, local, and forwarded probes.
fn serve_connection(mut stream: UnixStream, hub: Arc<Engine>) {
    tokio::spawn(async move {
        let served = accept_and_serve_request(&mut stream, hub).await;
        if let Err(error) = served {
            let _ = stream.shutdown().await;
            tracing::error!("hub: connection error: {error}");
        }
    });
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

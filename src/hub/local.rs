//! Host Oco and serve streamed requests from local or forwarded clients.
use super::socket_path;
use eyre::{Result, eyre};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

use super::dialect::Delta;
use super::inference::{self, Engine};
use crate::protocol::Message;
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
        .open(path.with_extension("lock"))?;
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

/// Run streaming inference and forward deltas to the sink.
async fn serve_one_turn(
    sink: &mut (impl AsyncWriteExt + Unpin),
    hub: Arc<Engine>,
    connection: u64,
    history: &[Message],
) -> Result<()> {
    let mut parser = hub.output_parser()?;
    let (generated_tx, mut generated_rx) =
        tokio::sync::mpsc::unbounded_channel::<inference::Generated>();

    let history = history.to_owned();
    let worker = hub.clone();
    let inference = tokio::spawn(async move {
        inference::generate(&worker, connection, &history, generated_tx, None).await
    });

    while let Some(event) = generated_rx.recv().await {
        match event {
            inference::Generated::Stop => break,
            inference::Generated::Token(token) => {
                let deltas = parser.push_token(token)?;
                for delta in deltas {
                    match delta {
                        Delta::Answer(text) => {
                            write_frame_to_stream(sink, &Frame::Answer(text)).await?;
                        }
                        Delta::Thinking(text) => {
                            write_frame_to_stream(sink, &Frame::Thinking(text)).await?;
                        }
                    }
                }
            }
        }
    }

    inference.await??;
    let calls = parser.finish();
    match calls {
        Err(error) => {
            write_frame_to_stream(sink, &Frame::ToolCallParseError(error.to_string())).await?;
        }
        Ok(calls) => {
            for call in calls {
                write_frame_to_stream(
                    sink,
                    &Frame::ToolCall {
                        name: call.name,
                        arguments_json: serde_json::to_string(&call.arguments)?,
                    },
                )
                .await?;
            }
        }
    }
    write_frame_to_stream(sink, &Frame::Stop).await?;

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
    // Requests on one connection extend one conversation, so they share one session.
    let connection = hub.connect();
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
            Frame::Request { .. } | Frame::BenchmarkRequest { .. } if admission.is_none() => {
                let refusal = Frame::Error(
                    "the probe did not say hello; probe and hub are probably different versions of please"
                        .into(),
                );
                return write_frame_to_stream(stream, &refusal).await;
            }
            Frame::Request { messages } => {
                tracing::info!("hub: received inference request");
                let (mut reader, mut writer) = stream.split();
                let turn = serve_one_turn(&mut writer, hub.clone(), connection, &messages);
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
            Frame::BenchmarkRequest {
                messages,
                max_tokens,
            } => {
                tracing::info!("hub: received benchmark request");
                let report = super::benchmark::measure(hub.clone(), messages, max_tokens).await;
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

//! Host Oco and serve probes, local or forwarded, over the hub's socket.
use super::socket_path;
use eyre::{Result, bail, eyre};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

use super::inference::Engine;
use crate::protocol::{Frame, ProtocolError, read_frame, write_frame};

/// Load the hub's explicit checkpoint, or the best one installed on this machine.
async fn load(checkpoint: Option<&str>) -> Result<Arc<Engine>> {
    let checkpoint = match checkpoint {
        Some(path) => super::discovery::inspect(path.into())?,
        None => {
            let selected = super::discovery::select()?;
            selected.ok_or(super::NoWeights)?
        }
    };
    tracing::info!(path=%checkpoint.path.display(), dialect=?checkpoint.dialect, "hub: selected checkpoint");
    let engine = Engine::load(&checkpoint.path, checkpoint.dialect).await?;
    tracing::info!("hub: model loaded");
    Ok(Arc::new(engine))
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
        bail!("hub: path exists but is not a socket: {}", path.display());
    }
    let connected =
        tokio::time::timeout(Duration::from_millis(250), UnixStream::connect(path)).await;
    match connected {
        Err(_) => bail!("hub: cannot verify that {} is stale", path.display()),
        Ok(Ok(_stream)) => bail!("hub already listening at {}", path.display()),
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
    read_frame(stream, store, Some(Duration::from_secs(30))).await
}

/// Wait for another request, allowing the probe to run tools before sending it.
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

/// Run `work` unless the probe hangs up first, which cancels it and gives `None`. A byte that
/// arrives first opens the probe's next request, and `work` then runs to its end.
async fn unless_hung_up(
    reader: &mut tokio::net::unix::ReadHalf<'_>,
    store: &mut Vec<u8>,
    work: impl Future<Output = Result<()>>,
) -> Result<Option<Result<()>>> {
    tokio::pin!(work);
    let mut byte = [0];
    tokio::select! {
        biased;
        read = reader.read(&mut byte) => {
            if read? == 0 {
                return Ok(None);
            }
            store.extend_from_slice(&byte);
            Ok(Some(work.await))
        }
        served = &mut work => Ok(Some(served)),
    }
}

/// Serve a probe's requests on one long-lived connection, however many turns they span.
async fn serve_requests(stream: &mut UnixStream, engine: Arc<Engine>) -> Result<()> {
    tracing::info!("hub: connection accepted");

    let mut store = Vec::with_capacity(4096);
    let mut request = read_request(stream, &mut store).await;
    // Only an introduced probe may use the engine, and only one probe at a time.
    let mut admission = None;

    loop {
        let frame = match request {
            Err(ProtocolError::Disconnect) => return Ok(()),
            Err(error) => return Err(error.into()),
            Ok(frame) => frame,
        };

        let served = match frame {
            Frame::CheckpointRequest => {
                let response = Frame::Checkpoint {
                    path: engine.path.display().to_string(),
                };
                write_frame(stream, &response).await
            }
            Frame::Hello { probe } => {
                let Some(admitted) = engine.occupancy.admit(probe) else {
                    tracing::info!("hub: turned away probe {probe:016x} while serving another");
                    return write_frame(stream, &Frame::Busy).await;
                };
                admission = Some(admitted);
                write_frame(stream, &Frame::Welcome).await
            }
            Frame::Advance { .. } | Frame::Restart { .. } | Frame::BenchmarkRequest { .. }
                if admission.is_none() =>
            {
                let refusal = Frame::Error(
                    "the probe did not say hello; probe and hub are probably different versions of please"
                        .into(),
                );
                return write_frame(stream, &refusal).await;
            }
            Frame::Advance {
                place,
                setting,
                id,
                messages,
            } => {
                tracing::info!("hub: received inference request");
                let (mut reader, mut writer) = stream.split();
                let turn =
                    super::turn::advance(&mut writer, &engine, &place, &setting, id, messages);
                // Dropping the receiver stops generation even when there are no deltas.
                let Some(served) = unless_hung_up(&mut reader, &mut store, turn).await? else {
                    return Ok(());
                };
                served
            }
            Frame::Restart {
                place,
                setting,
                summarize,
            } => {
                tracing::info!("hub: received restart request");
                let (mut reader, mut writer) = stream.split();
                let restart =
                    super::turn::restart(&mut writer, &engine, &place, &setting, summarize);
                // A probe that goes away cancels the restart before it is kept.
                let Some(served) = unless_hung_up(&mut reader, &mut store, restart).await? else {
                    return Ok(());
                };
                served
            }
            Frame::BenchmarkRequest {
                setting,
                messages,
                max_tokens,
            } => {
                tracing::info!("hub: received benchmark request");
                let report =
                    super::benchmark::measure(engine.clone(), setting, messages, max_tokens).await;
                match report {
                    Err(error) => Err(error),
                    Ok(report) => write_frame(stream, &Frame::BenchmarkResult(report)).await,
                }
            }
            _ => bail!("bad request: {frame:?}"),
        };

        if let Err(error) = served {
            write_frame(
                stream,
                &Frame::Error(format!("inference failed: {error:#}")),
            )
            .await?;
            return Ok(());
        }
        request = wait_for_request(stream, &mut store).await;
    }
}

/// Bind the shared socket and serve every probe with one loaded model.
pub async fn run() -> Result<()> {
    let path = socket_path();
    let (listener, _lock) = bind_listener(&path).await?;
    tracing::info!("hub: listening at {}", path.display());
    let engine = load(None).await?;

    loop {
        let (stream, _addr) = listener.accept().await?;
        serve_connection(stream, engine.clone());
    }
}

/// Embed a hub in this process and return the probe's end of its Unix socket pair.
/// An explicit checkpoint is used only by the local benchmark harness.
/// The embedded engine lives as long as the process: reconnecting after Ctrl-C only opens a
/// new socket pair onto it, keeping the model and its session loaded.
pub async fn embed(checkpoint: Option<&str>) -> Result<UnixStream> {
    static EMBEDDED: tokio::sync::OnceCell<Arc<Engine>> = tokio::sync::OnceCell::const_new();
    let engine = match checkpoint {
        Some(_) => load(checkpoint).await?,
        None => EMBEDDED.get_or_try_init(|| load(None)).await?.clone(),
    };
    let (probe_end, hub_end) = UnixStream::pair()?;
    serve_connection(hub_end, engine);
    Ok(probe_end)
}

/// Serve the same protocol for embedded, local, and forwarded probes.
fn serve_connection(mut stream: UnixStream, engine: Arc<Engine>) {
    tokio::spawn(async move {
        let served = serve_requests(&mut stream, engine).await;
        if let Err(error) = served {
            let _ = stream.shutdown().await;
            tracing::error!("hub: connection error: {error}");
        }
    });
}

#[cfg(test)]
mod tests;

//! The hub is a background process that hosts the inference engine and accepts requests from the CLI.
use eyre::{Result, eyre};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};

use crate::inference;
use crate::protocol::Message;
use crate::protocol::{Frame, read_frame_from_stream, write_frame_to_stream};

/// Loaded backend and model; shared across connections.
struct Hub {
    backend: gg::llama_backend::LlamaBackend,
    model: gg::model::LlamaModel,
}

/// Default UNIX socket location under `~/.please/socket`.
pub fn socket_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    std::path::Path::new(&home).join(".please").join("socket")
}

/// Ensure the socket directory exists and is private (0700 on Unix).
pub fn ensure_socket_dir(path: &std::path::Path) -> Result<()> {
    use std::fs;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(dir)?.permissions();
            perms.set_mode(0o700);
            fs::set_permissions(dir, perms)?;
        }
    }
    Ok(())
}

/// Remove a pre-existing socket file, erroring if a non-socket exists there.
pub fn cleanup_stale_socket(path: &std::path::Path) -> Result<()> {
    use std::fs;
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileTypeExt;
                if meta.file_type().is_socket() {
                    let _ = fs::remove_file(path);
                } else {
                    return Err(eyre!(
                        "hub: path exists but is not a socket: {}",
                        path.display()
                    ));
                }
            }
        }
        Err(err) => {
            if err.kind() != std::io::ErrorKind::NotFound {
                return Err(err.into());
            }
        }
    }
    Ok(())
}

/// Run streaming inference and forward deltas to the sink.
async fn serve_one_turn(
    sink: &mut (impl AsyncWriteExt + Unpin),
    hub: Arc<Hub>,
    history: &[Message],
) -> Result<()> {
    let (piece_tx, mut piece_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Use the provided chat history directly; template rendering occurs in inference.
    let history = history.to_owned();
    let also_hub = hub.clone();
    let inference = tokio::spawn(async move {
        inference::infer_into_stream(&also_hub.backend, &also_hub.model, &history, piece_tx).await
    });

    while let Some(piece) = piece_rx.recv().await {
        write_frame_to_stream(sink, &Frame::Answer(piece)).await?;
    }

    // Ensure inference completed
    let pending = inference.await.map_err(|e| eyre!(e))??;

    // If incomplete UTF-8 remains, emit replacement character once and log.
    if !pending.is_empty() {
        tracing::error!(
            remaining_bytes = pending.len(),
            ?pending,
            "hub: incomplete utf-8 at end of stream; emitting replacement char"
        );
        write_frame_to_stream(sink, &Frame::Answer("\u{FFFD}".to_string())).await?;
    }

    write_frame_to_stream(sink, &Frame::Stop).await?;

    Ok(())
}

/// Serve a long-lived client connection, handling multiple turns per session.
async fn accept_and_serve_request(stream: &mut UnixStream, hub: Arc<Hub>) -> Result<()> {
    // Apply conservative read timeouts to make slow or stuck probes go away.
    let per_read_timeout = Some(Duration::from_millis(250));
    let total_timeout = Some(Duration::from_secs(30));

    tracing::info!("hub: connection accepted");

    let mut store = Vec::with_capacity(4096);

    loop {
        // Wait for the next request; keep the connection alive between turns.
        let req: std::result::Result<Frame, crate::protocol::ProtocolError> =
            read_frame_from_stream(stream, &mut store, per_read_timeout, total_timeout).await;

        let req = match req {
            Err(crate::protocol::ProtocolError::Disconnect) => {
                // Normal end of session
                break;
            }
            Err(e) => return Err(eyre!(e)),
            Ok(frame) => frame,
        };

        tracing::info!("hub: received inference request");

        let history = match req {
            Frame::Request { messages } => messages,
            _ => return Err(eyre!("bad request: {req:?}")),
        };

        serve_one_turn(stream, hub.clone(), &history).await?;

        // Roll over to the next turn
    }
    Ok(())
}

/// Hub main loop: bind socket, load model once, accept clients forever.
pub async fn run() -> Result<()> {
    let socket_path = socket_path();
    ensure_socket_dir(&socket_path)?;
    cleanup_stale_socket(&socket_path)?;

    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!("hub: listening at {}", socket_path.display());

    // Load model once and accept connections in a loop.
    let Some(model_path) = crate::cli::discovery::choose_best_model_path() else {
        return Err(eyre!("hub: no model found"));
    };
    let model_path = model_path.to_string_lossy().to_string();
    tracing::info!(%model_path, "hub: selected model");
    let (backend, model) = crate::inference::load_model(&model_path)?;
    let hub = Arc::new(Hub { backend, model });

    tracing::info!("hub: model loaded");

    loop {
        let (mut stream, _addr) = listener.accept().await?;
        let hub = hub.clone();
        tokio::spawn(async move {
            let served = accept_and_serve_request(&mut stream, hub).await;
            if let Err(e) = served {
                let _ = stream.shutdown().await;
                tracing::error!("hub: connection error: {e}");
            }
        });
    }
}

/// Convenience for in-process use: serve a single client over a UnixStream pair.
pub async fn spawn() -> Result<UnixStream> {
    // Load model once and serve a single request over an in-process stream pair.
    let Some(model_path) = crate::cli::discovery::choose_best_model_path() else {
        return Err(eyre!("hub: no model found"));
    };
    tracing::info!(model_path=%model_path.display(), "hub: selected model");
    let model_path = model_path.to_string_lossy().to_string();
    let (backend, model) = crate::inference::load_model(&model_path)?;
    let hub = Hub { backend, model };

    let (probe_end, mut hub_end) = UnixStream::pair()?;
    tokio::spawn(async move {
        let served = accept_and_serve_request(&mut hub_end, Arc::new(hub)).await;
        if let Err(e) = served {
            let _ = hub_end.shutdown().await;
            tracing::error!("hub: connection error: {e}");
        }
    });

    Ok(probe_end)
}

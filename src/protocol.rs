//! Minimal postcard-framed protocol between CLI and hub.
use eyre::{Result, eyre};
use serde::{Deserialize, Serialize};

/// Probe-local session facts; the hub turns them into model instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub today: String,
    pub reasoning: String,
    pub stdin: Option<String>,
    /// None for a terminal; an empty path for a redirect whose destination is unknown.
    pub stdout_redirection: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    System(String),
    Developer(String),
    User(String),
    Reasoning(String),
    Tool(String),
    Assistant(String),
    /// Session context, present only at the beginning of probe history.
    Context(Context),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Frame {
    Request {
        messages: Vec<Message>,
    },
    Log(String),
    Answer(String),
    Thinking(String),
    ToolCall {
        name: String,
        /// Tool arguments encoded as JSON text because postcard cannot deserialize `serde_json::Value`.
        arguments_json: String,
    },
    ToolCallParseError(String),
    Stop,
    ModelInfoRequest,
    ModelInfo {
        engine: String,
        path: String,
    },
    Error(String),
    BenchmarkRequest {
        messages: Vec<Message>,
        max_tokens: usize,
    },
    /// Hub-side measurements encoded as JSON for the benchmark's JSON-lines output.
    BenchmarkResult(String),
    /// A probe's first frame on a connection, naming its process so the hub tells that probe
    /// reconnecting apart from another one wanting in.
    Hello {
        probe: u64,
    },
    /// The hub serves the probe that said hello.
    Welcome,
    /// The hub serves another probe; it closes this connection.
    Busy,
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("connection dropped while the request was being read")]
    Disconnect,
    #[error("timed out while reading request")]
    Timeout,
    #[error("io error: {0}")]
    Io(std::io::Error),
    #[error("cannot decode a frame ({0}); probe and hub are probably different versions of please")]
    Decode(postcard::Error),
}

/// The hub serves another probe and turned this one away.
#[derive(Debug, thiserror::Error)]
#[error("the model is busy serving another please session; retry once that session ends")]
pub struct HubBusy;

/// This process's name for the hub: random, so probes on different machines never collide,
/// and fixed for the process, so its reconnections are recognized.
fn probe_id() -> u64 {
    use std::hash::BuildHasher;
    static ID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *ID.get_or_init(|| std::hash::RandomState::new().hash_one(std::process::id()))
}

/// Introduce this probe on a fresh connection and wait until the hub takes it.
pub async fn greet(stream: &mut tokio::net::UnixStream) -> Result<()> {
    let hello = Frame::Hello { probe: probe_id() };
    write_frame_to_stream(stream, &hello).await?;
    let mut store = Vec::new();
    match read_frame_from_stream(stream, &mut store, None, None).await? {
        Frame::Welcome => Ok(()),
        Frame::Busy => Err(HubBusy.into()),
        frame => Err(eyre!("unexpected greeting from the hub: {frame:?}")),
    }
}

/// Serialize any frame-like value and write it to the sink.
pub async fn write_frame_to_stream<W: tokio::io::AsyncWriteExt + Unpin, T: serde::Serialize>(
    sink: &mut W,
    frame: &T,
) -> Result<()> {
    let bytes = postcard::to_allocvec(frame).map_err(|e| eyre!(e))?;
    sink.write_all(&bytes).await?;
    Ok(())
}

/// Read a single postcard frame from the stream, buffering as needed.
pub async fn read_frame_from_stream<T: serde::de::DeserializeOwned>(
    stream: &mut tokio::net::UnixStream,
    store: &mut Vec<u8>,
    per_read_timeout: Option<std::time::Duration>,
    total_timeout: Option<std::time::Duration>,
) -> std::result::Result<T, ProtocolError> {
    use tokio::io::AsyncReadExt;
    use tokio::time::Instant;

    let start = Instant::now();
    let per_read_timeout = per_read_timeout.unwrap_or(std::time::Duration::MAX);
    let total_timeout = total_timeout.unwrap_or(std::time::Duration::MAX);
    let mut chunk = [0u8; 4096];

    loop {
        if !store.is_empty() {
            match postcard::take_from_bytes::<T>(&store[..]) {
                Err(postcard::Error::DeserializeUnexpectedEnd) => {
                    // Need more bytes; fall through to the read path below.
                }
                Err(e) => {
                    // Broken transmission; abort
                    return Err(ProtocolError::Decode(e));
                }
                Ok((msg, rest)) => {
                    // Chop off the consumed prefix, keep remainder for next call
                    let consumed = store.len() - rest.len();
                    let _ = store.drain(0..consumed);
                    return Ok(msg);
                }
            }
        }

        if start.elapsed() > total_timeout {
            return Err(ProtocolError::Timeout);
        }

        match tokio::time::timeout(per_read_timeout, AsyncReadExt::read(stream, &mut chunk)).await {
            Err(_) => {
                // per-read timeout, try again
            }
            Ok(Err(e)) => return Err(ProtocolError::Io(e)),
            Ok(Ok(0)) => return Err(ProtocolError::Disconnect),
            Ok(Ok(n)) => store.extend_from_slice(&chunk[..n]),
        }
    }
}

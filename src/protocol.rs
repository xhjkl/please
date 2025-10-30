//! Minimal postcard-framed protocol between CLI and hub.
use eyre::{Result, eyre};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    System(String),
    Developer(String),
    User(String),
    Reasoning(String),
    Tool(String),
    Assistant(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Frame {
    Request { messages: Vec<Message> },
    Log(String),
    Answer(String),
    Stop,
}

#[derive(Debug)]
pub enum ProtocolError {
    Disconnect,
    Timeout,
    Io(std::io::Error),
    Decode(postcard::Error),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::Disconnect => {
                write!(f, "connection dropped while the request was being read")
            }
            ProtocolError::Io(e) => write!(f, "io error: {e}"),
            ProtocolError::Timeout => write!(f, "timed out while reading request"),
            ProtocolError::Decode(e) => write!(f, "decode error: {e}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

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
    use std::time::Instant;
    use tokio::io::AsyncReadExt;

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

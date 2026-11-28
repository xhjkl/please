//! Minimal postcard-framed protocol between probe and hub.
use eyre::{Result, eyre};
use serde::{Deserialize, Serialize};

/// How hard the model should think before answering, as a level on Inkling's scale from 0
/// to 1. Other families of checkpoints hear the nearest effort they have a name for.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Effort(f32);

impl Effort {
    /// What `PLEASE_TRY=light` stands for.
    pub const LOW: Self = Self(0.2);
    /// What `PLEASE_TRY=medium` stands for, and the default.
    pub const MEDIUM: Self = Self(0.7);
    /// What `PLEASE_TRY=hard` stands for.
    pub const HIGH: Self = Self(0.9);

    /// An effort at `level`, if that lies on the scale.
    pub fn new(level: f32) -> Option<Self> {
        (0.0..=1.0).contains(&level).then_some(Self(level))
    }

    /// The level on Inkling's scale.
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub fn level(self) -> f32 {
        self.0
    }
}

/// Facts about this invocation that the hub states to the model; they hold for every message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Setting {
    pub today: String,
    pub effort: Effort,
    /// Where the final answer goes: `None` for a terminal, an empty path for a redirect whose
    /// destination is unknown.
    pub redirection: Option<String>,
}

/// Where a probe runs; the hub keeps one room per place.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Place {
    /// The machine's SSH host key, or empty on a machine without one.
    pub machine: String,
    /// The canonical working directory.
    pub cwd: String,
}

/// One thing said in a conversation, in terms no model's chat format dictates; the hub words
/// and frames it for the loaded checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Message {
    /// What the person asked.
    User(String),
    /// Data the person piped in alongside a request: material, never instructions.
    Input(String),
    /// The harness's word in the middle of a turn.
    Note(Note),
    /// The model's reasoning before it answers or calls tools.
    Thinking(String),
    /// The model's visible answer.
    Answer(String),
    /// A tool call the model made.
    Call(ToolCall),
    /// What a tool returned to the model.
    Result(ToolResult),
}

/// Something the harness tells the model; the hub chooses the words.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Note {
    /// These commands still run; each needs a final result before the answer.
    Pending(Vec<u64>),
    /// An answer arrived while these commands still ran, so it was set aside.
    AnswerSetAside(Vec<u64>),
    /// The context is nearly full: fold the next chunk of the transcript into the summary the
    /// conversation restarts from, as drafted so far (none before the first chunk).
    Summarize {
        draft: Option<String>,
        chunk: String,
    },
    /// The conversation just restarted in the middle of a task: carry on from its summary.
    Resume,
}

/// A named tool invocation with its arguments as a JSON object's text (postcard cannot carry
/// `serde_json::Value`), under an ID unique within its room.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: u64,
    pub name: String,
    pub arguments: String,
}

/// A tool's output as text, JSON for structured results, under the ID of the call it answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: u64,
    pub name: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Frame {
    /// Advance the turn in the place's room by what is new since the hub's last reply: a
    /// request, or tool results; notes hold for this frame only. A retry after a dropped
    /// connection carries the same `id`, so the hub keeps its messages once.
    Advance {
        place: Place,
        setting: Setting,
        id: u64,
        messages: Vec<Message>,
    },
    /// Restart the conversation in the place's room: from a message summing it up, or from
    /// nothing.
    Restart {
        place: Place,
        setting: Setting,
        summarize: bool,
    },
    Log(String),
    /// The summary the conversation restarted from; the probe shows it apart from the answer.
    Restarted(String),
    Answer(String),
    Thinking(String),
    /// A call for the probe to answer with a result under the same ID.
    Call(ToolCall),
    Stop,
    CheckpointRequest,
    /// The checkpoint the hub has loaded.
    Checkpoint {
        path: String,
    },
    Error(String),
    BenchmarkRequest {
        setting: Setting,
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
    #[error("connection dropped while a frame was being read")]
    Disconnect,
    #[error("timed out while reading a frame")]
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

/// Make an ID no other call gets: random, so probes on different machines never collide.
pub fn fresh_id() -> u64 {
    use std::hash::BuildHasher;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let count = NEXT.fetch_add(1, Ordering::Relaxed);
    std::hash::RandomState::new().hash_one((count, std::process::id()))
}

/// This process's name for the hub, fixed for the process, so its reconnections are recognized.
fn probe_id() -> u64 {
    static ID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *ID.get_or_init(fresh_id)
}

/// Introduce this probe on a fresh connection and wait until the hub takes it.
pub async fn greet(stream: &mut tokio::net::UnixStream) -> Result<()> {
    let hello = Frame::Hello { probe: probe_id() };
    write_frame(stream, &hello).await?;
    let mut store = Vec::new();
    let greeting = read_frame(stream, &mut store, None).await?;
    match greeting {
        Frame::Welcome => Ok(()),
        Frame::Busy => Err(HubBusy.into()),
        frame => Err(eyre!("unexpected greeting from the hub: {frame:?}")),
    }
}

/// Write a frame to the sink.
pub async fn write_frame<W: tokio::io::AsyncWriteExt + Unpin>(
    sink: &mut W,
    frame: &Frame,
) -> Result<()> {
    let bytes = postcard::to_allocvec(frame)?;
    sink.write_all(&bytes).await?;
    Ok(())
}

/// Read one frame from the stream, buffering as needed. With `patience`, the whole frame
/// must arrive within it.
pub async fn read_frame(
    stream: &mut tokio::net::UnixStream,
    store: &mut Vec<u8>,
    patience: Option<std::time::Duration>,
) -> std::result::Result<Frame, ProtocolError> {
    use tokio::io::AsyncReadExt;

    let deadline = patience.map(|patience| tokio::time::Instant::now() + patience);
    let mut chunk = [0u8; 4096];

    loop {
        if !store.is_empty() {
            match postcard::take_from_bytes::<Frame>(&store[..]) {
                // Reading on for the rest of a frame split across reads.
                Err(postcard::Error::DeserializeUnexpectedEnd) => {}
                Err(e) => return Err(ProtocolError::Decode(e)),
                Ok((msg, rest)) => {
                    // Keeping the bytes past the frame for the next call.
                    let consumed = store.len() - rest.len();
                    store.drain(..consumed);
                    return Ok(msg);
                }
            }
        }

        let read = stream.read(&mut chunk);
        let read = match deadline {
            Some(deadline) => {
                let read = tokio::time::timeout_at(deadline, read).await;
                read.map_err(|_| ProtocolError::Timeout)?
            }
            None => read.await,
        };
        match read {
            Err(e) => return Err(ProtocolError::Io(e)),
            Ok(0) => return Err(ProtocolError::Disconnect),
            Ok(n) => store.extend_from_slice(&chunk[..n]),
        }
    }
}

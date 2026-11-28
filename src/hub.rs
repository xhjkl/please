//! Own checkpoint selection, loading, prompt framing, inference, and the rooms behind the
//! socket protocol. The probe executes tools; even an embedded hub serves it over a Unix
//! socket pair.
//! Only Apple Silicon builds carry a hub; others are probes for a forwarded one.

/// Why a probe-only build cannot do what needs a local hub.
pub const UNSUPPORTED: &str =
    "local inference requires Apple Silicon macOS; connect to a forwarded please hub";

/// No usable checkpoint is installed; the probe answers it with onboarding.
#[derive(Debug, thiserror::Error)]
#[error("no supported local weights found; run `please load`")]
pub struct NoWeights;

/// Locate please's own directory, `~/.please`, home to the socket, weights, caches, and rooms.
pub fn directory() -> std::path::PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    std::path::PathBuf::from(home).join(".please")
}

/// Locate the hub's Unix socket at `~/.please/socket`.
pub fn socket_path() -> std::path::PathBuf {
    directory().join("socket")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod admission;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod benchmark;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod clipping;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod dialect;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod digest;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod discovery;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod download;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod inference;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod prefix_cache;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod rooms;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod server;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod transcript;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod turn;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod wording;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use benchmark::run as benchmark;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use discovery::checkpoint_paths;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use download::run as download_checkpoint;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use server::{embed, run};

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn run() -> eyre::Result<()> {
    Err(eyre::eyre!(UNSUPPORTED))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn embed() -> eyre::Result<tokio::net::UnixStream> {
    Err(eyre::eyre!(UNSUPPORTED))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn benchmark(_checkpoint: &str, _cases_path: &str) -> eyre::Result<()> {
    Err(eyre::eyre!(UNSUPPORTED))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn download_checkpoint() -> eyre::Result<()> {
    Err(eyre::eyre!(UNSUPPORTED))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub fn checkpoint_paths(
    running: Option<std::path::PathBuf>,
) -> eyre::Result<Vec<std::path::PathBuf>> {
    Ok(running.into_iter().collect())
}

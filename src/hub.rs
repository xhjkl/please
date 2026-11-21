//! Own model selection, loading, framing, and inference behind the socket protocol.
//! The probe executes tools; even an embedded hub serves it over a Unix socket pair.
//! Only Apple Silicon builds carry a hub; others are probes for a forwarded one.

/// Why a probe-only build cannot do what needs a local hub.
pub const UNSUPPORTED: &str =
    "local inference requires Apple Silicon macOS; connect to a forwarded please hub";

/// No usable checkpoint is installed; the probe answers it with onboarding.
#[derive(Debug, thiserror::Error)]
#[error("no supported local weights found; run `please load`")]
pub struct NoWeights;

/// Default UNIX socket location under `~/.please/socket`.
pub fn socket_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    std::path::Path::new(&home).join(".please").join("socket")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod benchmark;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod dialect;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod discovery;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod download;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod inference;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod intuition;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod local;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod occupancy;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod preamble;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use benchmark::run as benchmark;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use discovery::weights_dir;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use download::run_download as load_weights;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use local::{run, spawn};

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn run() -> eyre::Result<()> {
    Err(eyre::eyre!(UNSUPPORTED))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn spawn(_model_path: Option<&str>) -> eyre::Result<tokio::net::UnixStream> {
    Err(eyre::eyre!(UNSUPPORTED))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn benchmark(_model_path: &str, _cases_path: &str) -> eyre::Result<()> {
    Err(eyre::eyre!(UNSUPPORTED))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub async fn load_weights(_which: Option<&str>) -> eyre::Result<()> {
    Err(eyre::eyre!(UNSUPPORTED))
}

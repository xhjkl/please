use eyre::{Result, eyre};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::net::UnixStream;

#[derive(Debug)]
pub enum ConnectError {
    Missing { path: PathBuf },
    PermissionDenied { path: PathBuf },
    NotSocket { path: PathBuf },
    NoListener { path: PathBuf },
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectError::Missing { path } => write!(f, "missing socket: {}", path.display()),
            ConnectError::PermissionDenied { path } => {
                write!(f, "permission denied: {}", path.display())
            }
            ConnectError::NotSocket { path } => write!(f, "not a socket: {}", path.display()),
            ConnectError::NoListener { path } => write!(f, "no listener at: {}", path.display()),
        }
    }
}

impl std::error::Error for ConnectError {}

/// Try to connect to an existing hub via Unix socket.
pub async fn try_connect_to_hub(path: &Path) -> std::result::Result<UnixStream, ConnectError> {
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

/// Spawn the hub process in the background. Does not wait for readiness.
async fn start_hub() -> Result<()> {
    use eyre::eyre;
    let exe = std::env::current_exe().map_err(|e| eyre!(e))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("run");
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());
    let _child = cmd.spawn().map_err(|e| eyre!(e))?;
    Ok(())
}

pub async fn obtain_control_stream() -> Result<UnixStream> {
    let path = crate::hub::socket_path();

    match try_connect_to_hub(&path).await {
        Err(ConnectError::NotSocket { path }) | Err(ConnectError::PermissionDenied { path }) => {
            let path = path.to_string_lossy();
            tracing::error!(
                %path,
                "probe: something is off with the socket",
            );
        }
        Err(ConnectError::NoListener { .. }) | Err(ConnectError::Missing { .. }) => {}
        Ok(stream) => {
            tracing::info!("probe: connected to existing hub at {}", path.display());
            return Ok(stream);
        }
    }

    // Decide how to start the hub when no listener is present.
    // By default, spawn an embedded hub for all OSes.
    // If PLEASE_SPAWN_HUB is set, try to start a detached background hub process instead.
    let prefers_daemon = std::env::var("PLEASE_SPAWN_HUB").is_ok();
    if prefers_daemon {
        start_hub().await?;

        let mut attempts = 0;
        loop {
            attempts += 1;
            match try_connect_to_hub(&path).await {
                Err(ConnectError::NotSocket { path })
                | Err(ConnectError::PermissionDenied { path }) => {
                    return Err(eyre!("probe: not a socket at {}", path.to_string_lossy()));
                }
                Err(ConnectError::NoListener { .. }) | Err(ConnectError::Missing { .. }) => {}
                Ok(stream) => {
                    return Ok(stream);
                }
            }
            if attempts > 3 {
                tracing::warn!(
                    "probe: failed to start or accept connections at {} — falling back to embedded hub",
                    path.display()
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(128)).await;
        }
    }

    let stream = crate::hub::spawn().await?;
    tracing::info!("probe: started embedded hub");
    Ok(stream)
}

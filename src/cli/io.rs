use std::io::{IsTerminal, Read};
use std::os::fd::AsFd;
use std::path::PathBuf;

/// Read piped stdin whole; `None` when stdin is a terminal or carries only whitespace.
pub fn read_whole_stdin() -> std::io::Result<Option<String>> {
    let mut stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Ok(None);
    }
    let mut buf = String::new();
    stdin.read_to_string(&mut buf)?;
    Ok((!buf.trim().is_empty()).then_some(buf))
}

/// Find the path of the regular file stdout is redirected to, if it can be found.
pub fn stdout_redirection_path() -> Option<String> {
    let stdout = std::io::stdout();
    let file = stdout.as_fd().try_clone_to_owned().ok()?;
    let metadata = std::fs::File::from(file).metadata().ok()?;
    // A terminal, a pipe, or a device has no path worth naming.
    if !metadata.is_file() {
        return None;
    }
    let path = stdout_path()?;
    Some(path.to_string_lossy().into_owned())
}

#[cfg(target_os = "macos")]
fn stdout_path() -> Option<PathBuf> {
    use nix::fcntl::{FcntlArg, fcntl};
    let mut path = PathBuf::new();
    fcntl(std::io::stdout(), FcntlArg::F_GETPATH(&mut path)).ok()?;
    path.is_absolute().then_some(path)
}

#[cfg(not(target_os = "macos"))]
fn stdout_path() -> Option<PathBuf> {
    let path = std::fs::read_link("/proc/self/fd/1").ok()?;
    let path = path.to_string_lossy();
    let path = PathBuf::from(path.strip_suffix(" (deleted)").unwrap_or(&path));
    path.is_absolute().then_some(path)
}

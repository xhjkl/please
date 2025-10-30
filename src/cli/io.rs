use eyre::eyre;
use std::io::Read;
use std::path::PathBuf;

#[cfg(unix)]
use std::path::Path;

#[cfg(unix)]
use std::os::fd::BorrowedFd;

#[cfg(unix)]
use nix::unistd::isatty;

#[cfg(target_os = "macos")]
use nix::fcntl::{FcntlArg, fcntl};

fn stdin_is_tty() -> bool {
    atty::is(atty::Stream::Stdin)
}

/// If stdin is not a TTY, read it fully as a single UTF-8 string.
/// Returns `None` when stdin is a TTY or when the input is empty/whitespace.
pub fn read_whole_stdin() -> eyre::Result<Option<String>> {
    if stdin_is_tty() {
        return Ok(None);
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| eyre!(e))?;
    if buf.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(buf))
    }
}

/// Best-effort detection of the file path stdout is redirected to.
/// Returns Some(path) when stdout is a regular file, otherwise None (TTY, pipe, socket, etc.).
#[cfg(unix)]
pub fn stdout_redirection_path() -> Option<String> {
    const FD: i32 = 1; // STDOUT

    // If stdout is a TTY, we are interactive.
    let fd_ref = unsafe { BorrowedFd::borrow_raw(FD) };
    if isatty(fd_ref).ok()? {
        return None;
    }

    // Resolve common fd symlink locations across Unix variants; fall back to F_GETPATH on macOS.
    let path = match try_readlink_fd(FD) {
        Some(p) => p,
        None => {
            #[cfg(target_os = "macos")]
            {
                try_fcntl_getpath(FD)?
            }
            #[cfg(not(target_os = "macos"))]
            {
                return None;
            }
        }
    };

    // Prefer returning the absolute path even if metadata fails (e.g., unlinked after open).
    // If metadata is available and confirms a regular file, that's ideal; otherwise still return.
    let _ = std::fs::metadata(&path).ok();
    Some(path.to_string_lossy().to_string())
}

#[cfg(unix)]
fn try_readlink_fd(fd: i32) -> Option<PathBuf> {
    let candidates = [
        format!("/proc/self/fd/{fd}"),    // Linux
        format!("/proc/curproc/fd/{fd}"), // *BSD variants
        format!("/dev/fd/{fd}"),          // macOS/*BSD
    ];
    for candidate in candidates {
        if let Ok(target) = std::fs::read_link(Path::new(&candidate)) {
            let s = target.to_string_lossy();
            let cleaned = s.strip_suffix(" (deleted)").unwrap_or(&s);
            let pb = PathBuf::from(cleaned);
            if pb.is_absolute() {
                return Some(pb);
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn try_fcntl_getpath(fd: i32) -> Option<PathBuf> {
    let fd_ref = unsafe { BorrowedFd::borrow_raw(fd) };
    let mut pb = PathBuf::new();
    let _ = fcntl(fd_ref, FcntlArg::F_GETPATH(&mut pb)).ok()?;
    if pb.is_absolute() { Some(pb) } else { None }
}

#[cfg(not(unix))]
pub fn stdout_redirection_path() -> Option<String> {
    None
}

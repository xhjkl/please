use std::fs::OpenOptions;
use std::io::{self, Read};
use std::os::unix::fs::OpenOptionsExt;
use tokio::io::{Interest, unix::AsyncFd};

/// Read canonical terminal input without leaving a blocking stdin read on cancellation.
pub async fn read_terminal_line() -> io::Result<String> {
    // kqueue needs the terminal device itself, since `/dev/tty` cannot be registered on macOS.
    let device = nix::unistd::ttyname(io::stdin())?;
    let tty = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOCTTY)
        .open(device)?;
    // `File` owns a stable descriptor until Tokio deregisters and drops it.
    let tty = unsafe { AsyncFd::register_with_interest(tty, Interest::READABLE) }?;
    let mut bytes = [0; 4096];
    let read = tty
        .async_io(Interest::READABLE, |mut tty| tty.read(&mut bytes))
        .await?;
    Ok(String::from_utf8_lossy(&bytes[..read]).into_owned())
}

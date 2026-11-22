//! Where this probe runs, which names its conversation on the hub.
use crate::protocol::Place;
use eyre::Result;

/// This host, working directory, and terminal session.
pub fn here() -> Result<Place> {
    let host = nix::unistd::gethostname()?.to_string_lossy().into_owned();
    let cwd = std::env::current_dir()?.canonicalize()?;
    Ok(Place {
        host,
        cwd: cwd.to_string_lossy().into_owned(),
        session: session(),
    })
}

/// The terminal session: its leader (usually the login shell, or `login` itself) and when
/// that started, so a later session reusing the process ID is another conversation. Without a
/// start time the leader alone names it.
fn session() -> String {
    let leader = nix::unistd::getsid(None).map_or(0, |pid| pid.as_raw());
    match started(leader) {
        Some(started) => format!("{leader}@{started}"),
        None => leader.to_string(),
    }
}

/// When process `pid` started, in microseconds since the epoch. The kernel's process table
/// answers for any owner's process, as a session led by root's `login` needs.
#[cfg(target_os = "macos")]
fn started(pid: i32) -> Option<String> {
    use nix::libc;
    // `struct kinfo_proc` opens with `kp_proc.p_starttime`, a `struct timeval` (<sys/sysctl.h>,
    // <sys/proc.h>); the record is 648 bytes on 64-bit macOS, and the buffer has room to spare.
    let mut record = [0u64; 128];
    let mut size = std::mem::size_of_val(&record);
    let mut mib = [libc::CTL_KERN, libc::KERN_PROC, libc::KERN_PROC_PID, pid];
    // SAFETY: sysctl writes at most `size` bytes into the buffer.
    let read = unsafe {
        let (name, length) = (mib.as_mut_ptr(), mib.len() as u32);
        let null = std::ptr::null_mut();
        libc::sysctl(name, length, record.as_mut_ptr().cast(), &mut size, null, 0)
    };
    let (seconds, micros) = (record[0], record[1] & 0xffff_ffff);
    (read == 0 && size >= 16 && seconds > 0).then(|| (seconds * 1_000_000 + micros).to_string())
}

/// When process `pid` started, in clock ticks after this boot, which the boot's ID names.
#[cfg(target_os = "linux")]
fn started(pid: i32) -> Option<String> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The start time is the 22nd field; the command name before it may hold spaces.
    let ticks = stat
        .rsplit_once(')')?
        .1
        .split_whitespace()
        .nth(19)?
        .to_string();
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap_or_default();
    Some(format!("{ticks}/{}", boot.trim()))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn started(_pid: i32) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_session_has_a_stable_identity() -> Result<()> {
        assert_eq!(session(), session());
        assert!(session().contains('@'), "{}", session());
        let place = here()?;
        assert!(!place.host.is_empty() && place.cwd.starts_with('/'));
        Ok(())
    }
}

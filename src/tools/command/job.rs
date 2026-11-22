use nix::errno::Errno;
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;
use std::collections::VecDeque;
use std::io;
use std::mem::MaybeUninit;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::signal::unix::{Signal as SignalReceiver, SignalKind, signal};
use tokio::sync::{mpsc::UnboundedSender, watch};

const INTERRUPT_GRACE: Duration = Duration::from_secs(3);
const DRAIN_GRACE: Duration = Duration::from_secs(1);
pub const STOP_WAIT: Duration = INTERRUPT_GRACE
    .saturating_add(DRAIN_GRACE)
    .saturating_add(Duration::from_secs(1));
const MAX_CAPTURE_BYTES: usize = 256 * 1024;
const MAX_LIVE_BYTES: usize = 1024 * 1024;

/// The registry's handle for a process supervised independently of tool waits.
pub struct Job {
    pid: u32,
    started: Instant,
    state: Arc<Mutex<State>>,
    stop: watch::Sender<Option<StopRequest>>,
    monitor: watch::Receiver<()>,
}

#[derive(Clone, Copy)]
enum StopRequest {
    Interrupt,
    Force,
}

struct State {
    stdout: Pipe,
    stderr: Pipe,
    completion: Option<Completion>,
}

/// Output drained together with the status observed at the same point.
pub struct Observation {
    pub pid: u32,
    pub elapsed: Duration,
    pub stdout: Chunk,
    pub stderr: Chunk,
    pub completion: Option<Completion>,
}

#[derive(Clone)]
pub struct Completion {
    pub exit: Result<Exit, String>,
    pub output_error: Option<String>,
    elapsed: Duration,
}

#[derive(Clone, Copy)]
pub struct Exit {
    pub status: ExitStatus,
    pub cause: ExitCause,
}

#[derive(Clone, Copy)]
pub enum ExitCause {
    Natural,
    Interrupted,
    Escalated,
    Forced,
}

impl Job {
    pub fn spawn(argv: &[String], live: Option<&UnboundedSender<String>>) -> io::Result<Self> {
        let process = Process::spawn(argv)?;
        let pid = process.group.0.as_raw() as u32;
        let started = Instant::now();
        let state = Arc::new(Mutex::new(State {
            stdout: Pipe::new(live),
            stderr: Pipe::new(live),
            completion: None,
        }));
        let (stop, requested) = watch::channel(None);
        let (finished, monitor) = watch::channel(());
        let capture = state.clone();
        tokio::spawn(async move {
            let completion = process.run(started, requested, &capture).await;
            capture.lock().unwrap().completion = Some(completion);
            drop(finished);
        });
        Ok(Self {
            pid,
            started,
            state,
            stop,
            monitor,
        })
    }

    pub fn interrupt(&self) {
        self.stop.send_if_modified(|request| {
            if request.is_some() {
                return false;
            }
            *request = Some(StopRequest::Interrupt);
            true
        });
    }

    pub fn force_stop(&self) {
        self.stop.send_replace(Some(StopRequest::Force));
    }

    /// Stream the unread tail and subsequent bytes during a control call.
    pub fn stream_to(&self, sender: &UnboundedSender<String>) {
        let mut state = self.state.lock().unwrap();
        state.stdout.stream_to(sender);
        state.stderr.stream_to(sender);
    }

    pub async fn wait(&self) {
        // The monitor only publishes into `state`; closing this channel wakes every waiter.
        let _ = self.monitor.clone().changed().await;
    }

    pub fn observe(&self) -> Observation {
        let mut state = self.state.lock().unwrap();
        let completion = match &state.completion {
            Some(completion) => Some(completion.clone()),
            None if self.monitor.has_changed().is_err() => Some(Completion {
                exit: Err("command monitor exited unexpectedly".into()),
                output_error: None,
                elapsed: self.started.elapsed(),
            }),
            None => None,
        };
        let (stdout, stderr, elapsed) = match &completion {
            Some(completion) => (
                state.stdout.final_chunk(),
                state.stderr.final_chunk(),
                completion.elapsed,
            ),
            None => (
                state.stdout.chunk(),
                state.stderr.chunk(),
                self.started.elapsed(),
            ),
        };
        Observation {
            pid: self.pid,
            elapsed,
            stdout,
            stderr,
            completion,
        }
    }
}

#[derive(Default)]
struct Pipe {
    bytes: VecDeque<u8>,
    omitted: usize,
    live: Option<Live>,
}

pub struct Chunk {
    pub text: String,
    pub omitted: usize,
}

impl Pipe {
    fn new(sender: Option<&UnboundedSender<String>>) -> Self {
        let live = sender.map(Live::new);
        Self {
            live,
            ..Self::default()
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        // The pipe reads at most 4 KiB at a time.
        let omitted = (self.bytes.len() + bytes.len()).saturating_sub(MAX_CAPTURE_BYTES);
        self.omitted += omitted;
        self.bytes.drain(..omitted);
        self.bytes.extend(bytes);
        if let Some(live) = self.live.take() {
            self.live = live.push(bytes);
        }
    }

    fn stream_to(&mut self, sender: &UnboundedSender<String>) {
        let bytes = self.bytes.make_contiguous();
        self.live = Live::new(sender).push(bytes);
    }

    fn chunk(&mut self) -> Chunk {
        // Ending live delivery under the same lock that drains the first tool result.
        self.live = None;
        let end = text_boundary(self.bytes.make_contiguous());
        self.drain(end)
    }

    fn final_chunk(&mut self) -> Chunk {
        self.finish_live();
        self.drain(self.bytes.len())
    }

    fn drain(&mut self, end: usize) -> Chunk {
        let bytes = self.bytes.make_contiguous();
        let text = String::from_utf8_lossy(&bytes[..end]).into_owned();
        self.bytes.drain(..end);
        let omitted = std::mem::take(&mut self.omitted);
        Chunk { text, omitted }
    }

    fn finish_live(&mut self) {
        if let Some(live) = self.live.take()
            && !live.pending.is_empty()
        {
            let _ = live
                .sender
                .send(String::from_utf8_lossy(&live.pending).into_owned());
        }
    }
}

struct Live {
    sender: UnboundedSender<String>,
    remaining: usize,
    pending: Vec<u8>,
}

impl Live {
    fn new(sender: &UnboundedSender<String>) -> Self {
        Self {
            sender: sender.clone(),
            remaining: MAX_LIVE_BYTES,
            pending: Vec::new(),
        }
    }

    fn push(mut self, bytes: &[u8]) -> Option<Self> {
        if self.sender.is_closed() {
            return None;
        }
        let kept = self.remaining.min(bytes.len());
        self.remaining -= kept;
        self.pending.extend_from_slice(&bytes[..kept]);
        let end = text_boundary(&self.pending);
        if end > 0 {
            let chunk = String::from_utf8_lossy(&self.pending[..end]).into_owned();
            self.sender.send(chunk).ok()?;
            self.pending.drain(..end);
        }
        if kept < bytes.len() {
            let _ = self
                .sender
                .send("\n[please: further live command output omitted]\n".into());
            return None;
        }
        Some(self)
    }
}

// Holding a partial UTF-8 character until its next read or the end of the stream.
fn text_boundary(bytes: &[u8]) -> usize {
    let tail_start = bytes.len().saturating_sub(3);
    let tail = &bytes[tail_start..];
    let start = tail.iter().rposition(|byte| byte & 0xc0 != 0x80);
    if let Some(start) = start
        && std::str::from_utf8(&tail[start..]).is_err_and(|error| error.error_len().is_none())
    {
        return tail_start + start;
    }
    bytes.len()
}

async fn read_stream(
    mut reader: impl AsyncRead + Unpin,
    state: &Mutex<State>,
    pipe: fn(&mut State) -> &mut Pipe,
) -> io::Result<()> {
    let mut buf = [0u8; 4096];
    loop {
        let read = reader.read(&mut buf).await;
        let mut state = state.lock().unwrap();
        let pipe = pipe(&mut state);
        match read {
            Err(error) => {
                pipe.finish_live();
                return Err(error);
            }
            Ok(0) => {
                pipe.finish_live();
                return Ok(());
            }
            Ok(read) => pipe.push(&buf[..read]),
        }
    }
}

/// One command's process group. The leader stays unreaped until the group is signalled.
struct ProcessGroup(Pid);

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        let killed = killpg(self.0, Signal::SIGKILL);
        if let Err(error) = killed
            && error != Errno::ESRCH
        {
            tracing::warn!(pid = %self.0, %error, "could not kill command group");
        }
    }
}

struct Process {
    // Dropping the group first keeps its PID reserved by `child` during cancellation.
    group: ProcessGroup,
    child: Child,
    child_signal: SignalReceiver,
}

impl Process {
    fn spawn(argv: &[String]) -> io::Result<Self> {
        let child_signal = signal(SignalKind::child())?;
        let mut command = Command::new(&argv[0]);
        command
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .kill_on_drop(true);
        let child = command.spawn()?;
        let pid = child.id().expect("new child has a pid");
        let group = ProcessGroup(Pid::from_raw(pid as i32));
        Ok(Self {
            group,
            child,
            child_signal,
        })
    }

    /// Drain the two pipes concurrently while supervising the group, even between tool calls.
    async fn run(
        mut self,
        started: Instant,
        requested: watch::Receiver<Option<StopRequest>>,
        output: &Mutex<State>,
    ) -> Completion {
        let stdout = self.child.stdout.take().expect("stdout is piped");
        let stderr = self.child.stderr.take().expect("stderr is piped");
        let exit = self.finish(requested);
        let capture = async {
            tokio::join!(
                read_stream(stdout, output, |state| &mut state.stdout),
                read_stream(stderr, output, |state| &mut state.stderr),
            )
        };
        tokio::pin!(exit, capture);
        let (exit, captured) = tokio::select! {
            exit = &mut exit => {
                let captured = tokio::time::timeout(DRAIN_GRACE, &mut capture).await;
                (exit, captured.ok())
            }
            captured = &mut capture => (exit.await, Some(captured)),
        };
        let output_error = match captured {
            None => Some("timed out draining command output".into()),
            Some((Ok(()), Ok(()))) => None,
            Some((Err(stdout), Ok(()))) => Some(format!("stdout: {stdout}")),
            Some((Ok(()), Err(stderr))) => Some(format!("stderr: {stderr}")),
            Some((Err(stdout), Err(stderr))) => Some(format!("stdout: {stdout}; stderr: {stderr}")),
        };
        Completion {
            exit: exit.map_err(|error| error.to_string()),
            output_error,
            elapsed: started.elapsed(),
        }
    }

    async fn finish(mut self, requested: watch::Receiver<Option<StopRequest>>) -> io::Result<Exit> {
        let cause = wait_for_stop(self.group.0, &mut self.child_signal, requested).await;
        drop(self.group);
        let status = self.child.wait().await?;
        Ok(Exit {
            status,
            cause: cause?,
        })
    }
}

async fn wait_for_stop(
    pid: Pid,
    child_signal: &mut SignalReceiver,
    mut requested: watch::Receiver<Option<StopRequest>>,
) -> io::Result<ExitCause> {
    let exited = wait_for_exit(pid, child_signal);
    tokio::pin!(exited);
    let stop = tokio::select! {
        biased;
        status = &mut exited => return status.map(|()| ExitCause::Natural),
        stop = requested.wait_for(Option::is_some) => stop.ok().and_then(|stop| *stop),
    };
    // Losing the last owner force-stops the process without waiting for SIGINT.
    if !matches!(stop, Some(StopRequest::Interrupt)) {
        return Ok(ExitCause::Forced);
    }

    let interrupted = killpg(pid, Signal::SIGINT);
    if let Err(error) = interrupted
        && error != Errno::ESRCH
    {
        return Err(error.into());
    }
    tokio::select! {
        biased;
        status = &mut exited => status.map(|()| ExitCause::Interrupted),
        _ = requested.wait_for(|stop| matches!(stop, Some(StopRequest::Force))) => Ok(ExitCause::Forced),
        _ = tokio::time::sleep(INTERRUPT_GRACE) => Ok(ExitCause::Escalated),
    }
}

async fn wait_for_exit(pid: Pid, child_signal: &mut SignalReceiver) -> io::Result<()> {
    loop {
        let exited = has_exited(pid)?;
        if exited {
            return Ok(());
        }
        child_signal.recv().await;
    }
}

fn has_exited(pid: Pid) -> io::Result<bool> {
    use nix::libc;

    loop {
        let mut info = MaybeUninit::<libc::siginfo_t>::zeroed();
        // Checking without reaping keeps the PID reserved until group cleanup. `waitid` writes
        // the child's status into `info`; a zero `si_pid` means no exit is ready.
        let waited = unsafe {
            libc::waitid(
                libc::P_PID,
                pid.as_raw() as libc::id_t,
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if waited == 0 {
            return Ok(unsafe { info.assume_init().si_pid() } != 0);
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    #[tokio::test]
    async fn observing_a_running_job_ends_live_delivery_at_the_chunk_boundary() {
        let directory = TempDir::new().unwrap();
        let release = directory.0.join("more-output");
        let argv = [
            "sh".into(),
            "-c".into(),
            "printf first; while [ ! -e \"$1\" ]; do sleep 0.01; done; printf second".into(),
            "sh".into(),
            release.to_str().unwrap().into(),
        ];
        let (live, mut received) = tokio::sync::mpsc::unbounded_channel();
        let job = Job::spawn(&argv, Some(&live)).unwrap();
        drop(live);

        let first = tokio::time::timeout(Duration::from_secs(5), received.recv())
            .await
            .unwrap();
        assert_eq!(first.as_deref(), Some("first"));
        let first = job.observe();
        assert_eq!(first.stdout.text, "first");
        assert!(first.completion.is_none());
        assert_eq!(
            received.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        );

        let (live, mut received) = tokio::sync::mpsc::unbounded_channel();
        job.stream_to(&live);
        drop(live);
        std::fs::write(release, []).unwrap();
        tokio::time::timeout(Duration::from_secs(5), job.wait())
            .await
            .unwrap();
        let finished = job.observe();
        assert_eq!(finished.stdout.text, "second");
        assert!(finished.completion.unwrap().exit.unwrap().status.success());
        assert_eq!(received.try_recv().unwrap(), "second");
        assert_eq!(
            received.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        );
    }
}

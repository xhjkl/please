use super::common::{Param, ParamType};
use serde::Deserialize;
use serde_json::json;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(40);
const INTERRUPT_GRACE: Duration = Duration::from_secs(3);
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_secs(1);
const MAX_CAPTURE_BYTES: usize = 256 * 1024;
const MAX_LIVE_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
pub struct Args {
    /// Argument vector: first element is the program, followed by args.
    argv: Vec<String>,
}

#[derive(Debug, Default, Clone)]
struct CapturedOutput {
    bytes: Vec<u8>,
    omitted: usize,
}

impl CapturedOutput {
    fn push(&mut self, chunk: &[u8]) {
        let remaining = MAX_CAPTURE_BYTES.saturating_sub(self.bytes.len());
        let kept = remaining.min(chunk.len());
        self.bytes.extend_from_slice(&chunk[..kept]);
        self.omitted += chunk.len() - kept;
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).to_string()
    }
}

type SharedOutput = Arc<Mutex<CapturedOutput>>;

enum CommandEnd {
    Finished {
        status: ExitStatus,
    },
    Interrupted {
        status: Option<ExitStatus>,
        killed: bool,
    },
}

async fn read_stream<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    output: SharedOutput,
    sink: Option<UnboundedSender<String>>,
) {
    let mut live_sent = 0usize;
    let mut live_notice_sent = false;
    let mut buf = [0u8; 4096];

    loop {
        let n = reader.read(&mut buf).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        let chunk = &buf[..n];

        if let Ok(mut output) = output.lock() {
            output.push(chunk);
        }

        if let Some(tx) = sink.as_ref() {
            let live_remaining = MAX_LIVE_BYTES.saturating_sub(live_sent);
            let live_kept = live_remaining.min(n);
            if live_kept > 0 {
                let _ = tx.send(String::from_utf8_lossy(&chunk[..live_kept]).to_string());
                live_sent += live_kept;
            }
            if live_kept < n && !live_notice_sent {
                let _ = tx.send("\n[please: further live command output omitted]\n".to_string());
                live_notice_sent = true;
            }
        }
    }
}

async fn wait_with_timeout(child: &mut tokio::process::Child) -> std::io::Result<CommandEnd> {
    let timeout = tokio::time::sleep(COMMAND_TIMEOUT);
    tokio::pin!(timeout);

    tokio::select! {
        status = child.wait() => status.map(|status| CommandEnd::Finished { status }),
        _ = &mut timeout => {
            interrupt_child(child);
            let grace = tokio::time::sleep(INTERRUPT_GRACE);
            tokio::pin!(grace);
            tokio::select! {
                status = child.wait() => status.map(|status| CommandEnd::Interrupted {
                    status: Some(status),
                    killed: false,
                }),
                _ = &mut grace => {
                    kill_child(child);
                    let status = child.wait().await.ok();
                    Ok(CommandEnd::Interrupted {
                        status,
                        killed: true,
                    })
                }
            }
        }
    }
}

#[cfg(unix)]
fn signal_child_group(child: &tokio::process::Child, signal: nix::sys::signal::Signal) {
    let Some(pid) = child.id() else {
        return;
    };
    let pid = nix::unistd::Pid::from_raw(-(pid as i32));
    let _ = nix::sys::signal::kill(pid, signal);
}

#[cfg(unix)]
fn interrupt_child(child: &tokio::process::Child) {
    signal_child_group(child, nix::sys::signal::Signal::SIGINT);
}

#[cfg(not(unix))]
fn interrupt_child(child: &mut tokio::process::Child) {
    let _ = child.start_kill();
}

#[cfg(unix)]
fn kill_child(child: &mut tokio::process::Child) {
    signal_child_group(child, nix::sys::signal::Signal::SIGKILL);
    let _ = child.start_kill();
}

#[cfg(not(unix))]
fn kill_child(child: &mut tokio::process::Child) {
    let _ = child.start_kill();
}

#[cfg(unix)]
fn kill_child_group_by_pid(pid: Option<u32>) {
    let Some(pid) = pid else {
        return;
    };
    let pid = nix::unistd::Pid::from_raw(-(pid as i32));
    let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGKILL);
}

#[cfg(not(unix))]
fn kill_child_group_by_pid(_pid: Option<u32>) {}

async fn drain_or_abort_readers(
    mut stdout_task: JoinHandle<()>,
    mut stderr_task: JoinHandle<()>,
    kill_group_if_stuck: Option<u32>,
) {
    let drain_grace = tokio::time::sleep(OUTPUT_DRAIN_GRACE);
    tokio::pin!(drain_grace);
    let mut stdout_done = false;
    let mut stderr_done = false;

    loop {
        if stdout_done && stderr_done {
            return;
        }
        tokio::select! {
            _ = &mut stdout_task, if !stdout_done => {
                stdout_done = true;
            }
            _ = &mut stderr_task, if !stderr_done => {
                stderr_done = true;
            }
            _ = &mut drain_grace => {
                kill_child_group_by_pid(kill_group_if_stuck);
                if !stdout_done {
                    stdout_task.abort();
                }
                if !stderr_done {
                    stderr_task.abort();
                }
                return;
            }
        }
    }
}

fn snapshot_output(output: &SharedOutput) -> CapturedOutput {
    output.lock().map(|output| output.clone()).unwrap_or_default()
}

/// Run a command and optionally stream bounded stdout/stderr chunks to `sink`.
/// The returned JSON includes bounded stdout/stderr plus omitted byte counters.
pub async fn call(args: Args, sink: Option<UnboundedSender<String>>) -> serde_json::Value {
    if args.argv.is_empty() {
        return json!({ "error": "argv must be non-empty" });
    }

    let started = Instant::now();
    let mut cmd = tokio::process::Command::new(&args.argv[0]);
    if args.argv.len() > 1 {
        cmd.args(&args.argv[1..]);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => return json!({ "error": error.to_string() }),
    };
    let child_pid = child.id();

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_output = SharedOutput::default();
    let stderr_output = SharedOutput::default();
    let stdout_sink = sink.clone();
    let stdout_for_task = stdout_output.clone();
    let stdout_task = tokio::spawn(async move {
        if let Some(stdout) = stdout_pipe {
            read_stream(stdout, stdout_for_task, stdout_sink).await;
        }
    });
    let stderr_for_task = stderr_output.clone();
    let stderr_task = tokio::spawn(async move {
        if let Some(stderr) = stderr_pipe {
            read_stream(stderr, stderr_for_task, sink).await;
        }
    });

    let end = match wait_with_timeout(&mut child).await {
        Ok(end) => end,
        Err(error) => return json!({ "error": error.to_string() }),
    };
    let kill_group_if_stuck = match &end {
        CommandEnd::Interrupted { .. } => child_pid,
        CommandEnd::Finished { .. } => None,
    };
    drain_or_abort_readers(stdout_task, stderr_task, kill_group_if_stuck).await;
    let stdout = snapshot_output(&stdout_output);
    let stderr = snapshot_output(&stderr_output);

    let running_for_seconds = format!("{:.1}s", started.elapsed().as_secs_f64());
    let mut output = json!({
        "runningFor": running_for_seconds,
        "stdout": stdout.text(),
        "stdoutBytesOmitted": stdout.omitted,
        "stderr": stderr.text(),
        "stderrBytesOmitted": stderr.omitted,
    });

    {
        let output = output
            .as_object_mut()
            .expect("run_command output starts as an object");
        match end {
            CommandEnd::Finished { status } => {
                output.insert("ok".to_string(), json!(status.success()));
                output.insert("status".to_string(), json!("finished"));
                output.insert("exitCode".to_string(), json!(status.code()));
            }
            CommandEnd::Interrupted { status, killed } => {
                output.insert("ok".to_string(), json!(false));
                output.insert("status".to_string(), json!("interrupted"));
                output.insert(
                    "timeout".to_string(),
                    json!({
                        "after": format!("{}s", COMMAND_TIMEOUT.as_secs()),
                        "signal": "SIGINT",
                        "killedAfterGrace": killed,
                    }),
                );
                if let Some(status) = status {
                    output.insert("exitCode".to_string(), json!(status.code()));
                }
            }
        }
    }

    output
}

pub fn spec() -> (&'static str, &'static str, Vec<Param>) {
    (
        "run_command",
        "Run a command by argv; output is capped and long commands are interrupted after 40s",
        vec![Param {
            name: "argv",
            desc: "Argument vector: [program, ...args]",
            param_type: ParamType::String,
            required: true,
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn captures_command_output_with_shape_for_history() {
        let result = call(
            Args {
                argv: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "printf hello; printf problem >&2".to_string(),
                ],
            },
            None,
        )
        .await;

        assert_eq!(result["status"], "finished");
        assert_eq!(result["exitCode"], 0);
        assert_eq!(result["stdout"], "hello");
        assert_eq!(result["stderr"], "problem");
        assert_eq!(result["stdoutBytesOmitted"], 0);
        assert_eq!(result["stderrBytesOmitted"], 0);
    }
}

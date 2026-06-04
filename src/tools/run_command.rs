use super::common::{Param, ParamType, Stride};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

const DEFAULT_COMMAND_WAIT: Duration = Duration::from_secs(40);
const INTERRUPT_GRACE: Duration = Duration::from_secs(3);
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_secs(1);
const MAX_CAPTURE_BYTES: usize = 256 * 1024;
const MAX_LIVE_BYTES: usize = 1024 * 1024;
pub const NAME: &str = "run_command";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    /// Argument vector for a new command.
    argv: Vec<String>,
    /// Seconds to wait before returning control to the model.
    #[serde(default)]
    wait_seconds: Option<f64>,
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

/// Child process kept alive across model subturns.
struct RunningCommand {
    started: Instant,
    pid: u32,
    child: tokio::process::Child,
    stdout_output: SharedOutput,
    stderr_output: SharedOutput,
    stdout_task: JoinHandle<()>,
    stderr_task: JoinHandle<()>,
}

#[derive(Default)]
pub(super) struct RunningCommands {
    commands: AsyncMutex<HashMap<u32, RunningCommand>>,
}

impl RunningCommands {
    /// Pids for subprocesses still owned by this turn.
    pub(super) async fn pids(&self) -> Vec<u32> {
        let commands = self.commands.lock().await;
        let mut pids = commands.keys().copied().collect::<Vec<_>>();
        pids.sort_unstable();
        pids
    }

    pub(super) async fn kill_all(&self) {
        let commands = std::mem::take(&mut *self.commands.lock().await);
        for (_pid, mut command) in commands {
            kill_child(&mut command.child);
            let _ = command.child.wait().await;
            command.stdout_task.abort();
            command.stderr_task.abort();
        }
    }
}

enum CommandEnd {
    Finished {
        status: ExitStatus,
    },
    Running {
        pid: u32,
    },
    Killed {
        status: Option<ExitStatus>,
        killed: bool,
    },
}

async fn read_stream<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    output: SharedOutput,
    live_output: Option<UnboundedSender<String>>,
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

        if let Some(tx) = live_output.as_ref() {
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

async fn wait_for_exit(
    child: &mut tokio::process::Child,
    duration: Duration,
) -> std::io::Result<Option<ExitStatus>> {
    let started = Instant::now();

    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }

        let elapsed = started.elapsed();
        if elapsed >= duration {
            return Ok(None);
        }

        let remaining = duration.saturating_sub(elapsed);
        tokio::time::sleep(remaining.min(Duration::from_millis(50))).await;
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
    output
        .lock()
        .map(|output| output.clone())
        .unwrap_or_default()
}

fn wait_duration(wait_seconds: Option<f64>) -> Result<Duration, String> {
    let Some(wait_seconds) = wait_seconds else {
        return Ok(DEFAULT_COMMAND_WAIT);
    };
    if !wait_seconds.is_finite() || wait_seconds < 0.0 {
        return Err("waitSeconds must be a finite non-negative number".to_string());
    }
    Duration::try_from_secs_f64(wait_seconds)
        .map_err(|_| "waitSeconds is too large to represent".to_string())
}

fn command_output(
    started: Instant,
    stdout: CapturedOutput,
    stderr: CapturedOutput,
) -> serde_json::Value {
    json!({
        "runningFor": format!("{:.1}s", started.elapsed().as_secs_f64()),
        "stdout": stdout.text(),
        "stdoutBytesOmitted": stdout.omitted,
        "stderr": stderr.text(),
        "stderrBytesOmitted": stderr.omitted,
    })
}

fn command_result(
    started: Instant,
    stdout: CapturedOutput,
    stderr: CapturedOutput,
    end: CommandEnd,
) -> serde_json::Value {
    let mut output = command_output(started, stdout, stderr);

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
            CommandEnd::Running { pid } => {
                output.insert("ok".to_string(), json!(false));
                output.insert("status".to_string(), json!("running"));
                output.insert("pid".to_string(), json!(pid));
                output.insert(
                    "next".to_string(),
                    json!("call control_command with action=\"wait\" and this pid to wait longer, or action=\"kill\" and this pid to stop it"),
                );
            }
            CommandEnd::Killed { status, killed } => {
                output.insert("ok".to_string(), json!(false));
                output.insert("status".to_string(), json!("killed"));
                output.insert(
                    "kill".to_string(),
                    json!({
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

async fn finish_command(command: RunningCommand, end: CommandEnd) -> serde_json::Value {
    let kill_group_if_stuck = match &end {
        CommandEnd::Killed { .. } => Some(command.pid),
        CommandEnd::Finished { .. } | CommandEnd::Running { .. } => None,
    };
    drain_or_abort_readers(
        command.stdout_task,
        command.stderr_task,
        kill_group_if_stuck,
    )
    .await;
    let stdout = snapshot_output(&command.stdout_output);
    let stderr = snapshot_output(&command.stderr_output);
    command_result(command.started, stdout, stderr, end)
}

fn running_command_result(command: &RunningCommand) -> serde_json::Value {
    let stdout = snapshot_output(&command.stdout_output);
    let stderr = snapshot_output(&command.stderr_output);
    command_result(
        command.started,
        stdout,
        stderr,
        CommandEnd::Running { pid: command.pid },
    )
}

async fn spawn_command(
    argv: &[String],
    live_output: Option<UnboundedSender<String>>,
) -> std::io::Result<RunningCommand> {
    let mut cmd = tokio::process::Command::new(&argv[0]);
    if argv.len() > 1 {
        cmd.args(&argv[1..]);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd.spawn()?;
    let Some(pid) = child.id() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "spawned command did not expose a pid",
        ));
    };

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_output = SharedOutput::default();
    let stderr_output = SharedOutput::default();
    let stdout_live_output = live_output.clone();
    let stdout_for_task = stdout_output.clone();
    let stdout_task = tokio::spawn(async move {
        if let Some(stdout) = stdout_pipe {
            read_stream(stdout, stdout_for_task, stdout_live_output).await;
        }
    });
    let stderr_for_task = stderr_output.clone();
    let stderr_task = tokio::spawn(async move {
        if let Some(stderr) = stderr_pipe {
            read_stream(stderr, stderr_for_task, live_output).await;
        }
    });

    Ok(RunningCommand {
        started: Instant::now(),
        pid,
        child,
        stdout_output,
        stderr_output,
        stdout_task,
        stderr_task,
    })
}

async fn start_command(
    argv: Vec<String>,
    wait_for: Duration,
    commands: Arc<RunningCommands>,
    live_output: Option<UnboundedSender<String>>,
) -> serde_json::Value {
    if argv.is_empty() {
        return json!({ "error": "argv must be non-empty" });
    }

    let mut command = match spawn_command(&argv, live_output).await {
        Ok(command) => command,
        Err(error) => return json!({ "error": error.to_string() }),
    };

    let status = match wait_for_exit(&mut command.child, wait_for).await {
        Ok(status) => status,
        Err(error) => return json!({ "error": error.to_string() }),
    };
    if let Some(status) = status {
        return finish_command(command, CommandEnd::Finished { status }).await;
    }

    let pid = command.pid;
    let output = running_command_result(&command);
    commands.commands.lock().await.insert(pid, command);
    output
}

/// Keep a carried subprocess moving for another wait slice.
pub(super) async fn wait_by_pid(
    pid: u32,
    wait_seconds: Option<f64>,
    stride: Stride,
) -> serde_json::Value {
    let wait_for = match wait_duration(wait_seconds) {
        Ok(wait_for) => wait_for,
        Err(error) => return json!({ "error": error }),
    };
    let commands = stride.running_commands();
    let command = commands.commands.lock().await.remove(&pid);
    let Some(mut command) = command else {
        return json!({ "error": format!("unknown pid `{pid}`") });
    };

    let status = match wait_for_exit(&mut command.child, wait_for).await {
        Ok(status) => status,
        Err(error) => return json!({ "error": error.to_string() }),
    };
    if let Some(status) = status {
        return finish_command(command, CommandEnd::Finished { status }).await;
    }

    let output = running_command_result(&command);
    commands.commands.lock().await.insert(pid, command);
    output
}

/// Interrupt a carried subprocess, killing after grace.
pub(super) async fn kill_by_pid(pid: u32, stride: Stride) -> serde_json::Value {
    let commands = stride.running_commands();
    let command = commands.commands.lock().await.remove(&pid);
    let Some(mut command) = command else {
        return json!({ "error": format!("unknown pid `{pid}`") });
    };

    match command.child.try_wait() {
        Ok(Some(status)) => {
            return finish_command(command, CommandEnd::Finished { status }).await;
        }
        Ok(None) => {}
        Err(error) => return json!({ "error": error.to_string() }),
    }

    interrupt_child(&mut command.child);
    let status = match wait_for_exit(&mut command.child, INTERRUPT_GRACE).await {
        Ok(status) => status,
        Err(error) => return json!({ "error": error.to_string() }),
    };
    let (status, killed) = match status {
        Some(status) => (Some(status), false),
        None => {
            kill_child(&mut command.child);
            (command.child.wait().await.ok(), true)
        }
    };

    finish_command(command, CommandEnd::Killed { status, killed }).await
}

/// Run a command and optionally stream bounded stdout/stderr chunks to live output.
/// The returned JSON includes bounded stdout/stderr plus omitted byte counters.
pub async fn call(args: Args, stride: Stride) -> serde_json::Value {
    let commands = stride.running_commands();
    if args.argv.is_empty() {
        return json!({ "error": "argv must be non-empty" });
    }
    let wait_for = match wait_duration(args.wait_seconds) {
        Ok(wait_for) => wait_for,
        Err(error) => return json!({ "error": error }),
    };
    start_command(args.argv, wait_for, commands, stride.live_output()).await
}

pub fn spec() -> (&'static str, &'static str, Vec<Param>) {
    (
        NAME,
        "Start a command by argv. Output is capped. Commands still running after waitSeconds, default 40, return their pid instead of being interrupted.",
        vec![
            Param {
                name: "argv",
                desc: "Argument vector for a new command: [program, ...args]",
                param_type: ParamType::String,
                required: true,
            },
            Param {
                name: "waitSeconds",
                desc: "Seconds to wait before returning control to the model; defaults to 40",
                param_type: ParamType::Number,
                required: false,
            },
        ],
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
                wait_seconds: None,
            },
            Stride::default(),
        )
        .await;

        assert_eq!(result["status"], "finished");
        assert_eq!(result["exitCode"], 0);
        assert_eq!(result["stdout"], "hello");
        assert_eq!(result["stderr"], "problem");
        assert_eq!(result["stdoutBytesOmitted"], 0);
        assert_eq!(result["stderrBytesOmitted"], 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn long_command_can_be_waited_instead_of_interrupted() {
        let stride = Stride::default();
        let result = call(
            Args {
                argv: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "sleep 0.15; printf done".to_string(),
                ],
                wait_seconds: Some(0.02),
            },
            stride.clone(),
        )
        .await;

        assert_eq!(result["status"], "running");
        let pid = result["pid"].as_u64().unwrap() as u32;

        let result = wait_by_pid(pid, Some(0.3), stride).await;

        assert_eq!(result["status"], "finished");
        assert_eq!(result["exitCode"], 0);
        assert_eq!(result["stdout"], "done");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn running_result_includes_partial_output() {
        let stride = Stride::default();
        let result = call(
            Args {
                argv: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "printf start; printf err >&2; sleep 999".to_string(),
                ],
                wait_seconds: Some(0.02),
            },
            stride.clone(),
        )
        .await;

        assert_eq!(result["status"], "running");
        assert_eq!(result["stdout"], "start");
        assert_eq!(result["stderr"], "err");
        let pid = result["pid"].as_u64().unwrap() as u32;

        let result = kill_by_pid(pid, stride).await;
        assert_eq!(result["status"], "killed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn control_wait_running_result_includes_later_partial_output() {
        let stride = Stride::default();
        let result = call(
            Args {
                argv: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "printf one; sleep 0.05; printf two; sleep 999".to_string(),
                ],
                wait_seconds: Some(0.01),
            },
            stride.clone(),
        )
        .await;

        assert_eq!(result["status"], "running");
        assert_eq!(result["stdout"], "one");
        let pid = result["pid"].as_u64().unwrap() as u32;

        let result = wait_by_pid(pid, Some(0.1), stride.clone()).await;

        assert_eq!(result["status"], "running");
        assert_eq!(result["stdout"], "onetwo");

        let result = kill_by_pid(pid, stride).await;
        assert_eq!(result["status"], "killed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn running_command_can_be_killed_by_pid() {
        let stride = Stride::default();
        let result = call(
            Args {
                argv: vec!["sh".to_string(), "-c".to_string(), "sleep 999".to_string()],
                wait_seconds: Some(0.02),
            },
            stride.clone(),
        )
        .await;

        assert_eq!(result["status"], "running");
        let pid = result["pid"].as_u64().unwrap() as u32;

        let result = kill_by_pid(pid, stride).await;

        assert_eq!(result["status"], "killed");
        assert_eq!(result["ok"], false);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn kill_reports_finished_if_command_already_exited() {
        let stride = Stride::default();
        let result = call(
            Args {
                argv: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "sleep 0.05; printf after".to_string(),
                ],
                wait_seconds: Some(0.01),
            },
            stride.clone(),
        )
        .await;

        assert_eq!(result["status"], "running");
        let pid = result["pid"].as_u64().unwrap() as u32;

        tokio::time::sleep(Duration::from_millis(100)).await;

        let result = kill_by_pid(pid, stride).await;

        assert_eq!(result["status"], "finished");
        assert_eq!(result["exitCode"], 0);
        assert_eq!(result["stdout"], "after");
    }
}

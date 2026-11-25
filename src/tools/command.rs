use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::os::unix::process::ExitStatusExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::common::{Param, ParamType};
use super::{Tool, ToolContext};

mod job;

use job::{ExitCause, Job, Observation, STOP_WAIT};

pub const RUN_NAME: &str = "run_command";
pub const CONTROL_NAME: &str = "control_command";
const DEFAULT_WAIT: Duration = Duration::from_secs(40);
const CLEANUP_WAIT: Duration = Duration::from_secs(2);
/// IDs stay distinct across the turns retained in a REPL conversation.
static NEXT_COMMAND_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Deserialize)]
struct Start {
    argv: Vec<String>,
    wait_seconds: Option<f64>,
}

#[derive(Deserialize)]
struct Control {
    command_id: u64,
    action: Action,
    wait_seconds: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Action {
    Wait,
    Kill,
}

/// Commands owned by an assistant turn, including exits the model has yet to collect.
#[derive(Default)]
pub struct Commands(Mutex<HashMap<u64, Arc<Job>>>);

impl Commands {
    fn insert(&self, job: Arc<Job>) -> u64 {
        let id = NEXT_COMMAND_ID.fetch_add(1, Ordering::Relaxed);
        let mut pending = self.0.lock().unwrap();
        pending.insert(id, job);
        id
    }

    fn get(&self, id: u64) -> Option<Arc<Job>> {
        let pending = self.0.lock().unwrap();
        pending.get(&id).cloned()
    }

    pub fn pending_ids(&self) -> Vec<u64> {
        let pending = self.0.lock().unwrap();
        let mut ids = pending.keys().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

    async fn observe(&self, id: u64, job: &Job, wait: Duration) -> Value {
        if !wait.is_zero() {
            let _ = tokio::time::timeout(wait, job.wait()).await;
        }
        let observation = job.observe();
        if observation.completion.is_some() {
            let mut pending = self.0.lock().unwrap();
            pending.remove(&id);
        }
        tool_result(id, observation)
    }

    /// Stop all jobs before waiting for any one to be reaped. Tool calls borrow this registry.
    pub async fn shutdown(self) {
        let jobs = self.0.into_inner().unwrap();
        for job in jobs.values() {
            job.force_stop();
        }
        let _ = tokio::time::timeout(CLEANUP_WAIT, async {
            for job in jobs.values() {
                job.wait().await;
            }
        })
        .await;
    }
}

fn tool_result(id: u64, observation: Observation) -> Value {
    let mut result = json!({
        "command_id": id,
        "pid": observation.pid,
        "running_for": format!("{:.1}s", observation.elapsed.as_secs_f64()),
        "stdout": observation.stdout.text,
        "stdout_bytes_omitted": observation.stdout.omitted,
        "stderr": observation.stderr.text,
        "stderr_bytes_omitted": observation.stderr.omitted,
    });
    let Some(completion) = observation.completion else {
        result["ok"] = json!(false);
        result["status"] = json!("running");
        return result;
    };
    if let Some(error) = completion.output_error {
        result["output_error"] = json!(error);
    }
    let exit = match completion.exit {
        Err(error) => {
            result["status"] = json!("error");
            result["error"] = json!(error);
            return result;
        }
        Ok(exit) => exit,
    };
    result["exit_code"] = json!(exit.status.code());
    if let Some(signal) = exit.status.signal() {
        result["exit_signal"] = json!(signal);
    }
    let (signal, escalated) = match exit.cause {
        ExitCause::Natural => {
            result["status"] = json!("finished");
            result["ok"] = json!(exit.status.success());
            return result;
        }
        ExitCause::Interrupted => ("SIGINT", false),
        ExitCause::Escalated => ("SIGINT", true),
        ExitCause::Forced => ("SIGKILL", false),
    };
    result["status"] = json!("killed");
    result["ok"] = json!(false);
    result["kill"] = json!({ "signal": signal, "killed_after_grace": escalated });
    result
}

fn wait_duration(wait_seconds: Option<f64>) -> Result<Duration, &'static str> {
    let Some(wait_seconds) = wait_seconds else {
        return Ok(DEFAULT_WAIT);
    };
    if !wait_seconds.is_finite() || wait_seconds < 0.0 {
        return Err("wait_seconds must be a finite non-negative number");
    }
    Duration::try_from_secs_f64(wait_seconds).map_err(|_| "wait_seconds is too large to represent")
}

async fn start(args: Start, context: ToolContext<'_>) -> Value {
    if args.argv.is_empty() {
        return json!({ "error": "argv must be non-empty" });
    }
    let wait = match wait_duration(args.wait_seconds) {
        Err(error) => return json!({ "error": error }),
        Ok(wait) => wait,
    };
    let job = Job::spawn(&args.argv, context.live_output);
    let job = match job {
        Err(error) => return json!({ "error": error.to_string() }),
        Ok(job) => Arc::new(job),
    };
    let id = context.commands.insert(job.clone());
    context.commands.observe(id, &job, wait).await
}

async fn control(args: Control, context: ToolContext<'_>) -> Value {
    let id = args.command_id;
    let Some(job) = context.commands.get(id) else {
        return json!({ "error": format!("unknown command_id {id}") });
    };
    let wait = match args.action {
        Action::Wait => match wait_duration(args.wait_seconds) {
            Err(error) => return json!({ "error": error }),
            Ok(wait) => wait,
        },
        Action::Kill => {
            job.interrupt();
            STOP_WAIT
        }
    };
    if let Some(live) = context.live_output {
        job.stream_to(live);
    }
    context.commands.observe(id, &job, wait).await
}

const START_SPEC: (&str, &str, &[Param]) = (
    RUN_NAME,
    "Start a command by argv. A command that outlives wait_seconds returns a command_id. Use control_command to get more output or stop it.",
    &[
        Param {
            name: "argv",
            desc: "Argument vector for a new command: [program, ...args]",
            param_type: ParamType::StringArray,
            required: true,
        },
        Param {
            name: "wait_seconds",
            desc: "Seconds to wait before returning control to the model (default 40, use 0 to poll immediately)",
            param_type: ParamType::Number,
            required: false,
        },
    ],
);

const CONTROL_SPEC: (&str, &str, &[Param]) = (
    CONTROL_NAME,
    "Collect the next stdout/stderr chunk from a command. Use wait to let it keep running or kill to stop it.",
    &[
        Param {
            name: "command_id",
            desc: "Command handle returned by run_command",
            param_type: ParamType::Number,
            required: true,
        },
        Param {
            name: "action",
            desc: "Whether to wait longer or stop the command",
            param_type: ParamType::Choice(&["wait", "kill"]),
            required: true,
        },
        Param {
            name: "wait_seconds",
            desc: "With wait, seconds to wait before returning control to the model (default 40, use 0 to poll immediately); kill ignores it",
            param_type: ParamType::Number,
            required: false,
        },
    ],
);

pub fn tools() -> [(&'static str, Tool); 2] {
    [tool!(START_SPEC, start), tool!(CONTROL_SPEC, control)]
}

#[cfg(test)]
mod tests;

use super::common::{Param, ParamType, Stride};
use serde::Deserialize;

pub const NAME: &str = "control_command";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    /// Process id returned by run_command.
    pid: u32,
    /// Whether to keep waiting or stop the command.
    action: Action,
    /// Seconds to wait before returning control to the model.
    #[serde(default)]
    wait_seconds: Option<f64>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Action {
    Wait,
    Kill,
}

pub async fn call(args: Args, stride: Stride) -> serde_json::Value {
    match args.action {
        Action::Wait => super::run_command::wait_by_pid(args.pid, args.wait_seconds, stride).await,
        Action::Kill => super::run_command::kill_by_pid(args.pid, stride).await,
    }
}

pub fn spec() -> (&'static str, &'static str, Vec<Param>) {
    (
        NAME,
        "Control a command that run_command left running.",
        vec![
            Param {
                name: "pid",
                desc: "Process id returned by run_command",
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
                name: "waitSeconds",
                desc: "Seconds to wait before returning control to the model; defaults to 40",
                param_type: ParamType::Number,
                required: false,
            },
        ],
    )
}

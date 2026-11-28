use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use tokio::sync::mpsc::UnboundedSender;

use serde_json::Value;
use spec::Spec;

type Handler =
    for<'turn> fn(Value, ToolContext<'turn>) -> Pin<Box<dyn Future<Output = Value> + Send + 'turn>>;

/// A tool the probe runs: what the model is told of it, and where its calls go.
pub struct Tool {
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub spec: Spec,
    handler: Handler,
}

macro_rules! tool {
    ($spec:path, $call:path) => {{
        let handler: $crate::tools::Handler = |arguments, context| {
            Box::pin(async move {
                match serde_json::from_value(arguments) {
                    Err(error) => serde_json::json!({ "error": error.to_string() }),
                    Ok(arguments) => $call(arguments, context).await,
                }
            })
        };
        (
            $spec.name,
            $crate::tools::Tool {
                spec: $spec,
                handler,
            },
        )
    }};
}

mod apply_patch;
mod command;
mod find_related;
mod list_files;
mod read_file;
pub mod spec;
mod workspace;

pub use apply_patch::preview_patch;
pub use command::Jobs;

/// The probe's tools, keyed by name.
pub type Tools = HashMap<&'static str, Tool>;

/// A tool borrows its turn's jobs and may receive a live output sink for this call.
#[derive(Clone, Copy)]
pub struct ToolContext<'turn> {
    jobs: &'turn Jobs,
    live_output: Option<&'turn UnboundedSender<String>>,
}

impl<'turn> ToolContext<'turn> {
    pub fn new(jobs: &'turn Jobs, live_output: Option<&'turn UnboundedSender<String>>) -> Self {
        Self { jobs, live_output }
    }
}

/// What the probe tells apart about a tool to approve its calls and show their output.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ToolKind {
    RunCommand,
    ControlCommand,
    ApplyPatch,
    Other,
}

impl ToolKind {
    /// Whether the tool's result carries a command's output streams.
    pub fn has_command_output(self) -> bool {
        matches!(self, Self::RunCommand | Self::ControlCommand)
    }

    /// Whether a call with `args` starts a command: one without an argv fails before anything
    /// runs.
    pub fn starts_command(self, args: &Value) -> bool {
        matches!(self, Self::RunCommand)
            && args
                .get("argv")
                .and_then(Value::as_array)
                .is_some_and(|argv| !argv.is_empty())
    }
}

/// Tell a tool's kind by its name; a tool that needs no handling of its own is `Other`.
pub fn kind_of(name: &str) -> ToolKind {
    match name {
        command::RUN_NAME => ToolKind::RunCommand,
        command::CONTROL_NAME => ToolKind::ControlCommand,
        apply_patch::NAME => ToolKind::ApplyPatch,
        _ => ToolKind::Other,
    }
}

/// Every tool the probe runs.
pub fn all_tools() -> Tools {
    Tools::from([
        tool!(list_files::SPEC, list_files::call),
        tool!(read_file::SPEC, read_file::call),
        tool!(find_related::SPEC, find_related::call),
        tool!(apply_patch::SPEC, apply_patch::call),
        tool!(command::RUN_SPEC, command::run),
        tool!(command::CONTROL_SPEC, command::control),
    ])
}

/// What a tool's result says, without the fields a command's result changes on every call: its
/// handle, process, and running time. `None` while a command still runs, since waiting on one
/// is progress rather than repetition.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn outcome(result: &str) -> Option<Value> {
    let mut value = serde_json::from_str(result).unwrap_or_else(|_| Value::String(result.into()));
    if let Some(fields) = value.as_object_mut() {
        if fields.get("status").and_then(Value::as_str) == Some("running") {
            return None;
        }
        for field in ["command_id", "pid", "running_for"] {
            fields.remove(field);
        }
    }
    Some(value)
}

/// Invoke a tool with services scoped to this tool call.
pub async fn invoke(tools: &Tools, context: ToolContext<'_>, name: &str, args: Value) -> Value {
    let Some(tool) = tools.get(name) else {
        return serde_json::json!({ "error": "No such function" });
    };
    (tool.handler)(args, context).await
}

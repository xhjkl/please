use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use tokio::sync::mpsc::UnboundedSender;

use serde_json::Value;

type Handler =
    for<'turn> fn(Value, ToolContext<'turn>) -> Pin<Box<dyn Future<Output = Value> + Send + 'turn>>;
type Tool = (&'static str, Handler, &'static [Param]);

macro_rules! tool {
    ($spec:path, $call:path) => {{
        let (name, description, params) = $spec;
        let handler: $crate::tools::Handler = |arguments, context| {
            Box::pin(async move {
                match serde_json::from_value(arguments) {
                    Err(error) => serde_json::json!({ "error": error.to_string() }),
                    Ok(arguments) => $call(arguments, context).await,
                }
            })
        };
        (name, (description, handler, params))
    }};
}

mod common;
pub use self::common::Param;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use self::common::ParamType;

mod apply_patch;
mod command;
mod find_related;
mod list_files;
mod read_file;

pub use apply_patch::summarize_patch_for_preview;
pub use command::Commands;

/// Exposed tools are represented as a map keyed by function name.
pub type ExposedTools = HashMap<&'static str, Tool>;

/// A tool borrows its turn's command registry and may receive a live output sink for this call.
#[derive(Clone, Copy)]
pub struct ToolContext<'turn> {
    commands: &'turn Commands,
    live_output: Option<&'turn UnboundedSender<String>>,
}

impl<'turn> ToolContext<'turn> {
    pub fn new(
        commands: &'turn Commands,
        live_output: Option<&'turn UnboundedSender<String>>,
    ) -> Self {
        Self {
            commands,
            live_output,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ToolKind {
    RunCommand,
    ControlCommand,
    ApplyPatch,
    Other,
}

impl ToolKind {
    pub fn has_command_output(self) -> bool {
        matches!(self, Self::RunCommand | Self::ControlCommand)
    }

    pub fn starts_command(self, args: &serde_json::Value) -> bool {
        matches!(self, Self::RunCommand)
            && args
                .get("argv")
                .and_then(|value| value.as_array())
                .is_some_and(|argv| !argv.is_empty())
    }
}

pub fn kind_of(name: &str) -> ToolKind {
    if name == command::RUN_NAME {
        return ToolKind::RunCommand;
    }
    if name == command::CONTROL_NAME {
        return ToolKind::ControlCommand;
    }
    if name == apply_patch::NAME {
        return ToolKind::ApplyPatch;
    }
    ToolKind::Other
}

pub const CONTROL_COMMAND_NAME: &str = command::CONTROL_NAME;

pub fn all_tools() -> ExposedTools {
    [
        tool!(list_files::SPEC, list_files::call),
        tool!(read_file::SPEC, read_file::call),
        tool!(find_related::SPEC, find_related::call),
        tool!(apply_patch::SPEC, apply_patch::call),
    ]
    .into_iter()
    .chain(command::tools())
    .collect()
}

/// Invoke a tool with services scoped to this tool call.
pub async fn invoke(
    tools: &ExposedTools,
    context: ToolContext<'_>,
    name: &str,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let Some((_, work, _)) = tools.get(name) else {
        return Err("No such function".to_string());
    };
    Ok(work(args, context).await)
}

use std::collections::HashMap;

pub mod common;
use self::common::{AsyncFn, Param, with_args};

mod apply_patch;
mod control_command;
mod list_files;
mod read_file;
mod run_command;

pub use self::common::Stride;
pub use apply_patch::summarize_patch_for_preview;

/// Exposed tools are represented as a map keyed by function name.
pub type ExposedTools = HashMap<&'static str, (&'static str, AsyncFn, Vec<Param>)>;

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

    pub fn is_control_command(self) -> bool {
        matches!(self, Self::ControlCommand)
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
    if name == run_command::NAME {
        return ToolKind::RunCommand;
    }
    if name == control_command::NAME {
        return ToolKind::ControlCommand;
    }
    if name == apply_patch::NAME {
        return ToolKind::ApplyPatch;
    }
    ToolKind::Other
}

pub const CONTROL_COMMAND_NAME: &str = control_command::NAME;

pub fn all_tools() -> ExposedTools {
    macro_rules! collect_tools {
      ($($module:ident),+ $(,)?) => {{
        let mut map: ExposedTools = HashMap::new();
        $(
            let (name, desc, params) = $module::spec();
            let call: AsyncFn = with_args::<$module::Args, _, _>($module::call);
            map.insert(name, (desc, call, params));
        )+
        map
      }};
    }

    collect_tools![
        list_files,
        read_file,
        run_command,
        control_command,
        apply_patch
    ]
}

/// Invoke a tool with services scoped to this tool call.
pub async fn invoke(
    tools: &ExposedTools,
    stride: Stride,
    name: &str,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let Some((_, work, _)) = tools.get(name) else {
        return Err("No such function".to_string());
    };
    Ok(work(args, stride).await)
}

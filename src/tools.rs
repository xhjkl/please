use std::collections::HashMap;

pub mod common;
use self::common::{AsyncFn, Param, with_args};

mod apply_patch;
mod list_files;
mod read_file;
mod run_command;

pub use self::common::Stride;
pub use apply_patch::summarize_patch_for_preview;

/// Exposed tools are represented as a map keyed by function name.
pub type ExposedTools = HashMap<&'static str, (&'static str, AsyncFn, Vec<Param>)>;

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

    collect_tools![list_files, read_file, run_command, apply_patch]
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

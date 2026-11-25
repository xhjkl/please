mod applying;
mod filesystem;
mod model;
mod parsing;
mod preview;
mod text;

use super::ToolContext;
use super::common::{Param, ParamType};
use serde::Deserialize;
use serde_json::json;

pub use preview::summarize_patch_for_preview;

pub const NAME: &str = "apply_patch";

#[derive(Deserialize)]
pub struct Args {
    /// Target path for overwrite mode (ignored in patch mode)
    path: Option<String>,
    /// Raw content to overwrite with, or an OpenAI-style patch to apply
    patch: Option<String>,
}

pub async fn call(args: Args, _context: ToolContext<'_>) -> serde_json::Value {
    let Some(patch) = args.patch else {
        return json!({ "error": "apply_patch requires parameter `patch`" });
    };
    if !parsing::contains_patch_syntax(&patch) {
        let Some(path) = args.path.as_deref() else {
            return json!({ "error": "overwrite mode requires `path`" });
        };
        let written = filesystem::write_verbatim_within_cwd(path, &patch);
        return match written {
            Err(error) => json!({ "error": error.to_string() }),
            Ok(()) => json!({ "ok": true, "mode": "overwrite", "path": path }),
        };
    }
    match parsing::parse_patch_ops(&patch) {
        Err(error) => json!({ "error": error }),
        Ok(ops) => filesystem::execute_patch_ops(ops),
    }
}

pub const SPEC: (&str, &str, &[Param]) = (
    NAME,
    "Apply edits via OpenAI-style patch markers or overwrite without markers. Patch format: wrap ops between '*** Begin Patch' and '*** End Patch'; each op starts with '*** Update File:', '*** Add File:' or '*** Delete File:'. Update bodies use + / - / space prefixes and optional @@ separators; add bodies are raw file content. Append a 'No newline at end of file' comment line to suppress trailing newline. Without markers, requires `path` and overwrites verbatim.",
    &[
        Param {
            name: "path",
            desc: "Target file path for simple overwrite (ignored for patch mode)",
            param_type: ParamType::String,
            required: false,
        },
        Param {
            name: "patch",
            desc: "Either raw content (overwrite) or an OpenAI patch",
            param_type: ParamType::String,
            required: true,
        },
    ],
);

#[cfg(test)]
mod tests;

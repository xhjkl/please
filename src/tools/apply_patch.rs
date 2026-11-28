mod applying;
mod filesystem;
mod model;
mod parsing;
mod preview;

use super::ToolContext;
use super::common::{Param, ParamKind, Spec};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;

pub use preview::preview_patch;

pub const NAME: &str = "apply_patch";

#[derive(Deserialize)]
pub struct Args {
    /// Target path for overwrite mode (ignored in patch mode)
    path: Option<String>,
    /// Raw content to overwrite with, or an OpenAI-style patch to apply
    patch: Option<String>,
}

pub async fn call(args: Args, _context: ToolContext<'_>) -> Value {
    let workspace = std::env::current_dir();
    match workspace {
        Err(error) => json!({ "error": error.to_string() }),
        Ok(workspace) => apply(&workspace, args),
    }
}

/// Carry out a call in `workspace`: overwrite one file verbatim, or run a patch's operations,
/// each reporting for itself.
fn apply(workspace: &Path, args: Args) -> Value {
    let Some(patch) = args.patch else {
        return json!({ "error": "apply_patch requires parameter `patch`" });
    };
    if !parsing::contains_patch_syntax(&patch) {
        let Some(path) = args.path.as_deref() else {
            return json!({ "error": "overwrite mode requires `path`" });
        };
        let written = filesystem::write_verbatim(workspace, path, &patch);
        return match written {
            Err(error) => json!({ "error": error.to_string() }),
            Ok(()) => json!({ "ok": true, "mode": "overwrite", "path": path }),
        };
    }
    let ops = match parsing::parse_patch_ops(&patch) {
        Err(error) => return json!({ "error": error }),
        Ok(ops) => ops,
    };
    let results: Vec<_> = ops
        .into_iter()
        .map(|op| filesystem::execute(workspace, op))
        .collect();
    json!({ "ok": true, "mode": "patch", "results": results })
}

pub const SPEC: Spec = Spec {
    name: NAME,
    description: "Apply edits via OpenAI-style patch markers or overwrite without markers. Patch format: wrap ops between '*** Begin Patch' and '*** End Patch'; each op starts with '*** Update File:', '*** Add File:' or '*** Delete File:'. Update bodies use + / - / space prefixes and optional @@ separators; add bodies are raw file content. Append a 'No newline at end of file' comment line to suppress trailing newline. Without markers, requires `path` and overwrites verbatim.",
    params: &[
        Param {
            name: "path",
            description: "Target file path for simple overwrite (ignored for patch mode)",
            kind: ParamKind::String,
            required: false,
        },
        Param {
            name: "patch",
            description: "Either raw content (overwrite) or an OpenAI patch",
            kind: ParamKind::String,
            required: true,
        },
    ],
};

#[cfg(test)]
mod tests;

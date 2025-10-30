mod applying;
mod filesystem;
mod model;
mod parsing;
mod preview;
mod text;

use super::common::{Param, ParamType};
use serde::Deserialize;
use serde_json::json;

pub use parsing::parse_patch_ops;
pub use preview::summarize_patch_for_preview;

#[derive(Deserialize)]
pub struct Args {
    /// Target path for overwrite mode (ignored in patch mode)
    #[serde(default)]
    path: Option<String>,
    /// Raw content to overwrite with, or an OpenAI-style patch to apply
    patch: Option<String>,
}

pub async fn call(args: Args) -> serde_json::Value {
    let content = match args.patch {
        Some(s) => s,
        None => return json!({ "error": "apply_patch requires parameter `patch`" }),
    };

    if !parsing::contains_patch_markers(&content) {
        // Overwrite mode: write verbatim to `path`
        let Some(path) = args.path.as_deref() else {
            return json!({ "error": "overwrite mode requires `path`" });
        };

        return match filesystem::write_verbatim_within_cwd(path, &content) {
            Ok(()) => json!({ "ok": true, "mode": "overwrite", "path": path }),
            Err(e) => json!({ "error": e.to_string() }),
        };
    }

    // Patch mode: parse -> execute; tolerate per-op errors, keep going.
    match parse_patch_ops(&content) {
        Ok(ops) => filesystem::execute_patch_ops(ops),
        Err(e) => json!({ "error": e }),
    }
}

pub fn spec() -> (&'static str, &'static str, Vec<Param>) {
    (
        "apply_patch",
        "Apply edits via OpenAI-style patch markers or overwrite without markers. Patch format: wrap ops between '*** Begin Patch' and '*** End Patch'; each op starts with '*** Update File:', '*** Add File:' or '*** Delete File:'. Update bodies use + / - / space prefixes and optional @@ separators; add bodies are raw file content. Append a 'No newline at end of file' comment line to suppress trailing newline. Without markers, requires `path` and overwrites verbatim.",
        vec![
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
    )
}

#[cfg(test)]
mod tests;

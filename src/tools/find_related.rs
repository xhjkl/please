//! Locate code by what it does, its identifiers, or a snippet, without reading whole files.
use super::ToolContext;
use super::common::{Param, ParamType, resolve_path_within_cwd};
use serde::Deserialize;
use serde_json::json;

mod index;
#[cfg(test)]
mod tests;

#[derive(Deserialize)]
pub struct Args {
    query: String,
    path: Option<String>,
}

pub async fn call(args: Args, _context: ToolContext<'_>) -> serde_json::Value {
    let scope = resolve_path_within_cwd(args.path.as_deref().unwrap_or("."));
    let scope = match scope {
        Err(error) => return json!({ "error": error.to_string() }),
        Ok(scope) => scope,
    };
    let is_dir = scope.is_dir();
    if !is_dir {
        return json!({ "error": format!("not a directory: {}", scope.display()) });
    }
    let found = tokio::task::spawn_blocking(move || index::search(&scope, &args.query)).await;
    match found {
        Err(error) => json!({ "error": error.to_string() }),
        Ok(Err(error)) => json!({ "error": error.to_string() }),
        Ok(Ok(text)) => json!(text),
    }
}

pub const SPEC: (&str, &str, &[Param]) = (
    "find_related",
    "Find where code related to a query lives: describe the behavior in words, or give an identifier, error message, or code snippet. Read-only and fast. Returns up to 5 files, best first, each as `path:first-last` with its most relevant numbered lines; read a file for more.",
    &[
        Param {
            name: "query",
            desc: "What to look for, in words or code",
            param_type: ParamType::String,
            required: true,
        },
        Param {
            name: "path",
            desc: "Directory to search; defaults to the current directory",
            param_type: ParamType::String,
            required: false,
        },
    ],
);

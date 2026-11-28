//! Locate code by what it does, its identifiers, or a snippet, without reading whole files.
use super::ToolContext;
use super::spec::{Param, ParamKind, Spec};
use super::workspace::resolve_path_within_cwd;
use serde::Deserialize;
use serde_json::json;

mod index;

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

pub const SPEC: Spec = Spec {
    name: "find_related",
    description: "Find where code related to a query lives: describe the behavior in words, or give an identifier, error message, or code snippet. Read-only and fast. Returns up to 5 files, best first, each as `path:first-last` with its most relevant numbered lines; read a file for more.",
    params: &[
        Param {
            name: "query",
            description: "What to look for, in words or code",
            kind: ParamKind::String,
            required: true,
        },
        Param {
            name: "path",
            description: "Directory to search; defaults to the current directory",
            kind: ParamKind::String,
            required: false,
        },
    ],
};

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn find_related_refuses_paths_outside_the_workspace() {
        let jobs = crate::tools::Jobs::default();
        let context = crate::tools::ToolContext::new(&jobs, None);
        let args = serde_json::json!({ "query": "anything", "path": "/" });
        let args = serde_json::from_value(args).unwrap();
        let result = super::call(args, context).await;
        assert!(result.get("error").is_some(), "{result}");
    }
}

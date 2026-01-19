use super::common::{Param, ParamType, resolve_path_within_cwd};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
pub struct Args {
    #[serde(default = "default_dot")]
    path: String,
    #[serde(default = "default_depth")]
    max_depth: usize,
}

fn default_dot() -> String {
    ".".to_string()
}

fn default_depth() -> usize {
    0
}

pub async fn call(
    args: Args,
    _sink: Option<tokio::sync::mpsc::UnboundedSender<String>>,
) -> serde_json::Value {
    let root = match resolve_path_within_cwd(&args.path) {
        Ok(p) => p,
        Err(e) => return serde_json::json!({ "error": e.to_string() }),
    };
    if !root.exists() {
        return serde_json::json!({ "error": format!("path does not exist: {}", root.display()) });
    }

    let mut out: Vec<String> = Vec::new();
    let max_depth = args.max_depth;

    fn is_excluded_dir(name: &str) -> bool {
        matches!(
            name,
            "target" | "node_modules" | "dist" | "build" | "lib" | "out"
        )
    }

    fn walk(
        cur: &Path,
        base: &Path,
        depth: usize,
        max_depth: usize,
        out: &mut Vec<String>,
    ) -> std::io::Result<()> {
        if depth > max_depth {
            return Ok(());
        }
        for entry in fs::read_dir(cur)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if is_excluded_dir(&name) {
                    continue;
                }
            }
            let rel = path.strip_prefix(base).unwrap_or(&path).to_path_buf();
            let mut s = rel.display().to_string();
            if path.is_dir() && !s.ends_with('/') {
                s.push('/');
            }
            out.push(s);
            if path.is_dir() {
                walk(&path, base, depth + 1, max_depth, out)?;
            }
        }
        Ok(())
    }
    let base = if root.is_dir() {
        root.clone()
    } else {
        root.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    if let Err(e) = walk(&root, &base, 0, max_depth, &mut out) {
        return serde_json::json!({ "error": e.to_string() });
    }
    serde_json::json!(out)
}

pub fn spec() -> (&'static str, &'static str, Vec<Param>) {
    (
        "list_files",
        "List files under a path recursively with optional depth",
        vec![
            Param {
                name: "path",
                desc: "Root path; defaults to current directory",
                param_type: ParamType::String,
                required: false,
            },
            Param {
                name: "max_depth",
                desc: "Max recursion depth; default 0, just the given directory",
                param_type: ParamType::Number,
                required: false,
            },
        ],
    )
}

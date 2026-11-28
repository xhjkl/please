use super::ToolContext;
use super::spec::{Param, ParamKind, Spec};
use super::workspace::resolve_path_within_cwd;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;
use std::{fs, io};

#[derive(Deserialize)]
pub struct Args {
    #[serde(default = "default_dot")]
    path: String,
    #[serde(default)]
    max_depth: usize,
}

fn default_dot() -> String {
    ".".to_string()
}

pub async fn call(args: Args, _context: ToolContext<'_>) -> serde_json::Value {
    let root = resolve_path_within_cwd(&args.path);
    let root = match root {
        Err(error) => return json!({ "error": error.to_string() }),
        Ok(root) => root,
    };
    let exists = root.exists();
    if !exists {
        return json!({ "error": format!("path does not exist: {}", root.display()) });
    }
    let mut listed = Vec::new();
    let walked = walk(&root, &root, 0, args.max_depth, &mut listed);
    if let Err(error) = walked {
        return json!({ "error": error.to_string() });
    }
    json!(listed)
}

fn is_excluded_dir(name: &str) -> bool {
    matches!(
        name,
        "target" | "node_modules" | "dist" | "build" | "lib" | "out"
    )
}

fn walk(
    dir: &Path,
    base: &Path,
    depth: usize,
    max_depth: usize,
    listed: &mut Vec<String>,
) -> io::Result<()> {
    if depth > max_depth {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let is_dir = path.is_dir();
        if is_dir && is_excluded_dir(&entry.file_name().to_string_lossy()) {
            continue;
        }
        let shown = path.strip_prefix(base).unwrap_or(&path).display();
        if is_dir {
            listed.push(format!("{shown}/"));
            walk(&path, base, depth + 1, max_depth, listed)?;
        } else {
            listed.push(shown.to_string());
        }
    }
    Ok(())
}

pub const SPEC: Spec = Spec {
    name: "list_files",
    description: "List files under a path recursively with optional depth",
    params: &[
        Param {
            name: "path",
            description: "Root path; defaults to current directory",
            kind: ParamKind::String,
            required: false,
        },
        Param {
            name: "max_depth",
            description: "Max recursion depth; default 0, just the given directory",
            kind: ParamKind::Number,
            required: false,
        },
    ],
};

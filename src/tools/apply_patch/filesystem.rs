use serde_json::{Value, json};
use std::fs;
use std::io::{self, ErrorKind};

use super::applying::apply_all_hunks;
use super::model::PatchOp;
use super::text::set_trailing_newline;
use crate::tools::common::resolve_path_within_cwd;

pub fn write_verbatim_within_cwd(path: &str, content: &str) -> io::Result<()> {
    let rel = resolve_path_within_cwd(path)?;
    if let Some(parent) = rel.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(rel, content)
}

fn read_within_cwd(path: &str) -> io::Result<String> {
    let rel = resolve_path_within_cwd(path)?;
    fs::read_to_string(rel)
}

fn remove_file_if_exists(path: &str) -> io::Result<()> {
    let rel = resolve_path_within_cwd(path)?;
    let removed = fs::remove_file(rel);
    match removed {
        Err(error) if error.kind() != ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

pub fn execute_patch_ops(ops: Vec<PatchOp>) -> Value {
    let results: Vec<Value> = ops.into_iter().map(execute).collect();
    json!({ "ok": true, "mode": "patch", "results": results })
}

/// Apply one operation, reporting its failure in its result rather than stopping the patch.
fn execute(op: PatchOp) -> Value {
    match op {
        PatchOp::Add {
            path,
            content,
            no_newline,
        } => {
            let content = set_trailing_newline(&content, !no_newline);
            let written = write_verbatim_within_cwd(&path, &content);
            result(&path, "add", written.map_err(|error| error.to_string()))
        }
        PatchOp::Delete { path } => {
            let removed = remove_file_if_exists(&path);
            result(&path, "delete", removed.map_err(|error| error.to_string()))
        }
        PatchOp::Update {
            path,
            hunks,
            no_newline,
        } => {
            let before = read_within_cwd(&path);
            let before = match before {
                Err(error) if error.kind() != ErrorKind::NotFound => {
                    return result(&path, "update", Err(format!("read: {error}")));
                }
                Err(_) => String::new(),
                Ok(before) => before,
            };
            let after = match apply_all_hunks(&before, &hunks) {
                Err(errors) => {
                    let errors: Vec<Value> = errors
                        .iter()
                        .map(|(hunk, error)| json!({ "hunk": hunk, "error": error }))
                        .collect();
                    return json!({ "path": path, "op": "update", "ok": false, "errors": errors });
                }
                Ok(after) => after,
            };
            let after = set_trailing_newline(&after, !no_newline);
            let written = write_verbatim_within_cwd(&path, &after);
            result(
                &path,
                "update",
                written.map_err(|error| format!("write: {error}")),
            )
        }
    }
}

fn result(path: &str, op: &str, done: Result<(), String>) -> Value {
    match done {
        Err(error) => json!({ "path": path, "op": op, "ok": false, "error": error }),
        Ok(()) => json!({ "path": path, "op": op, "ok": true }),
    }
}

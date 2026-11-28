use serde_json::{Value, json};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use super::applying::apply_all_hunks;
use super::model::PatchOp;
use crate::tools::common::resolve_path_within;

/// Find where `path` lies in `workspace`, refusing one that leaves it.
fn locate(workspace: &Path, path: &str) -> io::Result<PathBuf> {
    Ok(workspace.join(resolve_path_within(workspace, path)?))
}

/// Write `content` to `path` in `workspace`, creating the directories it lies in.
pub fn write_verbatim(workspace: &Path, path: &str, content: &str) -> io::Result<()> {
    let path = locate(workspace, path)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)
}

fn remove_file_if_exists(workspace: &Path, path: &str) -> io::Result<()> {
    let path = locate(workspace, path)?;
    let removed = fs::remove_file(path);
    match removed {
        Err(error) if error.kind() != ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// Apply one operation in `workspace`, reporting its failure in its result rather than
/// stopping the patch.
pub fn execute(workspace: &Path, op: PatchOp) -> Value {
    match op {
        PatchOp::Add {
            path,
            content,
            no_newline,
        } => {
            let content = set_trailing_newline(&content, !no_newline);
            let written = write_verbatim(workspace, &path, &content);
            result(&path, "add", written.map_err(|error| error.to_string()))
        }
        PatchOp::Delete { path } => {
            let removed = remove_file_if_exists(workspace, &path);
            result(&path, "delete", removed.map_err(|error| error.to_string()))
        }
        PatchOp::Update {
            path,
            hunks,
            no_newline,
        } => {
            let before = locate(workspace, &path).and_then(fs::read_to_string);
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
            let written = write_verbatim(workspace, &path, &after);
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

/// End `s` with one newline, or with none.
fn set_trailing_newline(s: &str, want_newline: bool) -> String {
    let mut t = s.trim_end_matches('\n').to_string();
    if want_newline {
        t.push('\n');
    }
    t
}

use serde_json::json;
use std::fs;
use std::io::ErrorKind;

use super::applying::apply_all_hunks;
use super::model::PatchOp;
use super::text::set_trailing_newline;
use crate::tools::common::resolve_path_within_cwd;

fn write_text_creating_dirs(
    path: &str,
    content: &str,
    want_trailing_newline: bool,
) -> std::io::Result<()> {
    let rel = resolve_path_within_cwd(path)?; // sanitized relative path
    if let Some(parent) = rel.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let content = set_trailing_newline(content, want_trailing_newline);
    fs::write(rel, content)
}

pub fn write_verbatim_within_cwd(path: &str, content: &str) -> std::io::Result<()> {
    let rel = resolve_path_within_cwd(path)?; // sanitized relative path
    if let Some(parent) = rel.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(rel, content)
}

fn remove_file_if_exists(path: &str) -> std::io::Result<()> {
    let rel = resolve_path_within_cwd(path)?; // sanitized relative path
    match fs::remove_file(rel) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

pub fn execute_patch_ops(ops: Vec<PatchOp>) -> serde_json::Value {
    let mut results = Vec::new();
    for op in ops {
        match op {
            PatchOp::Add {
                path,
                content,
                no_newline,
            } => {
                let res = write_text_creating_dirs(&path, &content, !no_newline);
                match res {
                    Ok(_) => results.push(json!({ "path": path, "op": "add", "ok": true })),
                    Err(e) => results.push(
                        json!({ "path": path, "op": "add", "ok": false, "error": e.to_string() }),
                    ),
                }
            }
            PatchOp::Delete { path } => {
                let res = remove_file_if_exists(&path);
                match res {
                    Ok(_) => results.push(json!({ "path": path, "op": "delete", "ok": true })),
                    Err(e) => results.push(
                        json!({ "path": path, "op": "delete", "ok": false, "error": e.to_string() }),
                    ),
                }
            }
            PatchOp::Update {
                path,
                hunks,
                no_newline,
            } => {
                let text0 = match resolve_path_within_cwd(&path).and_then(fs::read_to_string) {
                    Ok(s) => s,
                    Err(e) if e.kind() == ErrorKind::NotFound => String::new(),
                    Err(e) => {
                        results.push(json!({ "path": path, "op": "update", "ok": false, "error": format!("read: {}", e) }));
                        continue;
                    }
                };

                match apply_all_hunks(&text0, &hunks) {
                    Ok(text) => {
                        match write_text_creating_dirs(&path, &text, !no_newline) {
                            Ok(_) => results.push(json!({ "path": path, "op": "update", "ok": true })),
                            Err(e) => results.push(json!({ "path": path, "op": "update", "ok": false, "error": format!("write: {}", e) })),
                        }
                    }
                    Err(errs) => {
                        results.push(json!({
                            "path": path,
                            "op": "update",
                            "ok": false,
                            "errors": errs.iter().map(|(i, e)| json!({ "hunk": i, "error": e })).collect::<Vec<_>>()
                        }));
                    }
                }
            }
        }
    }
    json!({ "ok": true, "mode": "patch", "results": results })
}

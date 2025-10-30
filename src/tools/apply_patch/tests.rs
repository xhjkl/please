#![cfg(test)]

use serde_json::json;
use std::collections::BTreeMap;

use super::applying::{apply_all_hunks, apply_hunk};
use super::model::{Hunk, PatchOp};
use super::parsing::parse_patch_ops;
use super::text::set_trailing_newline;

fn execute_patch_ops_in_memory(
    files: &mut BTreeMap<String, String>,
    ops: Vec<PatchOp>,
) -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    for op in ops {
        match op {
            PatchOp::Add {
                path,
                content,
                no_newline,
            } => {
                let text = set_trailing_newline(&content, !no_newline);
                files.insert(path.clone(), text);
                results.push(json!({ "path": path, "op": "add", "ok": true }));
            }
            PatchOp::Delete { path } => {
                files.remove(&path);
                results.push(json!({ "path": path, "op": "delete", "ok": true }));
            }
            PatchOp::Update {
                path,
                hunks,
                no_newline,
            } => {
                let before = files.get(&path).cloned().unwrap_or_default();
                match apply_all_hunks(&before, &hunks) {
                    Ok(mut text) => {
                        text = set_trailing_newline(&text, !no_newline);
                        files.insert(path.clone(), text);
                        results.push(json!({ "path": path, "op": "update", "ok": true }));
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

    results
}

#[test]
fn pure_parse_missing_markers() {
    let err = parse_patch_ops("*** End Patch").unwrap_err();
    assert!(err.to_lowercase().contains("missing"));
}

#[test]
fn pure_parse_and_apply_update() {
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- a\n+ b\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    match &ops[0] {
        PatchOp::Update {
            path,
            hunks,
            no_newline,
        } => {
            assert_eq!(path, "text.text");
            assert!(!no_newline);
            let out = apply_all_hunks("a\n", &hunks).unwrap();
            assert_eq!(out, "b\n");
        }
        _ => panic!("expected update"),
    }
}

#[test]
fn update_simple_replace() {
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- hello\n+ hello there\n@@\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("text.text".to_string(), "hello\nworld\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert_eq!(files.get("text.text").unwrap(), "hello there\nworld\n");
}

#[test]
fn update_trailing_whitespace_ignored() {
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- hello\n+ hi\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("text.text".to_string(), "hello  \nworld\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert_eq!(files.get("text.text").unwrap(), "hi\nworld\n");
}

#[test]
fn update_with_context_space_missing() {
    // Context lines may miss leading space, treat as context
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\nA\n- B\n+ BB\nC\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("text.text".to_string(), "A\nB\nC\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert_eq!(files.get("text.text").unwrap(), "A\nBB\nC\n");
}

#[test]
fn update_pure_insert_at_eof() {
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n+ line2\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("text.text".to_string(), "line1\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert_eq!(files.get("text.text").unwrap(), "line1\nline2\n");
}

#[test]
fn update_handles_backslash_commentary() {
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- a\n\\ No newline at end of file\n+ b\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("text.text".to_string(), "a\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    // We set no_newline=true because the commentary follows an added line
    assert_eq!(files.get("text.text").unwrap(), "b");
}

#[test]
fn add_file_basic_and_fenced() {
    let patch = "*** Begin Patch\n*** Add File: some.text\nhello\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::<String, String>::new();
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(results.iter().any(|r| r["op"] == "add" && r["ok"] == true));
    // default: ensure single trailing newline
    assert_eq!(files.get("some.text").unwrap(), "hello\n");

    let patch2 = "*** Begin Patch\n*** Add File: another.text\n```\nhello\n```\n*** End Patch\n";
    let ops2 = parse_patch_ops(patch2).unwrap();
    let mut files2 = BTreeMap::<String, String>::new();
    let results2 = execute_patch_ops_in_memory(&mut files2, ops2);
    assert!(results2.iter().any(|r| r["op"] == "add" && r["ok"] == true));
    assert_eq!(files2.get("another.text").unwrap(), "hello\n");
}

#[test]
fn add_file_no_newline_escape_hatch() {
    let patch = "*** Begin Patch\n*** Add File: nonewline.text\nhello\n\\ No newline at end of file\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::<String, String>::new();
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(results.iter().any(|r| r["op"] == "add" && r["ok"] == true));
    assert_eq!(files.get("nonewline.text").unwrap(), "hello");
}

#[test]
fn delete_file_tolerates_missing() {
    let patch = "*** Begin Patch\n*** Delete File: missing.text\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::<String, String>::new();
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "delete" && r["ok"] == true)
    );
}

#[test]
fn multiple_ops_in_one_patch() {
    let patch = "*** Begin Patch\n*** Update File: some.text\n@@\n- A\n+ AA\n*** Add File: another.text\nB\n*** End Patch\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("some.text".to_string(), "A\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert!(results.iter().any(|r| r["op"] == "add" && r["ok"] == true));
    assert_eq!(files.get("some.text").unwrap(), "AA\n");
    // default newline normalization applies to Add without commentary
    assert_eq!(files.get("another.text").unwrap(), "B\n");
}

#[test]
fn crlf_and_cr_inputs_normalized() {
    let patch = "*** Begin Patch\r\n*** Update File: text.text\r\n@@\r\n- hello\r\n+ hi\r\n*** End Patch\r\n";
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("text.text".to_string(), "hello\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert_eq!(files.get("text.text").unwrap(), "hi\n");
}

#[test]
fn tolerant_casing_and_spacing_in_markers() {
    let patch = r#"
***    begin   PATCH
*** update file: text.text
@@
- hello
+ hi
***   END   patch
"#;
    let ops = parse_patch_ops(patch).unwrap();
    let mut files = BTreeMap::from([("text.text".to_string(), "hello\n".to_string())]);
    let results = execute_patch_ops_in_memory(&mut files, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert_eq!(files.get("text.text").unwrap(), "hi\n");
}

#[test]
fn parse_and_apply_add_update_delete_in_memory() {
    let patch = r#"
*** Begin Patch
*** Add File: some.text
```
hello
world
\ No newline at end of file
```

*** Update File: some.text
@@
-hello
+hello, friend

*** Delete File: nope.txt

*** End Patch
"#;

    let ops = parse_patch_ops(patch).expect("parse");
    let mut mem = BTreeMap::<String, String>::new();
    let results = execute_patch_ops_in_memory(&mut mem, ops);

    // Add ok
    assert!(results.iter().any(|r| r["op"] == "add" && r["ok"] == true));
    // Update ok
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    // Delete ok (tolerated)
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "delete" && r["ok"] == true)
    );

    // Validate content and trailing newline policy (no newline requested initially,
    // but update introduced a newline since we didn't carry the no_newline marker
    // on the final hunk).
    let text = mem.get("some.text").unwrap();
    assert!(text.ends_with('\n'));
    assert!(text.contains("hello, friend"));
}

#[test]
fn update_pure_insert_on_missing_file() {
    let patch = r#"
*** Begin Patch
*** Update File: newfile.rs
@@
+fn main() {}
*** End Patch
"#;
    let ops = parse_patch_ops(patch).expect("parse");
    let mut mem = BTreeMap::<String, String>::new();
    let results = execute_patch_ops_in_memory(&mut mem, ops);
    assert!(
        results
            .iter()
            .any(|r| r["op"] == "update" && r["ok"] == true)
    );
    assert_eq!(mem.get("newfile.rs").unwrap(), "fn main() {}\n");
}

#[test]
fn relaxed_trailing_whitespace_matching() {
    let before = "line 1  \nline 2\t\n";
    let h = Hunk {
        old_lines: vec!["line 1".into(), "line 2".into()],
        new_lines: vec!["line 1x".into(), "line 2y".into()],
    };
    let out = apply_hunk(before, &h).expect("apply");
    assert_eq!(out, "line 1x\nline 2y\n");
}

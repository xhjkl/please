use serde_json::Value;

use super::applying::{apply_all_hunks, apply_hunk};
use super::model::{Hunk, PatchOp};
use super::parsing::{contains_patch_syntax, parse_patch_ops};
use super::{Args, apply};
use crate::test_support::TempDir;

/// Make a workspace in a temporary directory, holding `files`.
fn workspace_with(files: &[(&str, &str)]) -> TempDir {
    let workspace = TempDir::new().unwrap();
    for (path, content) in files {
        std::fs::write(workspace.0.join(path), content).unwrap();
    }
    workspace
}

/// Apply `patch` in `workspace` as the tool does and return each operation's result.
fn patched(workspace: &TempDir, patch: &str) -> Vec<Value> {
    let args = Args {
        path: None,
        patch: Some(patch.into()),
    };
    let result = apply(&workspace.0, args);
    assert_eq!(result["mode"], "patch", "{result}");
    result["results"].as_array().cloned().unwrap()
}

/// Whether some `op` operation among `results` succeeded.
fn done(results: &[Value], op: &str) -> bool {
    let succeeded = |result: &Value| result["op"] == op && result["ok"] == true;
    results.iter().any(succeeded)
}

fn read(workspace: &TempDir, path: &str) -> String {
    std::fs::read_to_string(workspace.0.join(path)).unwrap()
}

#[test]
fn pure_parse_missing_markers() {
    let err = parse_patch_ops("*** End Patch").unwrap_err();
    assert!(err.to_lowercase().contains("missing"));
}

#[test]
fn incomplete_patch_marker_still_counts_as_patch_syntax() {
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- a\n+ b\n";
    assert!(contains_patch_syntax(patch));
    let err = parse_patch_ops(patch).unwrap_err();
    assert!(err.contains("*** End Patch"));
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
            let out = apply_all_hunks("a\n", hunks).unwrap();
            assert_eq!(out, "b\n");
        }
        _ => panic!("expected update"),
    }
}

#[test]
fn update_simple_replace() {
    let workspace = workspace_with(&[("text.text", "hello\nworld\n")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- hello\n+ hello there\n@@\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "hello there\nworld\n");
}

#[test]
fn update_trailing_whitespace_ignored() {
    let workspace = workspace_with(&[("text.text", "hello  \nworld\n")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- hello\n+ hi\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "hi\nworld\n");
}

#[test]
fn update_with_context_space_missing() {
    // Context lines may miss leading space, treat as context
    let workspace = workspace_with(&[("text.text", "A\nB\nC\n")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\nA\n- B\n+ BB\nC\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "A\nBB\nC\n");
}

#[test]
fn update_pure_insert_at_eof() {
    let workspace = workspace_with(&[("text.text", "line1\n")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n+ line2\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "line1\nline2\n");
}

#[test]
fn update_handles_backslash_commentary() {
    let workspace = workspace_with(&[("text.text", "a\n")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- a\n+ b\n\\ No newline at end of file\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "b");
}

#[test]
fn update_ignores_no_newline_marker_for_removed_line() {
    let workspace = workspace_with(&[("text.text", "a")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- a\n\\ No newline at end of file\n+ b\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "b\n");
}

#[test]
fn add_file_basic_and_fenced() {
    let workspace = workspace_with(&[]);
    let patch = "*** Begin Patch\n*** Add File: some.text\nhello\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "add"));
    // default: ensure single trailing newline
    assert_eq!(read(&workspace, "some.text"), "hello\n");

    let fenced = "*** Begin Patch\n*** Add File: another.text\n```\nhello\n```\n*** End Patch\n";
    assert!(done(&patched(&workspace, fenced), "add"));
    assert_eq!(read(&workspace, "another.text"), "hello\n");
}

#[test]
fn add_file_no_newline_escape_hatch() {
    let workspace = workspace_with(&[]);
    let patch = "*** Begin Patch\n*** Add File: nonewline.text\nhello\n\\ No newline at end of file\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "add"));
    assert_eq!(read(&workspace, "nonewline.text"), "hello");
}

#[test]
fn no_newline_marker_is_case_and_spacing_tolerant() {
    let workspace = workspace_with(&[]);
    let patch =
        "*** Begin Patch\n*** Add File: nonewline.text\nhello\n\\ NO NEW LINE\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "add"));
    assert_eq!(read(&workspace, "nonewline.text"), "hello");
}

#[test]
fn add_file_keeps_plain_no_newline_words() {
    let workspace = workspace_with(&[]);
    let patch = "*** Begin Patch\n*** Add File: notes.text\nnot a marker: no new line\nstill content\n*** End Patch\n";
    assert!(done(&patched(&workspace, patch), "add"));
    assert_eq!(
        read(&workspace, "notes.text"),
        "not a marker: no new line\nstill content\n"
    );
}

#[test]
fn delete_file_tolerates_missing() {
    let workspace = workspace_with(&[("present.text", "x\n")]);
    let patch = "*** Begin Patch\n*** Delete File: missing.text\n*** Delete File: present.text\n*** End Patch\n";
    let results = patched(&workspace, patch);
    assert!(results.iter().all(|result| result["ok"] == true));
    assert!(!workspace.0.join("present.text").exists());
}

#[test]
fn multiple_ops_in_one_patch() {
    let workspace = workspace_with(&[("some.text", "A\n")]);
    let patch = "*** Begin Patch\n*** Update File: some.text\n@@\n- A\n+ AA\n*** Add File: another.text\nB\n*** End Patch\n";
    let results = patched(&workspace, patch);
    assert!(done(&results, "update") && done(&results, "add"));
    assert_eq!(read(&workspace, "some.text"), "AA\n");
    // default newline normalization applies to Add without commentary
    assert_eq!(read(&workspace, "another.text"), "B\n");
}

#[test]
fn crlf_and_cr_inputs_normalized() {
    let workspace = workspace_with(&[("text.text", "hello\n")]);
    let patch = "*** Begin Patch\r\n*** Update File: text.text\r\n@@\r\n- hello\r\n+ hi\r\n*** End Patch\r\n";
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "hi\n");
}

#[test]
fn tolerant_casing_and_spacing_in_markers() {
    let workspace = workspace_with(&[("text.text", "hello\n")]);
    let patch = r#"
***    begin   PATCH
*** update file: text.text
@@
- hello
+ hi
***   END   patch
"#;
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "text.text"), "hi\n");
}

#[test]
fn add_update_and_delete_apply_in_order() {
    let workspace = workspace_with(&[]);
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
    let results = patched(&workspace, patch);
    // The delete of a missing file is tolerated.
    assert!(done(&results, "add") && done(&results, "update") && done(&results, "delete"));
    // The add asked for no trailing newline, but the update's final hunk did not carry the
    // marker, so the update put one back.
    assert_eq!(read(&workspace, "some.text"), "hello, friend\nworld\n");
}

#[test]
fn update_pure_insert_on_missing_file() {
    let workspace = workspace_with(&[]);
    let patch = r#"
*** Begin Patch
*** Update File: newfile.rs
@@
+fn main() {}
*** End Patch
"#;
    assert!(done(&patched(&workspace, patch), "update"));
    assert_eq!(read(&workspace, "newfile.rs"), "fn main() {}\n");
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

#[test]
fn only_the_end_marker_itself_ends_a_patch() {
    let workspace = workspace_with(&[]);
    let patch = "*** Begin Patch\n*** Add File: src/patch/frontend.rs\nfn end() {}\n*** Add File: src/patch/backend.rs\nfn begin() {}\n***  END   patch\n";
    assert_eq!(patched(&workspace, patch).len(), 2);
    // The directories a new file lies in are made with it.
    assert_eq!(read(&workspace, "src/patch/frontend.rs"), "fn end() {}\n");
    assert_eq!(read(&workspace, "src/patch/backend.rs"), "fn begin() {}\n");
    assert!(parse_patch_ops("*** Begin Patch\n*** End of the patch\n").is_err());
}

#[test]
fn context_naming_newlines_is_no_newline_marker() {
    let workspace = workspace_with(&[("text.text", "// Normalize newlines\na\n")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n// Normalize newlines\n- a\n+ b\n*** End Patch\n";
    patched(&workspace, patch);
    assert_eq!(read(&workspace, "text.text"), "// Normalize newlines\nb\n");

    let patch =
        "*** Begin Patch\n*** Add File: notes.text\nhello\n// no newline here\n*** End Patch\n";
    patched(&workspace, patch);
    assert_eq!(
        read(&workspace, "notes.text"),
        "hello\n// no newline here\n"
    );
}

#[test]
fn a_hunk_that_does_not_match_fails_its_operation_alone() {
    let workspace = workspace_with(&[("text.text", "hello\n")]);
    let patch = "*** Begin Patch\n*** Update File: text.text\n@@\n- goodbye\n+ hi\n*** Add File: other.text\nB\n*** End Patch\n";
    let results = patched(&workspace, patch);
    assert_eq!(results[0]["ok"], false, "{results:?}");
    assert_eq!(results[0]["errors"][0]["hunk"], 0);
    assert_eq!(read(&workspace, "text.text"), "hello\n");
    assert!(done(&results, "add"));
    assert_eq!(read(&workspace, "other.text"), "B\n");
}

#[test]
fn nothing_is_written_outside_the_workspace() {
    let outer = workspace_with(&[]);
    let workspace = outer.0.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let escape = |path: &str, patch: &str| {
        let args = Args {
            path: Some(path.into()),
            patch: Some(patch.into()),
        };
        apply(&workspace, args)
    };
    let added = escape(
        "",
        "*** Begin Patch\n*** Add File: ../escaped.text\nx\n*** End Patch\n",
    );
    assert_eq!(added["results"][0]["ok"], false, "{added}");
    let overwritten = escape("../escaped.text", "x");
    assert!(overwritten["error"].is_string(), "{overwritten}");
    let absolute = outer.0.join("escaped.text");
    let overwritten = escape(absolute.to_str().unwrap(), "x");
    assert!(overwritten["error"].is_string(), "{overwritten}");
    assert!(!absolute.exists());
}

#[test]
fn text_without_markers_overwrites_the_named_file() {
    let workspace = workspace_with(&[("old.text", "before\n")]);
    let overwrite = |path: Option<&str>| {
        let args = Args {
            path: path.map(String::from),
            patch: Some("after, verbatim".into()),
        };
        apply(&workspace.0, args)
    };
    let result = overwrite(Some("old.text"));
    assert_eq!(result["mode"], "overwrite", "{result}");
    assert_eq!(read(&workspace, "old.text"), "after, verbatim");
    let result = overwrite(Some("deep/down/new.text"));
    assert_eq!(result["ok"], true, "{result}");
    assert_eq!(read(&workspace, "deep/down/new.text"), "after, verbatim");
    // Without a path there is nothing to overwrite.
    assert!(overwrite(None)["error"].is_string());
}

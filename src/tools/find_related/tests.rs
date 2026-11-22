use super::index::{chunks, search, terms};
use crate::test_support::TempDir;
use std::fs;

#[test]
fn terms_split_identifiers_into_their_parts() {
    assert_eq!(
        terms("resolve_path_within_cwd(HubBusy, parseHTTPHeader)"),
        [
            "resolve_path_within_cwd",
            "resolve",
            "path",
            "within",
            "cwd",
            "hubbusy",
            "hub",
            "busy",
            "parsehttpheader",
            "parse",
            "http",
            "header",
        ]
    );
    assert_eq!(terms("a + MAX_ROWS"), ["max_rows", "max", "rows"]);
}

#[test]
fn chunks_cover_every_line_and_prefer_blank_boundaries() {
    let mut text = String::new();
    for block in 0..10 {
        for line in 0..15 {
            text += &format!("let value_{block}_{line} = {line};\n");
        }
        text += "\n";
    }
    let chunks = chunks("src/values.rs", &text);
    assert_eq!(chunks.first().map(|c| c.first), Some(1));
    assert_eq!(chunks.last().map(|c| c.last), Some(text.lines().count()));
    for pair in chunks.windows(2) {
        assert_eq!(pair[0].last + 1, pair[1].first);
    }
    for chunk in &chunks {
        assert!(chunk.last + 1 - chunk.first <= 60);
        assert!(text.lines().nth(chunk.last - 1).unwrap().is_empty());
    }
}

#[test]
fn search_ranks_the_file_that_matches_best_and_shows_its_lines() {
    let dir = TempDir::new().unwrap();
    fs::create_dir(dir.0.join("src")).unwrap();
    fs::write(
        dir.0.join("src/clipping.rs"),
        "//! Which conversation messages a prompt drops once the context fills.\n\
         pub fn choose(history: &[Message], limit: usize) -> Vec<usize> {\n    todo!()\n}\n",
    )
    .unwrap();
    fs::write(
        dir.0.join("src/spinner.rs"),
        "//! Progress indicator while the model thinks.\npub fn spin() {}\n",
    )
    .unwrap();
    fs::write(dir.0.join("Cargo.lock"), "context messages drop fills\n").unwrap();

    let report = search(&dir.0, "drop old messages when the context fills up").unwrap();
    let first = report.lines().next().unwrap();
    assert!(first.ends_with("src/clipping.rs:1-4"), "{report}");
    assert!(
        report.contains("    1  //! Which conversation messages"),
        "{report}"
    );
    assert!(!report.contains("Cargo.lock"), "{report}");

    let report = search(&dir.0, "nothing here matches zzzz").unwrap();
    assert!(report.starts_with("No file shares a word"), "{report}");
}

#[test]
fn search_sees_files_changed_since_the_last_search() {
    let dir = TempDir::new().unwrap();
    let path = dir.0.join("notes.txt");
    fs::write(&path, "alpha\n").unwrap();
    assert!(search(&dir.0, "alpha").unwrap().contains("notes.txt:1-1"));
    fs::write(&path, "alpha\nbeta gamma\n").unwrap();
    let report = search(&dir.0, "gamma").unwrap();
    assert!(report.contains("notes.txt:1-2"), "{report}");
    assert!(report.contains("    2  beta gamma"), "{report}");
}

#[tokio::test]
async fn find_related_refuses_paths_outside_the_workspace() {
    let commands = crate::tools::Commands::default();
    let context = crate::tools::ToolContext::new(&commands, None);
    let args = serde_json::json!({ "query": "anything", "path": "/" });
    let args = serde_json::from_value(args).unwrap();
    let result = super::call(args, context).await;
    assert!(result.get("error").is_some(), "{result}");
}

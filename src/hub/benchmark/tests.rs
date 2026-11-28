use super::*;
use crate::test_support::TempDir;

#[test]
fn cases_are_rows_whose_input_names_a_file_beside_them() -> Result<()> {
    let directory = TempDir::new()?;
    let path = directory.0.join("cases.tsv");
    std::fs::write(directory.0.join("notes.txt"), "piped\ndata\n")?;
    let rows = "request\tinput\tmax_tokens\tname\n\nSay hello.\t\t64\thello\nSum these up.\tnotes.txt\t256\tsummary\n";
    std::fs::write(&path, rows)?;
    let hello = Case {
        name: "hello".into(),
        max_tokens: 64,
        messages: vec![Message::User("Say hello.".into())],
    };
    let summary = Case {
        name: "summary".into(),
        max_tokens: 256,
        messages: vec![
            Message::Input("piped\ndata\n".into()),
            Message::User("Sum these up.".into()),
        ],
    };
    assert_eq!(cases(&path)?, [hello, summary]);

    std::fs::write(&path, "name\trequest\nhello\tSay hello.\n")?;
    let headless = cases(&path).unwrap_err().to_string();
    assert!(headless.contains("max_tokens"), "{headless}");
    std::fs::write(
        &path,
        "name\tmax_tokens\trequest\nhello\tmany\tSay hello.\n",
    )?;
    assert!(cases(&path).is_err());
    Ok(())
}

#[test]
fn a_row_has_a_cell_for_every_column() -> Result<()> {
    let measured = |tokens: Vec<u32>, times: Vec<f64>| Exchange {
        generation: inference::Generation {
            start: inference::Start::Prefilled,
            prompt_tokens: 2600,
            reused_tokens: 0,
            complete: true,
        },
        tokens,
        times,
        first_visible: None,
        answer: String::new(),
        thinking: String::new(),
        calls: Ok(vec![]),
    };
    let row = measured(vec![7, 8, 9], vec![0.5, 1.0, 1.5]).row("hello", "first")?;
    let cells: Vec<&str> = row.split('\t').collect();
    assert_eq!(cells.len(), COLUMNS.len());
    assert_eq!(
        cells[..6],
        ["hello", "first", "prefilled", "2600", "0", "3"]
    );
    // Two tokens after the first, a second apart in all; nothing visible, and no calls.
    assert_eq!(cells[7..11], ["-", "2.0", "1.500", "-"]);
    // The same tokens name the same output, whenever they arrive.
    let again = measured(vec![7, 8, 9], vec![0.1, 0.2, 0.3]).row("hello", "first")?;
    assert_eq!(again.split('\t').next_back(), cells.last().copied());
    let other = measured(vec![7, 8, 10], vec![0.5, 1.0, 1.5]).row("hello", "first")?;
    assert_ne!(other.split('\t').next_back(), cells.last().copied());
    assert!(measured(vec![], vec![]).row("hello", "first").is_err());
    Ok(())
}

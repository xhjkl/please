use super::model::{Hunk, PatchOp};
use super::text::normalize_eol;

/// The lines that open and close a patch, lowercase and without whitespace, as `find_marker`
/// compares them.
const BEGIN: &str = "***beginpatch";
const END: &str = "***endpatch";

#[derive(Copy, Clone)]
enum Header {
    Update,
    Add,
    Delete,
}

pub fn parse_patch_ops(raw: &str) -> Result<Vec<PatchOp>, String> {
    let src = normalize_eol(raw);
    let lines: Vec<&str> = src.lines().collect();
    let begin = find_marker(&lines, 0, BEGIN).ok_or("Missing *** Begin Patch")?;
    let end = find_marker(&lines, begin + 1, END).ok_or("Missing *** End Patch")?;

    let mut ops = Vec::new();
    let mut i = begin + 1;
    while i < end {
        let header = parse_header(lines[i]);
        i += 1;
        let Some((header, path)) = header else {
            continue;
        };
        let op = match header {
            Header::Update => {
                let (hunks, no_newline) = parse_update_hunks(&lines, &mut i, end);
                PatchOp::Update {
                    path,
                    hunks,
                    no_newline,
                }
            }
            Header::Add => {
                let (content, no_newline) = parse_add_block(&lines, &mut i, end);
                PatchOp::Add {
                    path,
                    content,
                    no_newline,
                }
            }
            Header::Delete => PatchOp::Delete { path },
        };
        ops.push(op);
    }
    Ok(ops)
}

pub fn contains_patch_syntax(s: &str) -> bool {
    let src = normalize_eol(s);
    let lines: Vec<&str> = src.lines().collect();
    find_marker(&lines, 0, BEGIN).is_some()
        || find_marker(&lines, 0, END).is_some()
        || lines
            .iter()
            .any(|line| line.trim().starts_with("***") && parse_header(line).is_some())
}

/// Find the first line from `from` that is `marker`, in any case or spacing. Nothing looser
/// counts: `*** Add File: src/patch/frontend.rs` names "end" and "patch" too.
fn find_marker(lines: &[&str], from: usize, marker: &str) -> Option<usize> {
    let found = lines[from..]
        .iter()
        .position(|line| squeezed(line) == marker)?;
    Some(from + found)
}

/// Lowercase a line and drop its whitespace, to compare it with a marker in any case or spacing.
fn squeezed(line: &str) -> String {
    let line: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    line.to_ascii_lowercase()
}

/// The operation a header line starts and the path it names.
fn parse_header(line: &str) -> Option<(Header, String)> {
    [Header::Update, Header::Add, Header::Delete]
        .into_iter()
        .find_map(|header| Some((header, parse_header_path(line, header)?)))
}

fn parse_header_path(line: &str, h: Header) -> Option<String> {
    let l = line.trim().trim_start_matches('*').trim();
    let kw = match h {
        Header::Update => "update file",
        Header::Add => "add file",
        Header::Delete => "delete file",
    };
    let compact = l.to_ascii_lowercase().replace(' ', "");
    if !compact.starts_with(&kw.replace(' ', "")) {
        return None;
    }
    let after = match l.find(':') {
        Some(pos) => &l[pos + 1..],
        // Out of range when the keyword's space is missing too, as in `*** AddFile`.
        None => l.get(kw.len()..)?,
    };
    let path = after.trim().trim_matches('"');
    if path.is_empty() {
        None
    } else {
        Some(path.to_string())
    }
}

fn parse_update_hunks(lines: &[&str], i: &mut usize, end: usize) -> (Vec<Hunk>, bool) {
    let mut hunks: Vec<Hunk> = Vec::new();
    if *i < end && lines[*i].trim_start().starts_with("```") {
        *i += 1;
    }

    let mut cur = Hunk::default();
    let mut no_newline = false;
    let mut last_line_survives = false;
    while *i < end {
        let raw = lines[*i];
        let t = raw.trim_start();
        if t.starts_with("***") {
            break;
        }
        *i += 1;
        if t.starts_with("```") {
            break;
        }

        if t.starts_with("@@") {
            if !cur.is_empty() {
                hunks.push(std::mem::take(&mut cur));
            }
            last_line_survives = false;
        } else if let Some(line) = raw.strip_prefix("+ ").or_else(|| raw.strip_prefix('+')) {
            cur.new_lines.push(line.to_string());
            last_line_survives = true;
        } else if let Some(line) = raw.strip_prefix("- ").or_else(|| raw.strip_prefix('-')) {
            cur.old_lines.push(line.to_string());
            last_line_survives = false;
        } else if !raw.starts_with(' ') && is_no_newline_comment_line(raw) {
            if last_line_survives {
                no_newline = true;
            }
            last_line_survives = false;
        } else {
            let line = raw.strip_prefix(' ').unwrap_or(raw);
            cur.old_lines.push(line.to_string());
            cur.new_lines.push(line.to_string());
            last_line_survives = true;
        }
    }
    if !cur.is_empty() {
        hunks.push(cur);
    }
    (hunks, no_newline)
}

fn parse_add_block(lines: &[&str], i: &mut usize, end: usize) -> (String, bool) {
    let mut out: Vec<&str> = Vec::new();
    let fenced = *i < end && lines[*i].trim_start().starts_with("```");
    if fenced {
        *i += 1;
    }
    while *i < end {
        let line = lines[*i];
        let t = line.trim_start();
        if !fenced && t.starts_with("***") {
            break;
        }
        *i += 1;
        if fenced && t.starts_with("```") {
            break;
        }
        out.push(line);
    }

    while out.last().is_some_and(|line| line.trim().is_empty()) {
        out.pop();
    }
    let no_newline = out
        .last()
        .is_some_and(|line| is_no_newline_comment_line(line));
    if no_newline {
        out.pop();
    }
    (out.join("\n"), no_newline)
}

/// Recognize the `\ No newline at end of file` marker in any case or spacing, its backslash
/// and its "at end of file" optional. Nothing else counts, so content like `// Normalize
/// newlines` stays content.
fn is_no_newline_comment_line(line: &str) -> bool {
    let line = squeezed(line);
    let line = line.strip_prefix('\\').unwrap_or(&line);
    matches!(line.strip_prefix("nonewline"), Some("" | "atendoffile"))
}

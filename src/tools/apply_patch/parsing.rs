use super::model::{Hunk, PatchOp};
use super::text::normalize_eol;

#[derive(Copy, Clone)]
enum Marker {
    Begin,
    End,
}

#[derive(Copy, Clone)]
enum Header {
    Update,
    Add,
    Delete,
}

pub fn parse_patch_ops(raw: &str) -> Result<Vec<PatchOp>, String> {
    let src = normalize_eol(raw);
    let lines: Vec<&str> = src.lines().collect();

    let mut i = match find_marker(&lines, 0, Marker::Begin) {
        Some(idx) => idx + 1,
        None => return Err("Missing *** Begin Patch".into()),
    };
    let end = match find_marker(&lines, i, Marker::End) {
        Some(idx) => idx,
        None => return Err("Missing *** End Patch".into()),
    };

    let mut ops: Vec<PatchOp> = Vec::new();
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() {
            i += 1;
            continue;
        }

        if let Some(path) = parse_header_path(line, Header::Update) {
            i += 1;
            let (hunks, no_newline) = parse_update_hunks(&lines, &mut i, end)?;
            ops.push(PatchOp::Update {
                path,
                hunks,
                no_newline,
            });
            continue;
        }
        if let Some(path) = parse_header_path(line, Header::Add) {
            i += 1;
            let (content, no_newline) = parse_add_block(&lines, &mut i, end);
            ops.push(PatchOp::Add {
                path,
                content,
                no_newline,
            });
            continue;
        }
        if let Some(path) = parse_header_path(line, Header::Delete) {
            i += 1;
            ops.push(PatchOp::Delete { path });
            continue;
        }

        i += 1;
    }

    Ok(ops)
}

pub(crate) fn contains_patch_markers(s: &str) -> bool {
    let src = normalize_eol(s);
    let lines: Vec<&str> = src.lines().collect();
    let Some(begin) = find_marker(&lines, 0, Marker::Begin) else {
        return false;
    };
    find_marker(&lines, begin + 1, Marker::End).is_some()
}

fn find_marker(lines: &[&str], mut i: usize, which: Marker) -> Option<usize> {
    while i < lines.len() {
        let t = lines[i].trim();
        let lower = t.to_ascii_lowercase();
        let has_stars = lower.starts_with("***");
        let is_begin = lower.contains("begin") && lower.contains("patch");
        let is_end = lower.contains("end") && lower.contains("patch");
        let ok = match which {
            Marker::Begin => has_stars && is_begin,
            Marker::End => has_stars && is_end,
        };
        if ok {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn parse_header_path(line: &str, h: Header) -> Option<String> {
    let l = line.trim().trim_start_matches('*').trim();
    let kw = match h {
        Header::Update => "update file",
        Header::Add => "add file",
        Header::Delete => "delete file",
    };
    let l_lower = l.to_ascii_lowercase();
    let kw_nospace = kw.replace(' ', "");
    if !(l_lower.starts_with(kw) || l_lower.replace(' ', "").starts_with(&kw_nospace)) {
        return None;
    }

    let after = if let Some(pos) = l.find(':') {
        &l[pos + 1..]
    } else {
        &l[kw.len()..]
    };
    let path = after.trim().trim_matches('"');
    if path.is_empty() {
        None
    } else {
        Some(path.to_string())
    }
}

fn parse_update_hunks(
    lines: &[&str],
    i: &mut usize,
    end: usize,
) -> Result<(Vec<Hunk>, bool), String> {
    let mut hunks: Vec<Hunk> = Vec::new();
    if *i < end && lines[*i].trim_start().starts_with("```") {
        *i += 1;
    }

    let mut cur = Hunk::default();
    let mut have_any = false;
    let mut no_newline = false;

    while *i < end {
        let t = lines[*i].trim_start();
        if t.starts_with("***") || t.eq_ignore_ascii_case("*** end patch") {
            break;
        }
        if t.starts_with("```") {
            *i += 1;
            break;
        }

        if t.starts_with("@@") {
            if have_any {
                hunks.push(cur);
                cur = Hunk::default();
                have_any = false;
            }
            *i += 1;
            continue;
        }

        let raw = lines[*i];
        if let Some(line) = raw.strip_prefix("+ ") {
            cur.new_lines.push(line.to_string());
            have_any = true;
        } else if let Some(line) = raw.strip_prefix("- ") {
            cur.old_lines.push(line.to_string());
            have_any = true;
        } else if let Some(line) = raw.strip_prefix('+') {
            cur.new_lines.push(line.to_string());
            have_any = true;
        } else if let Some(line) = raw.strip_prefix('-') {
            cur.old_lines.push(line.to_string());
            have_any = true;
        } else if let Some(line) = raw.strip_prefix(' ') {
            cur.old_lines.push(line.to_string());
            cur.new_lines.push(line.to_string());
            have_any = true;
        } else if is_no_newline_comment_line(raw) {
            no_newline = true;
        } else {
            cur.old_lines.push(raw.to_string());
            cur.new_lines.push(raw.to_string());
            have_any = true;
        }

        *i += 1;
    }

    if have_any {
        hunks.push(cur);
    }
    if *i < end && lines[*i].trim_start().starts_with("```") {
        *i += 1;
    }
    Ok((hunks, no_newline))
}

fn parse_add_block(lines: &[&str], i: &mut usize, end: usize) -> (String, bool) {
    let mut out: Vec<&str> = Vec::new();
    let mut no_newline = false;
    let fenced = *i < end && lines[*i].trim_start().starts_with("```");
    if fenced {
        *i += 1;
        while *i < end {
            let t = lines[*i].trim_start();
            if t.starts_with("```") {
                *i += 1;
                break;
            }
            out.push(lines[*i]);
            *i += 1;
        }
    } else {
        while *i < end {
            let t = lines[*i].trim_start();
            if t.starts_with("***") || t.eq_ignore_ascii_case("*** end patch") {
                break;
            }
            out.push(lines[*i]);
            *i += 1;
        }
    }

    while let Some(last) = out.last() {
        let trimmed = last.trim();
        if trimmed.is_empty() {
            out.pop();
            continue;
        }
        if is_no_newline_comment_line(trimmed) {
            out.pop();
            no_newline = true;
        }
        break;
    }
    (out.join("\n"), no_newline)
}

// Detects commentary indicating that there should be no trailing newline.
// Tolerant to leading backslash, mixed casing, and minor drift; requires tokens
// "no", then "new", then "line" to appear in order (substring match).
fn is_no_newline_comment_line(s: &str) -> bool {
    let mut t = s.trim();
    if let Some(rest) = t.strip_prefix('\\') {
        t = rest.trim();
    }
    let lower = t.to_ascii_lowercase();

    let mut idx = 0usize;
    for term in ["no", "new", "line"] {
        match lower[idx..].find(term) {
            Some(pos) => {
                idx += pos + term.len();
            }
            None => return false,
        }
    }
    true
}

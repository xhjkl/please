use super::model::Hunk;

pub fn apply_all_hunks(before: &str, hunks: &[Hunk]) -> Result<String, Vec<(usize, String)>> {
    let mut text = before.to_string();
    let mut errors: Vec<(usize, String)> = Vec::new();
    for (idx, h) in hunks.iter().enumerate() {
        match apply_hunk(&text, h) {
            Ok(next) => text = next,
            Err(e) => errors.push((idx, e)),
        }
    }
    if errors.is_empty() {
        Ok(text)
    } else {
        Err(errors)
    }
}

/// Replace the hunk's old lines with its new ones: where they match whole lines, trailing
/// whitespace aside, or else where their text occurs. A hunk without old lines appends.
pub fn apply_hunk(before: &str, h: &Hunk) -> Result<String, String> {
    if h.old_lines.is_empty() {
        let mut out = String::from(before);
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&h.new_lines.join("\n"));
        return Ok(out);
    }

    let mut lines: Vec<&str> = before.split('\n').collect();
    let matches = |window: &[&str]| {
        let mut pairs = window.iter().zip(&h.old_lines);
        pairs.all(|(line, old)| line.trim_end() == old.trim_end())
    };
    let found = lines.windows(h.old_lines.len()).position(matches);
    if let Some(start) = found {
        let end = start + h.old_lines.len();
        lines.splice(start..end, h.new_lines.iter().map(String::as_str));
        let mut out = lines.join("\n");
        if before.ends_with('\n') && !out.ends_with('\n') {
            out.push('\n');
        }
        return Ok(out);
    }

    // Falling back to a substring, which can match within a line.
    let old_seg = h.old_lines.join("\n");
    if before.contains(&old_seg) {
        return Ok(before.replacen(&old_seg, &h.new_lines.join("\n"), 1));
    }
    // Showing the start of the text, on one line.
    let mut shown = old_seg.replace('\n', "\\n");
    if shown.len() > 160 {
        shown.truncate(shown.floor_char_boundary(160));
        shown.push('…');
    }
    Err(format!("hunk old text not found: {shown}"))
}

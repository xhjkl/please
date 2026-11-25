use super::model::Hunk;
use super::text::{find_lines_window, preview};

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
    let old_lines: Vec<&str> = h.old_lines.iter().map(String::as_str).collect();
    if let Some((start, end)) = find_lines_window(&lines, &old_lines) {
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
    Err(format!("hunk old text not found: {}", preview(&old_seg)))
}

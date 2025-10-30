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

    let old_seg = h.old_lines.join("\n");
    let new_seg = h.new_lines.join("\n");

    let before_lines: Vec<&str> = before.split('\n').collect();
    let old_lines: Vec<&str> = old_seg.split('\n').collect();
    let ends_with_nl = before.ends_with('\n');

    if let Some((s, e)) = find_lines_window(&before_lines, &old_lines) {
        let mut owned: Vec<String> = before_lines.iter().map(|s| (*s).to_string()).collect();
        owned.splice(s..e, h.new_lines.clone());
        let mut out = owned.join("\n");
        if ends_with_nl && !out.ends_with('\n') {
            out.push('\n');
        }
        return Ok(out);
    }

    if let Some(pos) = before.find(&old_seg) {
        let mut out = String::with_capacity(before.len() - old_seg.len() + new_seg.len());
        out.push_str(&before[..pos]);
        out.push_str(&new_seg);
        out.push_str(&before[pos + old_seg.len()..]);
        return Ok(out);
    }

    Err(format!("hunk old text not found: {}", preview(&old_seg)))
}

pub fn normalize_eol(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

pub fn set_trailing_newline(s: &str, want_newline: bool) -> String {
    let mut t = s.trim_end_matches('\n').to_string();
    if want_newline {
        t.push('\n');
    }
    t
}

/// Find the first run of `before` that matches `old` line by line, ignoring trailing whitespace.
pub fn find_lines_window(before: &[&str], old: &[&str]) -> Option<(usize, usize)> {
    if old.is_empty() {
        return None;
    }
    let start = before.windows(old.len()).position(|window| {
        window
            .iter()
            .zip(old)
            .all(|(line, wanted)| line.trim_end() == wanted.trim_end())
    })?;
    Some((start, start + old.len()))
}

pub fn preview(s: &str) -> String {
    let s = s.replace('\n', "\\n");
    if s.len() > 160 {
        format!("{}…", &s[..s.floor_char_boundary(160)])
    } else {
        s
    }
}

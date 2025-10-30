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

pub fn find_lines_window(before: &[&str], old: &[&str]) -> Option<(usize, usize)> {
    if old.is_empty() || before.len() < old.len() {
        return None;
    }
    'outer: for start in 0..=before.len() - old.len() {
        for k in 0..old.len() {
            if !eq_line_relaxed(before[start + k], old[k]) {
                continue 'outer;
            }
        }
        return Some((start, start + old.len()));
    }
    None
}

fn eq_line_relaxed(a: &str, b: &str) -> bool {
    a.trim_end() == b.trim_end()
}

pub fn preview(s: &str) -> String {
    let s = s.replace('\n', "\\n");
    if s.len() > 160 {
        format!("{}…", &s[..160])
    } else {
        s
    }
}

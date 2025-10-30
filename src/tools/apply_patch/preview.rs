use super::model;
use super::parse_patch_ops;
use super::parsing;

/// Produce a full diff-like preview for a proposed patch (no truncation).
/// For overwrite mode, returns the full content. For patch mode, returns a
/// unified diff-style representation across all ops.
pub fn summarize_patch_for_preview(raw: &str) -> Option<String> {
    if !parsing::contains_patch_markers(raw) {
        // Overwrite mode: show full content as-is
        return Some(raw.to_string());
    }

    let ops = parse_patch_ops(raw).ok()?;
    // Build a full unified-diff-like preview for all ops
    let mut out = String::new();
    for op in ops.iter() {
        match op {
            model::PatchOp::Add { path, content, .. } => {
                out.push_str("--- /dev/null\n");
                out.push_str(&format!("+++ {path}\n"));
                out.push_str("@@\n");
                for l in content.lines() {
                    out.push('+');
                    out.push_str(l);
                    out.push('\n');
                }
                out.push('\n');
            }
            model::PatchOp::Delete { path } => {
                out.push_str(&format!("--- {path}\n"));
                out.push_str("+++ /dev/null\n");
                out.push_str("@@\n\n");
            }
            model::PatchOp::Update { path, hunks, .. } => {
                out.push_str(&format!("--- {path}\n"));
                out.push_str(&format!("+++ {path}\n"));
                for h in hunks.iter() {
                    out.push_str("@@\n");
                    let n = std::cmp::min(h.old_lines.len(), h.new_lines.len());
                    for i in 0..n {
                        let old = &h.old_lines[i];
                        let newl = &h.new_lines[i];
                        if old == newl {
                            out.push(' ');
                            out.push_str(old);
                            out.push('\n');
                        } else {
                            out.push('-');
                            out.push_str(old);
                            out.push('\n');
                            out.push('+');
                            out.push_str(newl);
                            out.push('\n');
                        }
                    }
                    for i in n..h.old_lines.len() {
                        out.push('-');
                        out.push_str(&h.old_lines[i]);
                        out.push('\n');
                    }
                    for i in n..h.new_lines.len() {
                        out.push('+');
                        out.push_str(&h.new_lines[i]);
                        out.push('\n');
                    }
                    out.push('\n');
                }
            }
        }
    }
    Some(out)
}

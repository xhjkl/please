use super::model::PatchOp;
use super::parsing::{contains_patch_syntax, parse_patch_ops};

/// Render a proposed patch in full for approval: overwrite content as is, and a patch as a
/// unified-diff-like listing of its operations.
pub fn preview_patch(raw: &str) -> String {
    if !contains_patch_syntax(raw) {
        return raw.to_string();
    }
    let ops = match parse_patch_ops(raw) {
        Err(error) => return format!("Invalid patch: {error}\n\n{raw}"),
        Ok(ops) => ops,
    };
    let mut out = String::new();
    for op in ops {
        match op {
            PatchOp::Add { path, content, .. } => {
                out += &format!("--- /dev/null\n+++ {path}\n@@\n");
                for line in content.lines() {
                    out += &format!("+{line}\n");
                }
                out.push('\n');
            }
            PatchOp::Delete { path } => out += &format!("--- {path}\n+++ /dev/null\n@@\n\n"),
            PatchOp::Update { path, hunks, .. } => {
                out += &format!("--- {path}\n+++ {path}\n");
                for hunk in hunks {
                    out += "@@\n";
                    let paired = hunk.old_lines.len().min(hunk.new_lines.len());
                    for (old, new) in hunk.old_lines.iter().zip(&hunk.new_lines) {
                        if old == new {
                            out += &format!(" {old}\n");
                        } else {
                            out += &format!("-{old}\n+{new}\n");
                        }
                    }
                    for old in &hunk.old_lines[paired..] {
                        out += &format!("-{old}\n");
                    }
                    for new in &hunk.new_lines[paired..] {
                        out += &format!("+{new}\n");
                    }
                    out.push('\n');
                }
            }
        }
    }
    out
}

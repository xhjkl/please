use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct Candidate {
    path: PathBuf,
    size_bytes: u64,
    mtime: SystemTime,
}

fn is_gpt_oss_gguf(path: &Path) -> bool {
    let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    let f = fname.to_ascii_lowercase();
    f.contains("gpt-oss") && f.ends_with(".gguf")
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(home) = std::env::var("HOME") {
        roots.push(Path::new(&home).join(".please").join("weights"));
    }

    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }

    roots
}

fn collect_local_gguf_candidates(root: &Path, max_depth: usize, out: &mut Vec<Candidate>) {
    if max_depth < 1 {
        return;
    }
    let Ok(rd) = fs::read_dir(root) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_file() {
            if is_gpt_oss_gguf(&path) {
                tracing::trace!(path=%path.display(), "discovery: found a gguf file");
                out.push(Candidate {
                    path,
                    size_bytes: meta.len(),
                    mtime: meta.modified().unwrap_or(UNIX_EPOCH),
                });
            }
        } else if meta.is_dir() {
            collect_local_gguf_candidates(&path, max_depth - 1, out);
        }
    }
}

pub fn choose_best_model_path() -> Option<PathBuf> {
    let mut candidates: Vec<Candidate> = Vec::new();

    for root in candidate_roots() {
        collect_local_gguf_candidates(&root, 4, &mut candidates);
    }

    if candidates.is_empty() {
        return None;
    }

    // Choose largest, break ties by freshness.
    candidates.sort_by(|a, b| match b.size_bytes.cmp(&a.size_bytes) {
        Ordering::Equal => b.mtime.cmp(&a.mtime),
        other => other,
    });

    candidates.into_iter().next().map(|c| c.path)
}

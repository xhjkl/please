use std::cmp::Ordering;
use std::fs;
use std::io::BufReader;
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

fn collect_ollama_candidates(home: &Path, out: &mut Vec<Candidate>) {
    tracing::trace!(?home, "discovery: collecting ollama candidates");
    let manifests_root = home
        .join(".ollama")
        .join("models")
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join("gpt-oss");
    let Ok(tags) = fs::read_dir(&manifests_root) else {
        return;
    };
    for tag_entry in tags.flatten() {
        let manifest_path = tag_entry.path();
        let Ok(meta) = fs::metadata(&manifest_path) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let Ok(file) = fs::File::open(&manifest_path) else {
            continue;
        };
        let reader = BufReader::new(file);
        let Ok(json) = serde_json::from_reader::<_, serde_json::Value>(reader) else {
            continue;
        };
        let layers = json
            .get("layers")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        for layer in layers {
            let media_type = layer
                .get("mediaType")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if media_type != "application/vnd.ollama.image.model" {
                continue;
            }
            let size = layer.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            if size == 0 {
                continue;
            }
            // derive from digest: `~/.ollama/models/blobs/sha256-<hex>`
            // because "from" may refer to nonexistent user
            let Some(digest) = layer.get("digest").and_then(|v| v.as_str()) else {
                continue;
            };
            let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
            let blob_path = home
                .join(".ollama")
                .join("models")
                .join("blobs")
                .join(format!("sha256-{hex}"));
            tracing::trace!(?blob_path, "discovery: found a blob path");
            let (mtime, size_bytes) = match fs::metadata(&blob_path) {
                Ok(bm) => (bm.modified().unwrap_or(UNIX_EPOCH), bm.len()),
                Err(_) => (meta.modified().unwrap_or(UNIX_EPOCH), size),
            };
            out.push(Candidate {
                path: blob_path,
                size_bytes,
                mtime,
            });
        }
    }
}

pub fn choose_best_model_path() -> Option<PathBuf> {
    let mut candidates: Vec<Candidate> = Vec::new();

    if std::env::var("PLEASE_SALVAGE").is_ok()
        && let Ok(home) = std::env::var("HOME")
    {
        tracing::trace!(?home, "discovery: collecting ollama candidates");
        collect_ollama_candidates(Path::new(&home), &mut candidates);
    }

    for root in candidate_roots() {
        collect_local_gguf_candidates(&root, 4, &mut candidates);
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| match b.size_bytes.cmp(&a.size_bytes) {
        Ordering::Equal => b.mtime.cmp(&a.mtime),
        other => other,
    });

    candidates.into_iter().next().map(|c| c.path)
}

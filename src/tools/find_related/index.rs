//! Lexical ranking of a directory's text: Okapi BM25 over identifiers, their snake_case and
//! camelCase parts, and file paths, scored per chunk of whole lines and reported per file.
//!
//! Files come from `git ls-files` (tracked and untracked, minus ignored) or, outside git, a walk
//! that skips hidden and build directories. The last scope's chunks stay cached for the process,
//! each file's keyed by its size and modification time, so a repeated search re-reads only the
//! files that changed.
use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::SystemTime;

/// Chunks end after this many lines or bytes, preferring a blank line past `MIN_LINES`.
const MAX_LINES: usize = 60;
const MIN_LINES: usize = 12;
const MAX_BYTES: usize = 1600;
/// Larger files are data or generated, not code worth ranking.
const MAX_FILE_BYTES: u64 = 256 * 1024;
/// Text one search reads at most; later files are skipped and the result says so.
const MAX_TOTAL_BYTES: u64 = 64 << 20;
/// Files reported per search, and lines shown per file.
const SHOWN_FILES: usize = 5;
const SHOWN_LINES: usize = 6;
const SHOWN_WIDTH: usize = 120;

/// A run of whole lines with the terms in it and in its file's path.
pub(super) struct Chunk {
    /// One-based numbers of the first and last line.
    pub first: usize,
    pub last: usize,
    terms: HashMap<String, u32>,
    length: u32,
}

/// One file's chunks, valid while its size and modification time stay the same.
struct File {
    size: u64,
    modified: SystemTime,
    chunks: Vec<Chunk>,
}

/// The files of the last searched scope.
struct Snapshot {
    scope: PathBuf,
    files: HashMap<String, File>,
}

static SNAPSHOT: Mutex<Option<Snapshot>> = Mutex::new(None);

/// Rank the text files under `scope` against `query`; report the best as the tool's result.
pub fn search(scope: &Path, query: &str) -> io::Result<String> {
    let mut held = SNAPSHOT.lock().unwrap_or_else(|error| error.into_inner());
    let mut cached = match held.take() {
        Some(snapshot) if snapshot.scope == scope => snapshot.files,
        _ => HashMap::new(),
    };
    let listed = list(scope)?;
    let (mut files, mut total, mut skipped) = (HashMap::new(), 0, 0);
    for path in &listed {
        let full = scope.join(path);
        let Ok(metadata) = std::fs::metadata(&full) else {
            continue; // deleted but still tracked
        };
        if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES || generated(path) {
            continue;
        }
        if total + metadata.len() > MAX_TOTAL_BYTES {
            skipped += 1;
            continue;
        }
        total += metadata.len();
        let modified = metadata.modified()?;
        let file = match cached.remove(path) {
            Some(file) if file.size == metadata.len() && file.modified == modified => file,
            _ => File {
                size: metadata.len(),
                modified,
                chunks: read(&full).map_or_else(Vec::new, |text| chunks(path, &text)),
            },
        };
        files.insert(path.clone(), file);
    }
    let weights = weights(&files, query);
    let mut report = present(scope, &rank(&files, &weights), &weights);
    if skipped > 0 {
        report += &format!(
            "Searched {} of {} files before reaching the {} MiB limit; pass `path` to search a subdirectory.\n",
            listed.len() - skipped,
            listed.len(),
            MAX_TOTAL_BYTES >> 20,
        );
    }
    *held = Some(Snapshot {
        scope: scope.to_owned(),
        files,
    });
    Ok(report)
}

/// Relative paths of the files under `scope`, sorted.
fn list(scope: &Path) -> io::Result<Vec<String>> {
    let listed = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(scope)
        .output();
    let mut paths = match listed {
        Ok(listed) if listed.status.success() => listed
            .stdout
            .split(|&byte| byte == 0)
            .filter_map(|path| String::from_utf8(path.to_vec()).ok())
            .filter(|path| !path.is_empty())
            .collect(),
        _ => {
            let mut paths = Vec::new();
            walk(scope, "", &mut paths)?;
            paths
        }
    };
    paths.sort();
    Ok(paths)
}

fn walk(scope: &Path, prefix: &str, paths: &mut Vec<String>) -> io::Result<()> {
    for entry in std::fs::read_dir(scope.join(prefix))? {
        let entry = entry?;
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let path = match prefix {
            "" => name.clone(),
            _ => format!("{prefix}/{name}"),
        };
        let kind = entry.file_type()?;
        let skipped = name.starts_with('.') || matches!(name.as_str(), "target" | "node_modules");
        if kind.is_dir() && !skipped {
            walk(scope, &path, paths)?;
        } else if kind.is_file() {
            paths.push(path);
        }
    }
    Ok(())
}

/// Lockfiles and minified bundles, whose matches would only crowd out source.
fn generated(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.ends_with(".lock")
        || name.ends_with("-lock.json")
        || name.ends_with(".min.js")
        || name == "go.sum"
}

/// A file's text, unless it is binary or not UTF-8.
fn read(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    if bytes[..bytes.len().min(8000)].contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

/// Cut a file into chunks of whole lines, skipping chunks with no visible text.
pub(super) fn chunks(path: &str, text: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let path_terms = terms(path);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let (mut end, mut bytes, mut blank) = (start, 0, None);
        while end < lines.len()
            && end - start < MAX_LINES
            && (end == start || bytes + lines[end].len() < MAX_BYTES)
        {
            if lines[end].trim().is_empty() && end - start >= MIN_LINES {
                blank = Some(end);
            }
            bytes += lines[end].len() + 1;
            end += 1;
        }
        if let (true, Some(blank)) = (end < lines.len(), blank) {
            end = blank + 1;
        }
        let body = &lines[start..end];
        if body.iter().any(|line| !line.trim().is_empty()) {
            let mut counts: HashMap<String, u32> = HashMap::new();
            let mut length = 0;
            let body_terms = body.iter().flat_map(|line| terms(line));
            for term in path_terms.iter().cloned().chain(body_terms) {
                *counts.entry(term).or_default() += 1;
                length += 1;
            }
            chunks.push(Chunk {
                first: start + 1,
                last: end,
                terms: counts,
                length,
            });
        }
        start = end;
    }
    chunks
}

/// Lowercased search terms: each identifier, plus its snake_case and camelCase parts.
pub(super) fn terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for word in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        let whole = word.trim_matches('_');
        if whole.chars().count() < 2 {
            continue;
        }
        terms.push(whole.to_lowercase());
        let mut parts = Vec::new();
        for piece in whole.split('_').filter(|piece| !piece.is_empty()) {
            let chars: Vec<char> = piece.chars().collect();
            let mut begin = 0;
            for i in 1..chars.len() {
                let lower_to_upper = chars[i - 1].is_lowercase() && chars[i].is_uppercase();
                let acronym_end = chars[i - 1].is_uppercase()
                    && chars[i].is_uppercase()
                    && chars.get(i + 1).is_some_and(|c| c.is_lowercase());
                if lower_to_upper || acronym_end {
                    parts.push(chars[begin..i].iter().collect::<String>());
                    begin = i;
                }
            }
            parts.push(chars[begin..].iter().collect());
        }
        if parts.len() > 1 {
            let parts = parts.into_iter().filter(|part| part.chars().count() >= 2);
            terms.extend(parts.map(|part| part.to_lowercase()));
        }
    }
    terms
}

/// A file's best chunk and how well it matched the query.
struct Hit<'a> {
    path: &'a str,
    chunk: &'a Chunk,
    score: f32,
}

/// The query's terms, each weighted by how rare it is among all chunks (BM25's idf).
fn weights(files: &HashMap<String, File>, query: &str) -> HashMap<String, f32> {
    let all = || files.values().flat_map(|file| &file.chunks);
    let n = all().count() as f32;
    let wanted: HashSet<String> = terms(query).into_iter().collect();
    wanted
        .into_iter()
        .map(|term| {
            let df = all()
                .filter(|chunk| chunk.terms.contains_key(&term))
                .count() as f32;
            (term, ((n - df + 0.5) / (df + 0.5) + 1.0).ln())
        })
        .collect()
}

/// Each file's best chunk by BM25, best first. A semantic ranking would merge with this
/// order, by reciprocal rank say, before the hits are presented.
fn rank<'a>(files: &'a HashMap<String, File>, weights: &HashMap<String, f32>) -> Vec<Hit<'a>> {
    let all = || files.values().flat_map(|file| &file.chunks);
    let n = all().count() as f32;
    let average = all().map(|chunk| chunk.length as f32).sum::<f32>() / n.max(1.0);
    let (k1, b) = (1.2, 0.75);
    let score = |chunk: &Chunk| -> f32 {
        let norm = k1 * (1.0 - b + b * chunk.length as f32 / average);
        let terms = weights.iter().map(|(term, idf)| {
            let tf = chunk.terms.get(term).copied().unwrap_or(0) as f32;
            idf * tf * (k1 + 1.0) / (tf + norm)
        });
        terms.sum()
    };
    let mut hits: Vec<Hit> = files
        .iter()
        .filter_map(|(path, file)| {
            let scored = file.chunks.iter().map(|chunk| (score(chunk), chunk));
            let (score, chunk) = scored.max_by(|a, b| a.0.total_cmp(&b.0))?;
            (score > 0.0).then_some(Hit { path, chunk, score })
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.path.cmp(b.path)));
    hits
}

/// The best hits as the tool's result: each file's span, then its most relevant lines.
fn present(scope: &Path, hits: &[Hit], weights: &HashMap<String, f32>) -> String {
    if hits.is_empty() {
        return "No file shares a word with the query; try other words or an identifier.\n".into();
    }
    let weight = |line: &str| -> f32 {
        let line: HashSet<String> = terms(line).into_iter().collect();
        line.iter().filter_map(|term| weights.get(term)).sum()
    };
    let mut report = String::new();
    for &Hit { path, chunk, .. } in hits.iter().take(SHOWN_FILES) {
        let shown = match scope == Path::new(".") {
            true => PathBuf::from(path),
            false => scope.join(path),
        };
        report += &format!("{}:{}-{}\n", shown.display(), chunk.first, chunk.last);
        let Some(text) = read(&scope.join(path)) else {
            continue; // changed since it was indexed
        };
        let body = text.lines().enumerate().skip(chunk.first - 1);
        let mut lines: Vec<(usize, &str, f32)> = body
            .take(chunk.last + 1 - chunk.first)
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| (index + 1, line, weight(line)))
            .collect();
        // Stable: among equal weights, including none, earlier lines stay.
        lines.sort_by(|a, b| b.2.total_cmp(&a.2));
        lines.truncate(SHOWN_LINES);
        lines.sort_by_key(|&(number, _, _)| number);
        for (number, line, _) in lines {
            let line: String = line.trim_end().chars().take(SHOWN_WIDTH).collect();
            report += &format!("{number:>5}  {line}\n");
        }
    }
    report
}

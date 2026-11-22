use std::fs;
use std::path::{Component, Path, PathBuf};
use std::{env, io};

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
#[derive(Debug, Clone)]
pub enum ParamType {
    String,
    StringArray,
    Choice(&'static [&'static str]),
    Number,
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
#[derive(Clone)]
pub struct Param {
    pub name: &'static str,
    pub desc: &'static str,
    pub param_type: ParamType,
    pub required: bool,
}

/// Resolve a user-supplied path to a relative path confined to the current working
/// directory ("workspace").
///
/// - Accepts relative paths (e.g., `./foo/../bar`) and collapses `.` / `..` without
///   allowing traversal above the workspace root.
/// - Accepts absolute paths **only if** they resolve (after following symlinks) under
///   the canonicalized CWD; such absolute inputs are converted to workspace-relative.
/// - Follows symlinks for the deepest existing ancestor; non-existent trailing segments
///   are preserved (the leaf need not exist).
///
/// Returns a normalized relative `PathBuf` under the workspace root (never leaks the
/// absolute CWD).
///
/// # Errors
/// - `PermissionDenied` if the path is absolute/outside the workspace or escapes via
///   `..`/symlinks resolution.
/// - Propagates I/O errors (e.g., from canonicalization of existing ancestors).
pub fn resolve_path_within_cwd(path: &str) -> io::Result<PathBuf> {
    let root = env::current_dir()?.canonicalize()?; // canonicalized workspace root
    let input = Path::new(path);

    // Empty or current directory resolves to "." (relative root).
    if path.is_empty() || input == Path::new(".") {
        return Ok(PathBuf::from("."));
    }

    // Absolute input: soft-canonicalize, then ensure containment and convert to relative.
    if input.is_absolute() {
        let abs = soft_canonicalize(input)?;
        if !abs.starts_with(&root) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "absolute paths must resolve under the workspace",
            ));
        }
        let rel = abs
            .strip_prefix(&root)
            .unwrap_or(Path::new(""))
            .to_path_buf();
        // Normalize leading separator away; ensure we return a clean relative path.
        let clean = if rel.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            rel
        };
        return Ok(clean);
    }

    // Relative input: collapse components without escaping above the root.
    let mut rel = PathBuf::new();
    for c in input.components() {
        match c {
            Component::CurDir => {}
            Component::Normal(part) => rel.push(part),
            Component::ParentDir => {
                if !rel.pop() {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "path attempts to navigate above the workspace root",
                    ));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "absolute/rooted components are not allowed",
                ));
            }
        }
    }

    // Join under root, resolve existing ancestor to handle symlinks, then re-verify containment.
    let candidate = root.join(&rel);
    let real = soft_canonicalize(&candidate)?;
    if !real.starts_with(&root) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "path resolves outside the workspace after following symlinks",
        ));
    }

    // Return the relative path suffix under the workspace root.
    let suffix = real
        .strip_prefix(&root)
        .unwrap_or(Path::new(""))
        .to_path_buf();
    if suffix.as_os_str().is_empty() {
        Ok(PathBuf::from("."))
    } else {
        Ok(suffix)
    }
}

/// Canonicalize the deepest existing ancestor of `p`, then append the missing tail.
/// This follows symlinks in the existing prefix but does not require the leaf to exist.
pub fn soft_canonicalize<P: AsRef<Path>>(p: P) -> io::Result<PathBuf> {
    let mut probe = p.as_ref();

    // Peel off non-existent tail components.
    let mut tail = Vec::new();
    while fs::symlink_metadata(probe).is_err() {
        match probe.parent() {
            Some(parent) => {
                if let Some(name) = probe.file_name() {
                    tail.push(name.to_os_string());
                }
                probe = parent;
            }
            None => break,
        }
    }

    // Canonicalize the existing prefix (if any), then append the tail back.
    let mut base = if fs::symlink_metadata(probe).is_ok() {
        probe.canonicalize()?
    } else {
        PathBuf::new()
    };
    for seg in tail.into_iter().rev() {
        base.push(seg);
    }
    Ok(base)
}

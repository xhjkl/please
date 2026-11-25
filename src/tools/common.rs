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

/// Resolve a user-supplied path to a relative path under the current directory, the
/// workspace. Symlinks in its existing prefix are followed and the leaf need not exist; a path
/// that leaves the workspace, whether absolute, by `..`, or by a symlink, fails with
/// `PermissionDenied`.
pub fn resolve_path_within_cwd(path: &str) -> io::Result<PathBuf> {
    let root = env::current_dir()?;
    let root = root.canonicalize()?;
    let input = Path::new(path);
    if path.is_empty() || input == Path::new(".") {
        return Ok(PathBuf::from("."));
    }
    if input.is_absolute() {
        let outside = "absolute paths must resolve under the workspace";
        return relative_within(&root, input, outside);
    }

    // Collapsing `..` lexically, so a relative path cannot climb above the root.
    let mut rel = PathBuf::new();
    for component in input.components() {
        match component {
            Component::ParentDir => {
                if !rel.pop() {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "path attempts to navigate above the workspace root",
                    ));
                }
            }
            Component::Normal(part) => rel.push(part),
            // A relative Unix path has neither a root nor a prefix.
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    let outside = "path resolves outside the workspace after following symlinks";
    relative_within(&root, &root.join(rel), outside)
}

/// Express `path` relative to `root` once its existing prefix is canonical, failing with
/// `outside` when it resolves elsewhere.
fn relative_within(root: &Path, path: &Path, outside: &'static str) -> io::Result<PathBuf> {
    let real = soft_canonicalize(path)?;
    let Ok(rel) = real.strip_prefix(root) else {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, outside));
    };
    if rel.as_os_str().is_empty() {
        return Ok(PathBuf::from("."));
    }
    Ok(rel.to_path_buf())
}

/// Canonicalize the deepest existing ancestor of `path`, then append the missing tail.
/// This follows symlinks in the existing prefix but does not require the leaf to exist.
fn soft_canonicalize(path: &Path) -> io::Result<PathBuf> {
    let mut probe = path;
    let mut tail = Vec::new();
    loop {
        let found = fs::symlink_metadata(probe);
        if found.is_ok() {
            let mut real = probe.canonicalize()?;
            real.extend(tail.iter().rev());
            return Ok(real);
        }
        let Some(parent) = probe.parent() else {
            return Ok(tail.iter().rev().collect());
        };
        if let Some(name) = probe.file_name() {
            tail.push(name);
        }
        probe = parent;
    }
}

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

/// Resolve `path` as the file system would, following symlinks wherever its components exist.
/// Past one that does not, nothing can be a symlink, so the rest resolves lexically; a `..`
/// there steps back into what exists, where resolution follows symlinks again.
fn soft_canonicalize(path: &Path) -> io::Result<PathBuf> {
    let mut real = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => {
                let next = real.join(name);
                let found = fs::symlink_metadata(&next);
                real = match found {
                    Err(_) => next,
                    Ok(_) => next.canonicalize()?,
                };
            }
            Component::ParentDir => {
                real.pop();
            }
            Component::RootDir | Component::Prefix(_) => real.push(component),
            Component::CurDir => {}
        }
    }
    Ok(real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    #[test]
    fn a_parent_past_a_missing_directory_steps_back_for_real() -> io::Result<()> {
        let (workspace, elsewhere) = (TempDir::new()?, TempDir::new()?);
        let root = workspace.0.canonicalize()?;
        let outside = elsewhere.0.canonicalize()?;
        fs::create_dir(root.join("a"))?;
        std::os::unix::fs::symlink(&outside, root.join("out"))?;

        let resolved = soft_canonicalize(&root.join("a/new/leaf"))?;
        assert_eq!(resolved, root.join("a/new/leaf"));
        let resolved = soft_canonicalize(&root.join("missing/../a/new.txt"))?;
        assert_eq!(resolved, root.join("a/new.txt"));
        // Stepping back into what exists follows its symlinks again.
        let resolved = soft_canonicalize(&root.join("missing/../out/file"))?;
        assert_eq!(resolved, outside.join("file"));

        let climbing = root.join("missing/../../etc/passwd");
        let escaped = relative_within(&root, &climbing, "outside").unwrap_err();
        assert_eq!(escaped.kind(), io::ErrorKind::PermissionDenied);
        let linked = relative_within(&root, &root.join("missing/../out/file"), "outside");
        assert_eq!(linked.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        Ok(())
    }
}

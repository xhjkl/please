use std::fs;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::{env, io};

#[derive(Debug, Clone)]
pub enum ParamType {
    String,
    #[allow(dead_code)]
    Choice(&'static [&'static str]),
    #[allow(dead_code)]
    Number,
    #[allow(dead_code)]
    Boolean,
}

#[derive(Clone)]
pub struct Param {
    pub name: &'static str,
    pub desc: &'static str,
    pub param_type: ParamType,
    pub required: bool,
}

/// Anything that can be called with a `serde_json::Value` payload.
pub type AsyncFn = Box<
    dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = serde_json::Value> + Send>>
        + Send
        + Sync,
>;

/// Adapt a typed async handler to an LLM/tool-call friendly `Fn(Value) -> Future<Value>`.
/// That keeps strongly-typed ergonomics at the edges; the closure is `Arc`-cloned for reuse.
/// Use this when registering typed tools behind a single uniform entrypoint expected by function-calling runtimes.
///
/// ```rust,ignore
/// #[derive(serde::Deserialize)]
/// struct Hello { name: String }
///
/// async fn hello(args: Hello) -> serde_json::Value {
///     serde_json::json!({ "hi": args.name })
/// }
///
/// // Expose a uniform tool-call shape for the LLM runtime
/// let wrapped = with_args(hello);
///
/// // Invoke with raw JSON like a function-calling payload
/// let out = wrapped(serde_json::json!({ "name": "Ada" })).await;
/// assert_eq!(out, serde_json::json!({ "hi": "Ada" }));
///
/// // Invalid inputs yield a normalized error object
/// let err = wrapped(serde_json::json!({ "name": 123 })).await;
/// assert!(err.get("error").is_some());
/// ```
pub fn with_args<Args, Fut, F>(f: F) -> AsyncFn
where
    Args: serde::de::DeserializeOwned + Send + 'static,
    F: Fn(Args) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = serde_json::Value> + Send + 'static,
{
    let f = Arc::new(f);
    Box::new(move |args: serde_json::Value| {
        let args = serde_json::from_value::<Args>(args).map_err(|e| e.to_string());
        let args = match args {
            Ok(args) => args,
            Err(error) => return Box::pin(async move { serde_json::json!({ "error": error }) }),
        };
        let f = Arc::clone(&f);
        Box::pin(async move { (f)(args).await })
    })
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

//! Minimal VRAM-aware context picker: try [native → 64k → 32k → 8k] and stop at first that fits.
//! Assumes F16 KV (2B per K & V).
use gg::model::LlamaModel;
use std::num::NonZeroU32;

/// Fraction of reported free VRAM we are willing to use for KV cache.
const GREED_FACTOR: f64 = 0.6;

/// Pick from [native, 64k, 32k, 8k] whichever should fit into the currently free video memory.
/// Values chosen empirically.
pub fn pick_n_ctx_by_vram(model: &LlamaModel, vram_free_bytes: u64) -> NonZeroU32 {
    const GB: u64 = 1024 * 1024 * 1024;

    let Ok(size_label) = model.meta_val_str("general.size_label") else {
        tracing::warn!("model: no size label found");
        return NonZeroU32::new(8_192).unwrap();
    };

    let native_ctx = model.n_ctx_train().max(1);
    let budget_bytes = (GREED_FACTOR * (vram_free_bytes as f64)) as u64;

    let model_size: usize = size_label
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(0);

    // https://github.com/ggml-org/llama.cpp/discussions/15396 § Minimum requirements
    let choices: &[(u64, u32)] = match model_size {
        // Given the model size, how much memory do we need for a context this large:
        120 => &[
            // --
            (96 * GB, native_ctx),
            (48 * GB, 65_536),
            (24 * GB, 32_768),
        ],
        20 => &[
            // --
            (24 * GB, native_ctx),
            (12 * GB, 65_536),
            (6 * GB, 32_768),
        ],
        _ => &[],
    };

    for &(threshold, ctx) in choices {
        if budget_bytes >= threshold {
            return NonZeroU32::new(ctx.min(native_ctx)).unwrap();
        }
    }

    tracing::warn!(
        "model: no context size found for budget {budget_bytes} bytes and model size {model_size}"
    );

    NonZeroU32::new(8_192).unwrap()
}

/// Returns free VRAM bytes if known (best-effort).
pub fn vram_free_bytes() -> Option<u64> {
    #[cfg(not(target_os = "macos"))]
    if let Some(v) = nvidia_free_bytes() {
        return Some(v);
    }

    #[cfg(target_os = "linux")]
    if let Some(v) = amd_free_bytes_sysfs() {
        return Some(v);
    }

    #[cfg(target_os = "macos")]
    if let Some(v) = metal_free_bytes() {
        return Some(v);
    }

    None
}

#[cfg(not(target_os = "macos"))]
fn nvidia_free_bytes() -> Option<u64> {
    use std::process::Command;

    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !out.status.success() {
        return None;
    }

    let mut best_mb: u64 = 0;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let mb = line.trim().parse::<u64>().ok()?;
        best_mb = best_mb.max(mb);
    }
    (best_mb > 0).then_some(best_mb * 1024 * 1024)
}

#[cfg(target_os = "linux")]
fn amd_free_bytes_sysfs() -> Option<u64> {
    use std::{fs, path::Path};

    fn read_u64_file<P: AsRef<Path>>(p: P) -> Option<u64> {
        let s = fs::read_to_string(p).ok()?;
        s.trim().parse::<u64>().ok()
    }

    fn read_first(device: &Path, names: &[&str]) -> Option<u64> {
        for n in names {
            if let Some(v) = read_u64_file(device.join(n)) {
                return Some(v);
            }
        }
        None
    }

    let mut best_free: u64 = 0;

    // Iterate DRM cards: /sys/class/drm/card*/device
    let entries = fs::read_dir("/sys/class/drm").ok()?;
    for e in entries.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains('-') {
            continue; // skip connectors like card0-DP-1
        }

        let dev = e.path().join("device");
        if !dev.exists() {
            continue;
        }

        // Ensure it's AMD (PCI vendor 0x1002)
        if let Some(vendor) = read_u64_file(dev.join("vendor").as_path()) {
            if vendor != 0x1002 && vendor != 1002 {
                continue;
            }
        } else {
            // If vendor missing, keep going only if mem_info files exist
        }

        // Prefer "visible vram" (APUs) when present, otherwise use total VRAM.
        let vis_total = read_first(
            &dev,
            &["mem_info_visible_vram_total", "mem_info_vis_vram_total"],
        );
        let vis_used = read_first(
            &dev,
            &["mem_info_visible_vram_used", "mem_info_vis_vram_used"],
        );

        let (total, used) = if let (Some(t), Some(u)) = (vis_total, vis_used) {
            (t, u)
        } else {
            let t = read_u64_file(dev.join("mem_info_vram_total"))?;
            let u = read_u64_file(dev.join("mem_info_vram_used"))?;
            (t, u)
        };

        let free = total.saturating_sub(used);
        best_free = best_free.max(free);
    }

    (best_free > 0).then_some(best_free)
}

#[cfg(target_os = "macos")]
fn metal_free_bytes() -> Option<u64> {
    let dev = metal::Device::system_default()?;
    let recommended = dev.recommended_max_working_set_size();
    let current = dev.current_allocated_size();
    Some(recommended.saturating_sub(current))
}

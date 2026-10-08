//! Hardware-memory heuristics for model and context selection.
use gg::model::LlamaModel;
use std::num::NonZeroU32;

/// Fraction of reported available memory we are willing to use for KV cache.
const GREED_FACTOR: f64 = 0.6;

/// Pick from [native, 64k, 32k, 8k] whichever should fit into the available memory.
/// Values chosen empirically.
pub fn pick_n_ctx(model: &LlamaModel, memory_bytes: u64) -> NonZeroU32 {
    const GB: u64 = 1024 * 1024 * 1024;

    let Ok(size_label) = model.meta_val_str("general.size_label") else {
        tracing::warn!("model: no size label found");
        return NonZeroU32::new(8_192).unwrap();
    };

    let native_ctx = model.n_ctx_train().max(1);
    let budget_bytes = (GREED_FACTOR * (memory_bytes as f64)) as u64;

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

#[cfg(target_os = "linux")]
pub fn available_memory_bytes() -> eyre::Result<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let memory_bytes = meminfo.lines().find_map(|line| {
        let value = line.strip_prefix("MemAvailable:")?;
        let kibibytes = value.split_whitespace().next()?.parse::<u64>().ok()?;
        kibibytes.checked_mul(1024)
    });
    memory_bytes.ok_or_else(|| eyre::eyre!("MemAvailable is missing from /proc/meminfo"))
}

#[cfg(target_os = "macos")]
pub fn available_memory_bytes() -> eyre::Result<u64> {
    let device = metal::Device::system_default();
    let device = device.ok_or_else(|| eyre::eyre!("no system Metal device found"))?;
    let recommended = device.recommended_max_working_set_size();
    let allocated = device.current_allocated_size();
    Ok(recommended.saturating_sub(allocated))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn available_memory_bytes() -> eyre::Result<u64> {
    Err(eyre::eyre!("model selection supports Linux and macOS"))
}

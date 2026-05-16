use eyre::{Result, eyre};
use inference_engine::backend_metal::MetalEngine;
use inference_engine::harmony_adapter::{HarmonyAdapter, Message, Role};
use inference_engine::{
    EngineRequest, GenerationEvent, GenerationLimits, PromptPlan, SamplingConfig, StopReason,
};
#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let args = Args::parse()?;
    #[cfg(feature = "profile")]
    let started = Instant::now();
    let harmony = HarmonyAdapter::gpt_oss()?;
    #[cfg(feature = "profile")]
    let harmony_load = started.elapsed();
    #[cfg(feature = "profile")]
    let started = Instant::now();
    let messages = [Message::from((Role::User, args.prompt.clone()))];
    let tokens = harmony.render_completion_tokens(&messages)?;
    #[cfg(feature = "profile")]
    let tokenize = started.elapsed();
    let context_capacity = tokens
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| eyre!("context capacity overflow"))?;
    if args.pinned_prefix_len > tokens.len() {
        return Err(eyre!(
            "--pinned-prefix {} exceeds rendered prompt length {}",
            args.pinned_prefix_len,
            tokens.len()
        ));
    }
    #[cfg(feature = "profile")]
    let started = Instant::now();
    let engine = MetalEngine::load_canonical_with_layers(args.layers)?;
    #[cfg(feature = "profile")]
    let load = started.elapsed();
    let request = EngineRequest {
        prompt: PromptPlan {
            tokens,
            pinned_prefix_len: args.pinned_prefix_len,
            context_capacity,
            notices: Vec::new(),
        },
        sampling: args.sampling,
        limits: GenerationLimits {
            max_new_tokens: args.max_new_tokens,
            max_output_bytes: args.max_output_bytes,
        },
        fixture: None,
    };

    let mut events = Vec::new();
    #[cfg(feature = "profile")]
    let mut profile = None;
    for iteration in 0..args.repeat {
        let is_final = iteration + 1 == args.repeat;
        #[cfg(feature = "profile")]
        if is_final {
            let (next_events, next_profile) = engine.generate_profiled(request.clone())?;
            events = next_events;
            profile = Some(next_profile);
            continue;
        }

        let _ = is_final;
        {
            events = engine.generate(request.clone())?;
        }
    }
    let mut text = String::new();
    let mut stop_reason = None;
    for event in events {
        match event {
            GenerationEvent::Text(chunk) => text.push_str(&chunk),
            GenerationEvent::Stop(reason) => stop_reason = Some(reason),
            GenerationEvent::Token(_) | GenerationEvent::Notice(_) => {}
        }
    }

    println!("inference-engine generate:");
    println!("- layers: {}", args.layers);
    println!("- max new tokens: {}", args.max_new_tokens);
    println!("- pinned prefix tokens: {}", args.pinned_prefix_len);
    println!("- repeat: {}", args.repeat);
    println!(
        "- stop_reason: {:?}",
        stop_reason.unwrap_or(StopReason::Cancelled)
    );
    println!("\n{text}");
    #[cfg(feature = "profile")]
    {
        if let Some(profile) = profile {
            println!("\nsetup profile:");
            println!("- harmony load: {}", format_duration(harmony_load));
            println!("- tokenize/render: {}", format_duration(tokenize));
            println!("- engine load: {}", format_duration(load));
            print!("{}", profile.render_for_cli());
        }
    }
    Ok(())
}

struct Args {
    prompt: String,
    layers: usize,
    max_new_tokens: usize,
    max_output_bytes: usize,
    sampling: SamplingConfig,
    pinned_prefix_len: usize,
    repeat: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut prompt = "Summarize the staged diff.".to_string();
        let mut layers = 24usize;
        let mut max_new_tokens = 8usize;
        let mut max_output_bytes = 1024 * 1024usize;
        let mut sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let mut pinned_prefix_len = 0usize;
        let mut repeat = 1usize;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--prompt" => {
                    prompt = args
                        .next()
                        .ok_or_else(|| eyre!("--prompt requires a value"))?;
                }
                "--layers" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--layers requires a value"))?;
                    layers = value.parse()?;
                }
                "--max-new" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--max-new requires a value"))?;
                    max_new_tokens = value.parse()?;
                }
                "--max-output-bytes" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--max-output-bytes requires a value"))?;
                    max_output_bytes = value.parse()?;
                }
                "--seed" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--seed requires a value"))?;
                    sampling.seed = value.parse()?;
                }
                "--temperature" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--temperature requires a value"))?;
                    sampling.temperature = value.parse()?;
                }
                "--top-k" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--top-k requires a value"))?;
                    sampling.top_k = value.parse()?;
                }
                "--top-p" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--top-p requires a value"))?;
                    sampling.top_p = value.parse()?;
                }
                "--pinned-prefix" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--pinned-prefix requires a value"))?;
                    pinned_prefix_len = value.parse()?;
                }
                "--repeat" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--repeat requires a value"))?;
                    repeat = value.parse()?;
                }
                _ => return Err(eyre!("unknown argument {arg}")),
            }
        }

        if max_new_tokens == 0 {
            return Err(eyre!("--max-new must be greater than zero"));
        }
        if max_output_bytes == 0 {
            return Err(eyre!("--max-output-bytes must be greater than zero"));
        }
        if sampling.temperature < 0.0 {
            return Err(eyre!("--temperature must be non-negative"));
        }
        if !(0.0..=1.0).contains(&sampling.top_p) {
            return Err(eyre!("--top-p must be in 0..=1"));
        }
        if repeat == 0 {
            return Err(eyre!("--repeat must be greater than zero"));
        }

        Ok(Self {
            prompt,
            layers,
            max_new_tokens,
            max_output_bytes,
            sampling,
            pinned_prefix_len,
            repeat,
        })
    }
}

#[cfg(feature = "profile")]
fn format_duration(duration: Duration) -> String {
    let ns = duration.as_nanos();
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.1}ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.1}us", ns as f64 / 1_000.0)
    } else {
        format!("{ns}ns")
    }
}

use eyre::{Result, eyre};
use inference_engine::backend_metal::MetalEngine;
use inference_engine::harmony_adapter::{HarmonyAdapter, Message, Role};
use inference_engine::{
    EngineRequest, GenerationEvent, GenerationLimits, PromptPlan, SamplingConfig, StopReason,
};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let harmony = HarmonyAdapter::gpt_oss()?;
    let messages = [Message::from((Role::User, args.prompt.clone()))];
    let tokens = harmony.render_completion_tokens(&messages)?;
    let context_capacity = tokens
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| eyre!("context capacity overflow"))?;
    let engine = MetalEngine::load_canonical_with_layers(args.layers)?;
    let request = EngineRequest {
        prompt: PromptPlan {
            tokens,
            pinned_prefix_len: 0,
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

    let events = engine.generate(request)?;
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
    println!(
        "- stop_reason: {:?}",
        stop_reason.unwrap_or(StopReason::Cancelled)
    );
    println!("\n{text}");
    Ok(())
}

struct Args {
    prompt: String,
    layers: usize,
    max_new_tokens: usize,
    max_output_bytes: usize,
    sampling: SamplingConfig,
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

        Ok(Self {
            prompt,
            layers,
            max_new_tokens,
            max_output_bytes,
            sampling,
        })
    }
}

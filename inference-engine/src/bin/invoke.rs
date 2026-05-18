use eyre::{Result, eyre};
#[cfg(feature = "profile")]
use inference_engine::backend_metal::MetalProfile;
use inference_engine::{
    Generated, GenerationStream, HarmonyAdapter, Message, MetalModel, MetalTimings,
};
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let timings_enabled = engine_timings_enabled();
    let started = Instant::now();
    let harmony = HarmonyAdapter::gpt_oss()?;
    let harmony_load = started.elapsed();
    if let Some(contexts) = &args.bench_contexts {
        return run_context_bench(&args, &harmony, contexts, harmony_load);
    }

    let started = Instant::now();
    let tokens = prompt_tokens_for_args(&args, &harmony)?;
    let tokenize = started.elapsed();
    let context_capacity = tokens
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| eyre!("context capacity overflow"))?;

    let mut generation = None;
    #[cfg(feature = "profile")]
    let mut profile: Option<MetalProfile> = None;
    let mut timings: Option<MetalTimings> = None;
    let started = Instant::now();
    let load;
    let model = MetalModel::load_canonical_with_layers(args.layers)?;
    load = started.elapsed();
    let episode = model.episode(context_capacity)?;
    for iteration in 0..args.repeat {
        let token_count = episode.token_count();
        episode.splice_tokens(0..token_count, &tokens)?;
        let is_final = iteration + 1 == args.repeat;
        #[cfg(feature = "profile")]
        if is_final {
            model.reset_profile();
            let stream = episode.generate(args.max_new_tokens)?;
            generation = Some(render_stream(&harmony, stream)?);
            profile = Some(model.profile_report());
            continue;
        }

        if is_final && timings_enabled {
            let stream = episode.generate_timed(args.max_new_tokens)?;
            let (stream, next_timings) = stream.into_parts();
            generation = Some(render_stream(&harmony, stream)?);
            timings = Some(
                next_timings
                    .recv()
                    .map_err(|_| eyre!("generation ended without timings"))?,
            );
            continue;
        }
        let stream = episode.generate(args.max_new_tokens)?;
        generation = Some(render_stream(&harmony, stream)?);
    }
    let Some(generation) = generation else {
        return Err(eyre!("generation did not run"));
    };

    println!("inference-engine invoke:");
    println!("- source: gguf");
    println!("- layers: {}", args.layers);
    println!("- max new tokens: {}", args.max_new_tokens);
    println!("- repeat: {}", args.repeat);
    println!("- finish: {:?}", generation.finish);
    println!("- tokens: {:?}", generation.tokens);
    println!("\n{}", generation.text);
    if timings_enabled {
        println!("\nsetup timings:");
        println!("- harmony load: {}", format_duration(harmony_load));
        println!("- tokenize/render: {}", format_duration(tokenize));
        println!("- engine load: {}", format_duration(load));
        #[cfg(feature = "profile")]
        println!(
            "- note: `--features profile` is the heavy profiler; light timings are reported by non-profile builds"
        );
        if let Some(timings) = timings {
            print!("{timings}");
        }
    }
    #[cfg(feature = "profile")]
    {
        if let Some(profile) = profile {
            println!("\nsetup profile:");
            println!("- harmony load: {}", format_duration(harmony_load));
            println!("- tokenize/render: {}", format_duration(tokenize));
            println!("- engine load: {}", format_duration(load));
            print!("{profile}");
        }
    }
    Ok(())
}

struct Args {
    prompt: String,
    target_prompt_tokens: Option<usize>,
    bench_contexts: Option<Vec<usize>>,
    samples: usize,
    layers: usize,
    max_new_tokens: usize,
    repeat: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut prompt = "Summarize the staged diff.".to_string();
        let mut target_prompt_tokens = None;
        let mut bench_contexts = None;
        let mut samples = 3usize;
        let mut layers = 24usize;
        let mut max_new_tokens = 8usize;
        let mut repeat = 1usize;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--prompt" => {
                    prompt = args
                        .next()
                        .ok_or_else(|| eyre!("--prompt requires a value"))?;
                }
                "--target-prompt-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--target-prompt-tokens requires a value"))?;
                    target_prompt_tokens = Some(value.parse()?);
                }
                "--bench-contexts" => {
                    let value = args.next().ok_or_else(|| {
                        eyre!("--bench-contexts requires a comma-separated value")
                    })?;
                    bench_contexts = Some(parse_contexts(&value)?);
                }
                "--samples" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--samples requires a value"))?;
                    samples = value.parse()?;
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
        if repeat == 0 {
            return Err(eyre!("--repeat must be greater than zero"));
        }
        if samples == 0 {
            return Err(eyre!("--samples must be greater than zero"));
        }

        Ok(Self {
            prompt,
            target_prompt_tokens,
            bench_contexts,
            samples,
            layers,
            max_new_tokens,
            repeat,
        })
    }
}

fn parse_contexts(value: &str) -> Result<Vec<usize>> {
    let contexts = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<_>, _>>()?;
    if contexts.is_empty() {
        return Err(eyre!(
            "--bench-contexts did not contain any context lengths"
        ));
    }
    Ok(contexts)
}

fn prompt_tokens_for_args(args: &Args, harmony: &HarmonyAdapter) -> Result<Vec<u32>> {
    let Some(target) = args.target_prompt_tokens else {
        let messages = [Message::user(args.prompt.clone())];
        return harmony.render_completion_tokens(&messages);
    };
    prompt_tokens_near(harmony, target, &args.prompt)
}

fn prompt_tokens_near(
    harmony: &HarmonyAdapter,
    target_prompt_tokens: usize,
    seed: &str,
) -> Result<Vec<u32>> {
    let target_prompt_tokens = target_prompt_tokens.max(1);
    let mut prompt = seed.to_string();
    let mut tokens = harmony.render_completion_tokens(&[Message::user(prompt.clone())])?;
    let filler = "\n\nContext calibration paragraph: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. The exact words are unimportant; the benchmark only needs stable token pressure.";
    while tokens.len() + 64 < target_prompt_tokens {
        prompt.push_str(filler);
        tokens = harmony.render_completion_tokens(&[Message::user(prompt.clone())])?;
    }
    while tokens.len() < target_prompt_tokens {
        prompt.push_str(" x");
        tokens = harmony.render_completion_tokens(&[Message::user(prompt.clone())])?;
    }
    Ok(tokens)
}

fn run_context_bench(
    args: &Args,
    harmony: &HarmonyAdapter,
    contexts: &[usize],
    harmony_load: Duration,
) -> Result<()> {
    let started = Instant::now();
    let model = MetalModel::load_canonical_with_layers(args.layers)?;
    let load = started.elapsed();

    println!("inference-engine context bench:");
    println!("- source: gguf");
    println!("- layers: {}", args.layers);
    println!("- max new tokens: {}", args.max_new_tokens);
    println!("- samples/context: {}", args.samples);
    println!("- harmony load: {}", format_duration(harmony_load));
    println!("- engine load: {}", format_duration(load));

    for target in contexts {
        let max_prompt_tokens = 4096usize
            .checked_sub(args.max_new_tokens)
            .ok_or_else(|| eyre!("--max-new exceeds maximum resident context"))?;
        let prompt_target = (*target).min(max_prompt_tokens);
        let started = Instant::now();
        let tokens = prompt_tokens_near(harmony, prompt_target, &args.prompt)?;
        let tokenize = started.elapsed();
        let context_capacity = tokens
            .len()
            .checked_add(args.max_new_tokens)
            .ok_or_else(|| eyre!("context capacity overflow"))?;
        let episode = model.episode(context_capacity)?;

        println!("\ncontext sample:");
        println!("- requested prompt tokens: {target}");
        println!("- actual prompt tokens: {}", tokens.len());
        println!("- context capacity: {context_capacity}");
        println!("- tokenize/render: {}", format_duration(tokenize));

        for sample in 0..args.samples {
            let token_count = episode.token_count();
            episode.splice_tokens(0..token_count, &tokens)?;
            let stream = episode.generate_timed(args.max_new_tokens)?;
            let (stream, timings) = stream.into_parts();
            let generation = render_stream(harmony, stream)?;
            let timings = timings
                .recv()
                .map_err(|_| eyre!("generation ended without timings"))?;

            println!("\nbench sample {}:", sample + 1);
            println!("- finish: {:?}", generation.finish);
            println!("- tokens: {:?}", generation.tokens);
            print!("{timings}");
        }
    }

    Ok(())
}

struct RenderedGeneration {
    tokens: Vec<u32>,
    text: String,
    finish: RenderFinish,
}

#[derive(Debug)]
enum RenderFinish {
    Stop,
    LimitReached,
}

fn render_stream(harmony: &HarmonyAdapter, stream: GenerationStream) -> Result<RenderedGeneration> {
    let mut tokens = Vec::new();
    let mut finish = None;
    for event in stream {
        match event {
            Generated::Token(token) => tokens.push(token),
            Generated::Stop => {
                finish = Some(RenderFinish::Stop);
                break;
            }
            Generated::LimitReached => {
                finish = Some(RenderFinish::LimitReached);
                break;
            }
            Generated::Error(message) => return Err(eyre!("generation failed: {message}")),
            Generated::ExpertMiss { layer, expert } => {
                return Err(eyre!(
                    "generation needs expert slab layer={layer} expert={expert}, but this runner cannot service expert misses yet"
                ));
            }
        }
    }
    let Some(finish) = finish else {
        return Err(eyre!("generation stream closed without a finish event"));
    };
    let text = harmony.decode_utf8(&tokens)?;
    Ok(RenderedGeneration {
        tokens,
        text,
        finish,
    })
}

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

fn engine_timings_enabled() -> bool {
    let Some(value) = std::env::var("PLEASE_ENGINE_TIMINGS").ok() else {
        return false;
    };
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

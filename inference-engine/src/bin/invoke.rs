use eyre::{Result, eyre};
#[cfg(feature = "profile")]
use inference_engine::backend_metal::MetalProfile;
use inference_engine::backend_metal::run_attention_probe;
use inference_engine::{
    Generated, GenerationStream, HarmonyAdapter, Message, MetalModel, MetalTimings,
};
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let args = Args::parse()?;
    if args.attention_probe {
        let report = run_attention_probe()?;
        print!("{report}");
        return Ok(());
    }
    if args.stage_envelope && !cfg!(feature = "profile") {
        return Err(eyre!(
            "--stage-envelope requires `cargo run --features profile`"
        ));
    }

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
    stage_envelope: bool,
    attention_probe: bool,
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
        let mut stage_envelope = false;
        let mut attention_probe = false;
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
                "--stage-envelope" => {
                    stage_envelope = true;
                }
                "--attention-probe" => {
                    attention_probe = true;
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
            stage_envelope,
            attention_probe,
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

        let mut summaries = Vec::with_capacity(args.samples);
        for sample in 0..args.samples {
            let token_count = episode.token_count();
            episode.splice_tokens(0..token_count, &tokens)?;
            #[cfg(feature = "profile")]
            if args.stage_envelope {
                model.reset_profile();
            }
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
            summaries.push(BenchSample::from_timings(&timings));
            print_stage_envelope(&model, args, tokens.len(), sample + 1);
        }
        print_context_summary(tokens.len(), &summaries);
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct BenchSample {
    hot_wall_ns: u128,
    hot_gpu_ns: u128,
    hot_gap_ns: u128,
    command_buffers_per_token: f64,
    encoders_per_token: f64,
    dispatches_per_token: f64,
    readbacks_per_token: f64,
}

impl BenchSample {
    fn from_timings(timings: &MetalTimings) -> Self {
        let hot_token_count = timings.hot_token_count.max(1);
        let hot_wall_ns = timings.hot_token_wall.as_nanos() / hot_token_count as u128;
        let hot_gpu_ns = timings.hot_token_gpu_ns / hot_token_count as u128;
        Self {
            hot_wall_ns,
            hot_gpu_ns,
            hot_gap_ns: hot_wall_ns.saturating_sub(hot_gpu_ns),
            command_buffers_per_token: timings.hot_command_buffers as f64 / hot_token_count as f64,
            encoders_per_token: timings.hot_compute_encoders as f64 / hot_token_count as f64,
            dispatches_per_token: timings.hot_dispatches as f64 / hot_token_count as f64,
            readbacks_per_token: timings.hot_readback_calls as f64 / hot_token_count as f64,
        }
    }
}

fn print_context_summary(prompt_tokens: usize, samples: &[BenchSample]) {
    if samples.is_empty() {
        return;
    }

    let wall_values = sorted_values(samples.iter().map(|sample| sample.hot_wall_ns));
    let gpu_values = sorted_values(samples.iter().map(|sample| sample.hot_gpu_ns));
    let gap_values = sorted_values(samples.iter().map(|sample| sample.hot_gap_ns));
    let representative = &samples[samples.len() - 1];

    println!("\ncontext summary:");
    println!("- prompt tokens: {prompt_tokens}");
    println!("- samples: {}", samples.len());
    println!(
        "- median hot wall: {}",
        format_duration_ns(median(&wall_values))
    );
    println!(
        "- median hot GPU: {}",
        format_duration_ns(median(&gpu_values))
    );
    println!("- p95 hot wall: {}", format_duration_ns(p95(&wall_values)));
    println!("- p95 hot GPU: {}", format_duration_ns(p95(&gpu_values)));
    println!("- min wall/GPU gap: {}", format_duration_ns(gap_values[0]));
    println!(
        "- median wall/GPU gap: {}",
        format_duration_ns(median(&gap_values))
    );
    println!(
        "- command buffers/token: {:.1}",
        representative.command_buffers_per_token
    );
    println!("- encoders/token: {:.1}", representative.encoders_per_token);
    println!(
        "- dispatches/token: {:.1}",
        representative.dispatches_per_token
    );
    println!(
        "- readbacks/token: {:.1}",
        representative.readbacks_per_token
    );
}

fn sorted_values(values: impl Iterator<Item = u128>) -> Vec<u128> {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values
}

fn median(values: &[u128]) -> u128 {
    values[values.len() / 2]
}

fn p95(values: &[u128]) -> u128 {
    let index = (values.len() * 95).div_ceil(100).saturating_sub(1);
    values[index]
}

fn format_duration_ns(ns: u128) -> String {
    format_duration(Duration::from_nanos(ns.min(u64::MAX as u128) as u64))
}

fn print_stage_envelope(model: &MetalModel, args: &Args, prompt_tokens: usize, sample: usize) {
    if !args.stage_envelope {
        return;
    }

    #[cfg(not(feature = "profile"))]
    let _ = (model, prompt_tokens, sample);

    #[cfg(feature = "profile")]
    {
        let profile = model.profile_report();
        let rows = profile.stage_envelope_rows();
        println!("\nstage envelope:");
        println!("- prompt tokens: {prompt_tokens}");
        println!("- sample: {sample}");
        if rows.is_empty() {
            println!("- no stage rows; Metal counter samples may be unsupported");
            return;
        }
        println!("- source: profile counter samples; absolute times include observer effect");
        println!("stage              sampled   pct_of_sample");
        println!("-----------------  --------  -------------");
        for row in rows {
            println!(
                "{:<17}  {:>8}  {:>12.1}%",
                row.stage,
                format_duration_ns(row.average_ns),
                row.percent_of_token
            );
        }
    }
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

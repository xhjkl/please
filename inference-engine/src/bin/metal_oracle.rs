use eyre::{Result, eyre};
use inference_engine::backend_cpu;
use inference_engine::backend_metal;
use inference_engine::gptoss_spec::weights;
use inference_engine::harmony_adapter::{HarmonyAdapter, Message, Role};
use inference_engine::model_store;
use inference_engine::{GreedyDecodeProbeReport, PromptFixture, SamplingConfig};

const FULL_DEPTH_LAYERS: usize = 24;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let report = model_store::inspect_canonical_safetensors()?;
    let validation = weights::validate_gpt_oss_20b_source(&report);
    let harmony = HarmonyAdapter::gpt_oss()?;
    let fixture = prompt_fixture(&harmony, args.prefill_tokens)?;

    let mut out = String::new();
    out.push_str(&report.render_for_cli());
    out.push_str(&validation.render_for_cli());
    out.push_str(&fixture.render_for_cli());

    let prefill_tokens = fixture
        .prompt_tokens
        .iter()
        .copied()
        .take(fixture.prefill_token_count)
        .collect::<Vec<_>>();

    let Some(token) = fixture.prompt_tokens.first().copied() else {
        return Err(eyre!("metal oracle fixture produced no prompt tokens"));
    };
    let ctx = backend_metal::MetalOracleContext::with_lm_head(&report)?;

    if args.decode_only {
        let greedy_decode = backend_metal::probe_sample_decode(
            &ctx,
            &report,
            &harmony,
            &prefill_tokens,
            args.layers,
            args.max_new_tokens,
            args.sampling.clone(),
        )?;
        out.push_str(&greedy_decode.render_for_cli());
        if args.cpu_check {
            let cpu = backend_cpu::probe_greedy_decode(
                &report,
                &harmony,
                &prefill_tokens,
                args.cpu_check_layers,
                args.cpu_check_max_new,
            )?;
            let metal = backend_metal::probe_greedy_decode(
                &ctx,
                &report,
                &harmony,
                &prefill_tokens,
                args.cpu_check_layers,
                args.cpu_check_max_new,
            )?;
            out.push_str(&cpu.render_for_cli());
            out.push_str(&metal.render_for_cli());
            out.push_str(&render_greedy_decode_comparison(&cpu, &metal));
        }
        if args.kv_stress {
            let stress_tokens = repeated_tokens(&fixture.prompt_tokens, args.kv_stress_tokens)?;
            let window = backend_metal::probe_kv_cache_window_rollover_attention(
                &ctx,
                &report,
                &stress_tokens,
            )?;
            let dense = backend_metal::probe_kv_cache_dense_accumulation_attention(
                &ctx,
                &report,
                &stress_tokens,
            )?;
            out.push_str(&window.render_for_cli());
            out.push_str(&dense.render_for_cli());
        }
        println!("{out}");
        return Ok(());
    }

    let cpu = backend_cpu::probe_prompt_prefill_oracle(
        &report,
        &prefill_tokens,
        args.layers,
        &args.logit_tokens,
    )?;
    out.push_str("\ncpu oracle baseline:\n");
    out.push_str(&format!("- layers: {}\n", cpu.layers));
    out.push_str(&format!(
        "- embedding final first 8: {:?}\n",
        cpu.embedding_final_first8
    ));
    for checkpoint in &cpu.layer_checkpoints {
        out.push_str(&format!(
            "- layer {}: final_l2 {:.7}, final_mean {:.7}, final_first8 {:?}\n",
            checkpoint.layer, checkpoint.final_l2, checkpoint.final_mean, checkpoint.final_first8
        ));
    }
    out.push_str(&format!(
        "- final_norm first 8: {:?}\n",
        cpu.final_norm_first8
    ));
    out.push_str("- selected logits:\n");
    for logit in &cpu.selected_logits {
        out.push_str(&format!("  - token {}: {:.7}\n", logit.token, logit.logit));
    }

    let metal = backend_metal::probe_rms_norm_embedding(&ctx, &report, token)?;
    out.push_str(&metal.render_for_cli());
    let q_proj = backend_metal::probe_layer0_q_proj(&ctx, &report, token)?;
    out.push_str(&q_proj.render_for_cli());
    let k_proj = backend_metal::probe_layer0_k_proj(&ctx, &report, token)?;
    out.push_str(&k_proj.render_for_cli());
    let v_proj = backend_metal::probe_layer0_v_proj(&ctx, &report, token)?;
    out.push_str(&v_proj.render_for_cli());
    let q_rope = backend_metal::probe_layer0_q_rope(&ctx, &report, token, 0)?;
    out.push_str(&q_rope.render_for_cli());
    let k_rope = backend_metal::probe_layer0_k_rope(&ctx, &report, token, 0)?;
    out.push_str(&k_rope.render_for_cli());
    let attention = backend_metal::probe_layer0_single_token_attention(&ctx, &report, token)?;
    out.push_str(&attention.render_for_cli());
    let sequence_attention =
        backend_metal::probe_layer0_sequence_attention(&ctx, &report, &prefill_tokens)?;
    out.push_str(&sequence_attention.render_for_cli());
    let kv_decode =
        backend_metal::probe_layer0_kv_cache_decode_attention(&ctx, &report, &prefill_tokens)?;
    out.push_str(&kv_decode.render_for_cli());
    let prefill_output =
        backend_metal::probe_prefill_layers_output(&ctx, &report, &prefill_tokens, args.layers)?;
    out.push_str(&prefill_output.render_for_cli());
    let prefill_final_norm =
        backend_metal::probe_prefill_final_norm(&ctx, &report, &prefill_tokens, args.layers)?;
    out.push_str(&prefill_final_norm.render_for_cli());
    let prefill_logits = backend_metal::probe_prefill_selected_logits(
        &ctx,
        &report,
        &prefill_tokens,
        args.layers,
        &args.logit_tokens,
    )?;
    out.push_str(&prefill_logits.render_for_cli());
    if let Some(decode_token) = fixture.prompt_tokens.get(prefill_tokens.len()).copied() {
        let decode_final_norm = backend_metal::probe_decode_one_final_norm(
            &ctx,
            &report,
            &prefill_tokens,
            decode_token,
            args.layers,
        )?;
        out.push_str(&decode_final_norm.render_for_cli());
        let decode_logits = backend_metal::probe_decode_one_selected_logits(
            &ctx,
            &report,
            &prefill_tokens,
            decode_token,
            args.layers,
            &args.logit_tokens,
        )?;
        out.push_str(&decode_logits.render_for_cli());
        let decode_text = backend_metal::probe_decode_one_greedy_text(
            &ctx,
            &report,
            &harmony,
            &prefill_tokens,
            decode_token,
            args.layers,
        )?;
        out.push_str(&decode_text.render_for_cli());
        let decode_lm_head_topk = backend_metal::probe_decode_one_lm_head_topk(
            &ctx,
            &report,
            &harmony,
            &prefill_tokens,
            decode_token,
            args.layers,
            8,
        )?;
        out.push_str(&decode_lm_head_topk.render_for_cli());
    } else {
        out.push_str(
            "\nmetal decode-one oracle probe:\n- skipped: fixture has no token after prefill prefix\n",
        );
    }
    let greedy_decode = backend_metal::probe_sample_decode(
        &ctx,
        &report,
        &harmony,
        &prefill_tokens,
        args.layers,
        args.max_new_tokens,
        args.sampling.clone(),
    )?;
    out.push_str(&greedy_decode.render_for_cli());
    let o_proj = backend_metal::probe_layer0_o_proj(&ctx, &report, token)?;
    out.push_str(&o_proj.render_for_cli());
    let residual = backend_metal::probe_layer0_attention_residual(&ctx, &report, token)?;
    out.push_str(&residual.render_for_cli());
    let post_attention_norm =
        backend_metal::probe_layer0_post_attention_rms_norm(&ctx, &report, token)?;
    out.push_str(&post_attention_norm.render_for_cli());
    let router = backend_metal::probe_layer0_router(&ctx, &report, token)?;
    out.push_str(&router.render_for_cli());
    let router_top4 = backend_metal::probe_layer0_router_top4(&ctx, &report, token)?;
    out.push_str(&router_top4.render_for_cli());
    let gate_up = backend_metal::probe_layer0_top_expert_gate_up(&ctx, &report, token)?;
    out.push_str(&gate_up.render_for_cli());
    let swiglu = backend_metal::probe_layer0_top_expert_swiglu(&ctx, &report, token)?;
    out.push_str(&swiglu.render_for_cli());
    let down_proj = backend_metal::probe_layer0_top_expert_down_proj(&ctx, &report, token)?;
    out.push_str(&down_proj.render_for_cli());
    let moe = backend_metal::probe_layer0_moe_top4(&ctx, &report, token)?;
    out.push_str(&moe.render_for_cli());
    let layer0 = backend_metal::probe_layer0_output(&ctx, &report, token)?;
    out.push_str(&layer0.render_for_cli());
    let stack_final_norm =
        backend_metal::probe_single_token_final_norm(&ctx, &report, token, args.layers)?;
    out.push_str(&stack_final_norm.render_for_cli());
    let stack_logits = backend_metal::probe_single_token_selected_logits(
        &ctx,
        &report,
        token,
        args.layers,
        &args.logit_tokens,
    )?;
    out.push_str(&stack_logits.render_for_cli());

    println!("{out}");
    Ok(())
}

fn prompt_fixture(harmony: &HarmonyAdapter, prefill_token_count: usize) -> Result<PromptFixture> {
    let messages = [Message::from((Role::User, "Summarize the staged diff."))];
    let tokens = harmony.render_completion_tokens(&messages)?;
    let prompt = harmony.decode_utf8(&tokens)?;
    let prompt_token_prefix = tokens.iter().copied().take(16).collect();
    let prompt_token_suffix = tokens
        .iter()
        .copied()
        .skip(tokens.len().saturating_sub(16))
        .collect();

    let prefill_token_count = prefill_token_count.min(tokens.len());
    Ok(PromptFixture {
        fixture_name: Some("plain user prompt".to_string()),
        prompt_bytes: prompt.len(),
        prompt_token_count: tokens.len(),
        prompt_tokens: tokens,
        prompt_token_prefix,
        prompt_token_suffix,
        prefill_token_count,
    })
}

struct Args {
    layers: usize,
    prefill_tokens: usize,
    max_new_tokens: usize,
    decode_only: bool,
    cpu_check: bool,
    cpu_check_layers: usize,
    cpu_check_max_new: usize,
    kv_stress: bool,
    kv_stress_tokens: usize,
    sampling: SamplingConfig,
    logit_tokens: Vec<u32>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut layers = 1usize;
        let mut prefill_tokens = backend_cpu::PROMPT_PREFILL_TOKEN_LIMIT;
        let mut max_new_tokens = 2usize;
        let mut decode_only = false;
        let mut cpu_check = false;
        let mut cpu_check_layers = 1usize;
        let mut cpu_check_max_new = 2usize;
        let mut kv_stress = false;
        let mut kv_stress_tokens = 132usize;
        let mut sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let mut logit_tokens = parse_csv_u32("277,8526,387,263,278,289,10581,1808")?;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--full-depth" => {
                    layers = FULL_DEPTH_LAYERS;
                    decode_only = true;
                }
                "--decode-only" => decode_only = true,
                "--cpu-check" => cpu_check = true,
                "--cpu-check-layers" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--cpu-check-layers requires a value"))?;
                    cpu_check_layers = value.parse()?;
                    cpu_check = true;
                }
                "--cpu-check-max-new" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--cpu-check-max-new requires a value"))?;
                    cpu_check_max_new = value.parse()?;
                    cpu_check = true;
                }
                "--kv-stress" => kv_stress = true,
                "--kv-stress-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--kv-stress-tokens requires a value"))?;
                    kv_stress_tokens = value.parse()?;
                    kv_stress = true;
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
                "--layers" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--layers requires a value"))?;
                    layers = value.parse()?;
                }
                "--prefill-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--prefill-tokens requires a value"))?;
                    prefill_tokens = value.parse()?;
                }
                "--max-new" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--max-new requires a value"))?;
                    max_new_tokens = value.parse()?;
                }
                "--logit-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--logit-tokens requires a comma-separated value"))?;
                    logit_tokens = parse_csv_u32(&value)?;
                }
                _ => return Err(eyre!("unknown argument {arg}")),
            }
        }

        if max_new_tokens == 0 {
            return Err(eyre!("--max-new must be greater than zero"));
        }
        if cpu_check_max_new == 0 {
            return Err(eyre!("--cpu-check-max-new must be greater than zero"));
        }
        if kv_stress_tokens == 0 {
            return Err(eyre!("--kv-stress-tokens must be greater than zero"));
        }
        if sampling.temperature < 0.0 {
            return Err(eyre!("--temperature must be non-negative"));
        }
        if !(0.0..=1.0).contains(&sampling.top_p) {
            return Err(eyre!("--top-p must be in 0..=1"));
        }

        Ok(Self {
            layers,
            prefill_tokens,
            max_new_tokens,
            decode_only,
            cpu_check,
            cpu_check_layers,
            cpu_check_max_new,
            kv_stress,
            kv_stress_tokens,
            sampling,
            logit_tokens,
        })
    }
}

fn repeated_tokens(tokens: &[u32], len: usize) -> Result<Vec<u32>> {
    if tokens.is_empty() {
        return Err(eyre!("cannot build KV stress tokens from an empty fixture"));
    }
    Ok(tokens.iter().copied().cycle().take(len).collect())
}

fn render_greedy_decode_comparison(
    cpu: &GreedyDecodeProbeReport,
    metal: &GreedyDecodeProbeReport,
) -> String {
    let mut out = String::new();
    let cpu_tokens = cpu
        .generated
        .iter()
        .map(|token| token.token)
        .collect::<Vec<_>>();
    let metal_tokens = metal
        .generated
        .iter()
        .map(|token| token.token)
        .collect::<Vec<_>>();
    let max_logit_delta = cpu
        .generated
        .iter()
        .zip(&metal.generated)
        .map(|(cpu, metal)| (cpu.logit - metal.logit).abs())
        .fold(0.0f32, f32::max);

    out.push_str("\ngreedy decode cpu/metal check:\n");
    out.push_str(&format!("- layers: {}\n", cpu.layers));
    out.push_str(&format!("- prompt tokens: {}\n", cpu.prompt_tokens));
    out.push_str(&format!("- max new tokens: {}\n", cpu.max_new_tokens));
    out.push_str(&format!("- token_match: {}\n", cpu_tokens == metal_tokens));
    out.push_str(&format!("- max_logit_delta: {:.9}\n", max_logit_delta));
    out.push_str(&format!("- cpu text: {:?}\n", cpu.text));
    out.push_str(&format!("- metal text: {:?}\n", metal.text));
    out
}

fn parse_csv_u32(value: &str) -> Result<Vec<u32>> {
    value
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| Ok(part.parse()?))
        .collect()
}

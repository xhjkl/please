use eyre::{Result, eyre};
use inference_engine::backend_cpu;
use inference_engine::gptoss_spec::weights;
use inference_engine::harmony_adapter::{HarmonyAdapter, Message, Role};
use inference_engine::model_store;
use inference_engine::{CpuPromptPrefillReport, PromptFixture};

fn main() -> Result<()> {
    let report = model_store::inspect_canonical_safetensors()?;
    let validation = weights::validate_gpt_oss_20b_source(&report);
    let harmony = HarmonyAdapter::gpt_oss()?;
    let mut fixtures = probe_fixtures();
    let fixture_limit = configured_usize("PLEASE_PROBE_FIXTURE_LIMIT", fixtures.len());
    fixtures.truncate(fixture_limit);
    if fixtures.is_empty() {
        return Err(eyre!("probe fixture suite is empty"));
    }

    let prompt_fixtures = fixtures
        .iter()
        .map(|fixture| prompt_fixture(&harmony, fixture))
        .collect::<Result<Vec<_>>>()?;
    let primary = prompt_fixtures.first().expect("checked non-empty");

    let mut out = String::new();
    out.push_str(&report.render_for_cli());
    out.push_str(&validation.render_for_cli());
    out.push_str(&render_fixture_header(&fixtures));

    match backend_cpu::probe_first_prompt_embedding(&report, primary)? {
        Some(probe) => out.push_str(&probe.render_for_cli()),
        None => out.push_str("\ncpu direct SafeTensors probe:\n- skipped: prompt has no tokens\n"),
    }

    match backend_cpu::probe_layer0_math(&report, primary)? {
        Some(probe) => out.push_str(&probe.render_for_cli()),
        None => out.push_str("\ncpu layer-0 math probe:\n- skipped: prompt has no tokens\n"),
    }

    match backend_cpu::probe_single_token_full_stack(&report, primary)? {
        Some(probe) => out.push_str(&probe.render_for_cli()),
        None => {
            out.push_str("\ncpu single-token full-stack probe:\n- skipped: prompt has no tokens\n")
        }
    }

    let mut fixture_reports = Vec::new();
    for fixture in &prompt_fixtures {
        let probe = backend_cpu::probe_prompt_prefill(&report, fixture)?;
        fixture_reports.push(FixtureProbeReport { fixture, probe });
    }

    out.push_str(&render_fixture_summary(&fixture_reports));
    for fixture in &fixture_reports {
        out.push_str(&fixture.fixture.render_for_cli());
        match &fixture.probe {
            Some(probe) => out.push_str(&probe.render_for_cli()),
            None => out.push_str("\ncpu prompt-prefill probe:\n- skipped: prompt has no tokens\n"),
        }
    }

    println!("{out}");
    Ok(())
}

fn prompt_fixture(harmony: &HarmonyAdapter, fixture: &ProbeFixture) -> Result<PromptFixture> {
    let tokens = harmony.render_completion_tokens(&fixture.messages)?;
    let prompt = harmony.decode_utf8(&tokens)?;
    let prompt_token_prefix = tokens.iter().copied().take(16).collect();
    let prompt_token_suffix = tokens
        .iter()
        .copied()
        .skip(tokens.len().saturating_sub(16))
        .collect();

    let prefill_token_count = fixture.prefill_token_limit.min(tokens.len());
    Ok(PromptFixture {
        fixture_name: Some(fixture.name.to_string()),
        prompt_bytes: prompt.len(),
        prompt_token_count: tokens.len(),
        prompt_tokens: tokens,
        prompt_token_prefix,
        prompt_token_suffix,
        prefill_token_count,
    })
}

fn probe_fixtures() -> Vec<ProbeFixture> {
    let prefill_token_limit = configured_usize(
        "PLEASE_PROBE_PREFILL_TOKENS",
        backend_cpu::PROMPT_PREFILL_TOKEN_LIMIT,
    );
    vec![
        text_fixture(
            "plain user prompt",
            [(Role::User, "Summarize the staged diff.")],
            prefill_token_limit,
        ),
        text_fixture(
            "system and developer preamble",
            [
                (Role::System, "You are a local CLI code assistant."),
                (
                    Role::Developer,
                    "Prefer concise answers and preserve exact filenames.",
                ),
                (Role::User, "What changed?"),
            ],
            prefill_token_limit,
        ),
        text_fixture(
            "code text",
            [(
                Role::User,
                "Explain:\n\trust\n\tlet Some(x) = maybe else { return Ok(()); };",
            )],
            prefill_token_limit,
        ),
        text_fixture(
            "unicode text",
            [(
                Role::User,
                "UTF-8 check: cafe, Tbilisi, 東京, zero-width\u{200b}space.",
            )],
            prefill_token_limit,
        ),
    ]
}

fn text_fixture<const N: usize>(
    name: &'static str,
    messages: [(Role, &'static str); N],
    prefill_token_limit: usize,
) -> ProbeFixture {
    ProbeFixture {
        name,
        prefill_token_limit,
        messages: messages
            .into_iter()
            .map(|(role, content)| Message::from((role, content.to_string())))
            .collect(),
    }
}

fn render_fixture_header(fixtures: &[ProbeFixture]) -> String {
    let mut out = String::new();
    out.push_str("\nprobe fixture suite:\n");
    out.push_str(&format!("- fixtures: {}\n", fixtures.len()));
    let prefill_token_limit = fixtures
        .first()
        .map(|fixture| fixture.prefill_token_limit)
        .unwrap_or(backend_cpu::PROMPT_PREFILL_TOKEN_LIMIT);
    out.push_str(&format!(
        "- configured prefill tokens per fixture: {}\n",
        prefill_token_limit
    ));
    out.push_str("- knobs: PLEASE_PROBE_FIXTURE_LIMIT, PLEASE_PROBE_PREFILL_TOKENS\n");
    for fixture in fixtures {
        out.push_str(&format!(
            "  - {}: {} messages, {} prefill tokens\n",
            fixture.name,
            fixture.messages.len(),
            fixture.prefill_token_limit
        ));
    }
    out
}

fn configured_usize(name: &str, default: usize) -> usize {
    let Some(value) = std::env::var(name).ok() else {
        return default;
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn render_fixture_summary(fixtures: &[FixtureProbeReport<'_>]) -> String {
    let mut out = String::new();
    out.push_str("\ncpu prompt-prefill fixture summary:\n");
    for fixture in fixtures {
        let name = fixture
            .fixture
            .fixture_name
            .as_deref()
            .unwrap_or("<unnamed>");
        let Some(probe) = &fixture.probe else {
            out.push_str(&format!("- {name}: skipped\n"));
            continue;
        };
        let top = probe
            .top_logits
            .first()
            .map(|logit| format!("token {} ({:.7})", logit.token, logit.logit))
            .unwrap_or_else(|| "no logits".to_string());
        out.push_str(&format!(
            "- {}: {} tokens, top1 {}\n",
            name,
            probe.prompt_tokens.len(),
            top
        ));
    }
    out
}

struct ProbeFixture {
    name: &'static str,
    messages: Vec<Message>,
    prefill_token_limit: usize,
}

struct FixtureProbeReport<'a> {
    fixture: &'a PromptFixture,
    probe: Option<CpuPromptPrefillReport>,
}

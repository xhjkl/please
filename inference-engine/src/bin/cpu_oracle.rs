use eyre::{Result, eyre};
use inference_engine::backend_cpu;
use inference_engine::model_store;

const DEFAULT_TOKENS: &str = "200006,1428,200008,64614";
const DEFAULT_LOGIT_TOKENS: &str = "277,8526,387,263,278,289,10581,1808";

fn main() -> Result<()> {
    let args = Args::parse()?;
    let report = model_store::inspect_canonical_safetensors()?;
    let oracle = backend_cpu::probe_prompt_prefill_oracle(
        &report,
        &args.tokens,
        args.layers,
        &args.logit_tokens,
    )?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&oracle)?);
        return Ok(());
    }

    println!("rust cpu oracle:");
    println!("- weights: {}", oracle.weights);
    println!("- tokens: {:?}", oracle.tokens);
    println!("- layers: {}", oracle.layers);
    println!(
        "- embedding final first 8: {:?}",
        oracle.embedding_final_first8
    );
    for checkpoint in &oracle.layer_checkpoints {
        println!(
            "- layer {}: final_l2 {:.7}, final_mean {:.7}, final_first8 {:?}",
            checkpoint.layer, checkpoint.final_l2, checkpoint.final_mean, checkpoint.final_first8
        );
    }
    println!("- final_norm first 8: {:?}", oracle.final_norm_first8);
    println!("- selected logits:");
    for logit in &oracle.selected_logits {
        println!("  - token {}: {:.7}", logit.token, logit.logit);
    }

    Ok(())
}

struct Args {
    tokens: Vec<u32>,
    layers: usize,
    logit_tokens: Vec<u32>,
    json: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut tokens = parse_csv_u32(DEFAULT_TOKENS)?;
        let mut layers = 24usize;
        let mut logit_tokens = parse_csv_u32(DEFAULT_LOGIT_TOKENS)?;
        let mut json = false;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--tokens requires a comma-separated value"))?;
                    tokens = parse_csv_u32(&value)?;
                }
                "--layers" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--layers requires a value"))?;
                    layers = value.parse()?;
                }
                "--logit-tokens" => {
                    let value = args
                        .next()
                        .ok_or_else(|| eyre!("--logit-tokens requires a comma-separated value"))?;
                    logit_tokens = parse_csv_u32(&value)?;
                }
                "--json" => json = true,
                _ => return Err(eyre!("unknown argument {arg}")),
            }
        }

        Ok(Self {
            tokens,
            layers,
            logit_tokens,
            json,
        })
    }
}

fn parse_csv_u32(value: &str) -> Result<Vec<u32>> {
    value
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| Ok(part.parse()?))
        .collect()
}

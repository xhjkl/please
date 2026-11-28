//! Measure generation in this process, without executing the model's tool calls.
//!
//! `please bench CHECKPOINT CASES` reads its cases from a tab-separated file and prints a
//! tab-separated row for every exchange it measures, which `column -t` lines up.
use super::dialect::{Delta, FunctionCall};
use super::inference::{self, Engine};
use crate::protocol::{Destination, Message, Setting, ToolResult};
use eyre::{Result, WrapErr, bail, eyre};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// What a row of measurements holds, in `Exchange::row`'s order. Times are seconds from the
/// request; `tokens_per_s` is the decoding speed after the first token; `output` names the
/// generated tokens, so two rows that generated the same reply share it. A cell with nothing
/// to say holds `-`, so every row has every column for `column -t`.
const COLUMNS: [&str; 12] = [
    "case",
    "exchange",
    "start",
    "prompt_tokens",
    "reused_tokens",
    "generated_tokens",
    "first_token_s",
    "first_visible_s",
    "tokens_per_s",
    "total_s",
    "calls",
    "output",
];

/// One benchmark case, a row of the cases file: what opens a new conversation.
#[derive(Debug, PartialEq)]
struct Case {
    name: String,
    max_tokens: usize,
    messages: Vec<Message>,
}

/// Read the cases a tab-separated file holds: a header row naming its columns, then a case
/// per row. `name`, `max_tokens`, and `request` are required. `input` may name a file whose
/// content is piped in with the request; its path is relative to the cases file.
fn cases(path: &Path) -> Result<Vec<Case>> {
    let source = std::fs::read_to_string(path)
        .wrap_err_with(|| format!("cannot read the cases in {}", path.display()))?;
    let mut rows = source.lines().filter(|row| !row.trim().is_empty());
    let header = rows.next();
    let header = header.ok_or_else(|| eyre!("{} has no header row", path.display()))?;
    let header: Vec<&str> = header.split('\t').collect();
    let column = |name: &str| header.iter().position(|&column| column == name);
    let required =
        |name: &str| column(name).ok_or_else(|| eyre!("{} has no `{name}` column", path.display()));
    let (name, max_tokens, request) = (
        required("name")?,
        required("max_tokens")?,
        required("request")?,
    );
    let input = column("input");
    let mut cases = Vec::new();
    for row in rows {
        let cells: Vec<&str> = row.split('\t').collect();
        let cell = |column: usize| cells.get(column).copied().unwrap_or_default();
        let name = cell(name);
        let limit = cell(max_tokens);
        let limit = limit
            .parse()
            .wrap_err_with(|| format!("case {name}: max_tokens {limit:?} is not a number"))?;
        let mut messages = Vec::new();
        let piped = input.map(cell).filter(|file| !file.is_empty());
        if let Some(file) = piped {
            let file = path.with_file_name(file);
            let data = std::fs::read_to_string(&file)
                .wrap_err_with(|| format!("case {name}: cannot read {}", file.display()))?;
            messages.push(Message::Input(data));
        }
        messages.push(Message::User(cell(request).into()));
        cases.push(Case {
            name: name.into(),
            max_tokens: limit,
            messages,
        });
    }
    Ok(cases)
}

/// Measure every case of the cases file on `checkpoint`, loaded into this process, and print
/// each exchange's row as soon as it is measured.
/// Each case has a history of its own, as a new conversation would: `first` starts from the
/// cacheable prefix, which the session still holds from an earlier case or restores from the
/// cache or prefills (`start` says which), and `follow_up` continues it with the reply and
/// stub tool results, as the probe's next delivery would.
pub async fn run(checkpoint: &str, cases_path: &str) -> Result<()> {
    let cases = cases(Path::new(cases_path))?;
    let checkpoint = super::discovery::inspect(checkpoint.into())?;
    let engine = Engine::load(&checkpoint.path, checkpoint.dialect).await?;
    let engine = Arc::new(engine);
    let seconds = engine.load_seconds;
    eprintln!(
        "please bench: loaded {} in {seconds:.1}s",
        engine.path.display()
    );
    // The date and the effort are this invocation's, as any request's would be.
    let setting = crate::setting::current(Destination::Terminal);
    println!("{}", COLUMNS.join("\t"));
    for case in cases {
        let owner = crate::protocol::fresh_id();
        let limit = case.max_tokens;
        let first = exchange(&engine, owner, &setting, case.messages.clone(), limit).await?;
        println!("{}", first.row(&case.name, "first")?);
        let mut follow_up = case.messages;
        let next = follow_up.len() as u64 + 1;
        follow_up.extend(first.reply(next));
        let second = exchange(&engine, owner, &setting, follow_up, limit).await?;
        println!("{}", second.row(&case.name, "follow_up")?);
    }
    Ok(())
}

/// One measured request and what it generated.
struct Exchange {
    generation: inference::Generation,
    tokens: Vec<u32>,
    /// When each token arrived, in seconds from the request.
    times: Vec<f64>,
    first_visible: Option<f64>,
    answer: String,
    thinking: String,
    calls: Result<Vec<FunctionCall>>,
}

/// Generate one reply to the case's history and record each token's arrival, timed where
/// the tokens are made, so no socket stands between the model and the clock.
async fn exchange(
    engine: &Arc<Engine>,
    owner: u64,
    setting: &Setting,
    messages: Vec<Message>,
    max_tokens: usize,
) -> Result<Exchange> {
    let mut parser = engine.output_parser()?;
    let mut first_visible = None;
    let mut answer = String::new();
    let mut thinking = String::new();
    let mut tokens = Vec::new();
    let mut times = Vec::new();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = engine.clone();
    let setting = setting.clone();
    let started = Instant::now();
    let task = tokio::spawn(async move {
        inference::generate(&worker, owner, &setting, &messages, tx, Some(max_tokens)).await
    });
    while let Some(token) = rx.recv().await {
        let elapsed = started.elapsed().as_secs_f64();
        times.push(elapsed);
        tokens.push(token);
        let deltas = parser.push_token(token)?;
        for delta in deltas {
            first_visible.get_or_insert(elapsed);
            match delta {
                Delta::Answer(text) => answer.push_str(&text),
                Delta::Thinking(text) => thinking.push_str(&text),
            }
        }
    }
    let generation = task.await??.ok_or_else(|| eyre!("no tokens generated"))?;
    let calls = parser.finish();
    Ok(Exchange {
        generation,
        tokens,
        times,
        first_visible,
        answer,
        thinking,
        calls,
    })
}

impl Exchange {
    /// What the probe would append before its next delivery, its calls numbered as the hub
    /// would number their entries from `next`; tools are never executed here.
    fn reply(&self, next: u64) -> Vec<Message> {
        let mut messages = super::step::prose(self.thinking.clone(), self.answer.clone());
        let calls = self.calls.as_deref().unwrap_or_default();
        for call in calls {
            let id = next + messages.len() as u64;
            messages.push(Message::Call(call.with_id(id)));
            messages.push(Message::Result(ToolResult {
                id,
                name: call.name.clone(),
                content: r#"{"error":"the benchmark does not execute tools"}"#.into(),
            }));
        }
        if calls.is_empty() {
            messages.push(Message::User("Continue.".into()));
        }
        messages
    }

    /// This exchange's measurements as a row under [`COLUMNS`].
    fn row(&self, case: &str, exchange: &str) -> Result<String> {
        let (Some(&first), Some(&last)) = (self.times.first(), self.times.last()) else {
            bail!("case {case}: no tokens generated");
        };
        let generated = self.tokens.len();
        // The first token comes from the prefill, so it tells nothing of decoding speed.
        let speed = match generated > 1 {
            true => format!("{:.1}", (generated - 1) as f64 / (last - first)),
            false => "-".into(),
        };
        let visible = match self.first_visible {
            Some(seconds) => format!("{seconds:.3}"),
            None => "-".into(),
        };
        let calls = match &self.calls {
            Err(_) => "unparsed".into(),
            Ok(calls) if calls.is_empty() => "-".into(),
            Ok(calls) => {
                let names: Vec<_> = calls.iter().map(|call| call.name.as_str()).collect();
                names.join(",")
            }
        };
        let ids: Vec<u8> = self.tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        let cells: [String; COLUMNS.len()] = [
            case.into(),
            exchange.into(),
            format!("{:?}", self.generation.start).to_lowercase(),
            self.generation.prompt_tokens.to_string(),
            self.generation.reused_tokens.to_string(),
            generated.to_string(),
            format!("{first:.3}"),
            visible,
            speed,
            format!("{last:.3}"),
            calls,
            super::digest::name_of(&ids),
        ];
        Ok(cells.join("\t"))
    }
}

#[cfg(test)]
mod tests;

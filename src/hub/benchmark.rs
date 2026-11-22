//! Measure generation inside the hub without executing the model's tool calls.
use super::{
    dialect::Delta,
    inference::{self, Engine},
};
use crate::protocol::{
    Call, Frame, Message, Setting, ToolResult, read_frame_from_stream, write_frame_to_stream,
};
use eyre::{Result, eyre};
use serde::Deserialize;
use std::{sync::Arc, time::Instant};

/// One benchmark conversation, read from the cases file.
#[derive(Deserialize)]
struct Case {
    name: String,
    max_tokens: usize,
    setting: Setting,
    messages: Vec<Message>,
}

/// Send benchmark cases to a colocated hub using the normal socket boundary.
pub async fn run(model_path: &str, cases_path: &str) -> Result<()> {
    let source = std::fs::read_to_string(cases_path)?;
    let cases: Vec<Case> = serde_json::from_str(&source)?;
    let mut stream = super::spawn(Some(model_path)).await?;
    crate::protocol::greet(&mut stream).await?;
    let mut store = Vec::new();
    for case in cases {
        let request = Frame::BenchmarkRequest {
            setting: case.setting,
            messages: case.messages,
            max_tokens: case.max_tokens,
        };
        write_frame_to_stream(&mut stream, &request).await?;
        let response: Frame = read_frame_from_stream(&mut stream, &mut store, None, None).await?;
        let report = match response {
            Frame::Error(error) => return Err(eyre!(error)),
            Frame::BenchmarkResult(report) => report,
            _ => return Err(eyre!("unexpected benchmark response: {response:?}")),
        };
        let mut report: serde_json::Value = serde_json::from_str(&report)?;
        report["case"] = case.name.into();
        println!("{report}");
    }
    Ok(())
}

/// Capture model-side timings so socket latency does not distort generation measurements.
/// Each case opens a fresh conversation, as a new invocation would: `first` starts from the
/// static head, which the hub's session still holds from an earlier case or restores from the
/// cache or prefills (`start` says which), and `follow_up` continues it with the reply and
/// stub tool results, as the probe's next request would.
pub(super) async fn measure(
    hub: Arc<Engine>,
    setting: Setting,
    messages: Vec<Message>,
    max_tokens: usize,
) -> Result<String> {
    let connection = hub.connect();
    let first = exchange(&hub, connection, &setting, messages.clone(), max_tokens).await?;
    let mut follow_up = messages;
    follow_up.extend(first.reply());
    let second = exchange(&hub, connection, &setting, follow_up, max_tokens).await?;
    let report = serde_json::json!({
        "model": hub.path, "load_seconds": hub.load_seconds,
        "first": first.report()?, "follow_up": second.report()?,
    });
    Ok(report.to_string())
}

/// One measured request and what it generated.
struct Exchange {
    turn: inference::Turn,
    tokens: Vec<u32>,
    times: Vec<f64>,
    first_visible: Option<f64>,
    answer: String,
    thinking: String,
    calls: Result<Vec<super::dialect::ToolCall>>,
}

/// Send one request on the case's connection and record each token's arrival, as the probe
/// would see it, without the socket in between.
async fn exchange(
    hub: &Arc<Engine>,
    connection: u64,
    setting: &Setting,
    messages: Vec<Message>,
    max_tokens: usize,
) -> Result<Exchange> {
    let mut parser = hub.output_parser()?;
    let mut first_visible = None;
    let mut answer = String::new();
    let mut thinking = String::new();
    let mut tokens = Vec::new();
    let mut times = Vec::new();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = hub.clone();
    let setting = setting.clone();
    let started = Instant::now();
    let task = tokio::spawn(async move {
        inference::generate(
            &worker,
            connection,
            &setting,
            &messages,
            tx,
            Some(max_tokens),
        )
        .await
    });
    while let Some(event) = rx.recv().await {
        match event {
            inference::Generated::Stop => break,
            inference::Generated::Token(token) => {
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
        }
    }
    let turn = task.await??.ok_or_else(|| eyre!("no tokens generated"))?;
    let calls = parser.finish();
    Ok(Exchange {
        turn,
        tokens,
        times,
        first_visible,
        answer,
        thinking,
        calls,
    })
}

impl Exchange {
    /// What the probe would append before its next request; tools are never executed here.
    fn reply(&self) -> Vec<Message> {
        let mut messages = Vec::new();
        if !self.thinking.is_empty() {
            messages.push(Message::Thinking(self.thinking.clone()));
        }
        if !self.answer.is_empty() {
            messages.push(Message::Answer(self.answer.clone()));
        }
        let calls = self.calls.as_deref().unwrap_or_default();
        for call in calls {
            messages.push(Message::Call(Call {
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            }));
            messages.push(Message::Result(ToolResult {
                name: call.name.clone(),
                content: r#"{"error":"the benchmark does not execute tools"}"#.into(),
            }));
        }
        if calls.is_empty() {
            messages.push(Message::User("Continue.".into()));
        }
        messages
    }

    fn report(&self) -> Result<serde_json::Value> {
        let first = *self
            .times
            .first()
            .ok_or_else(|| eyre!("no tokens generated"))?;
        let last = *self.times.last().unwrap_or(&first);
        let (tool_calls, parse_error) = match &self.calls {
            Err(error) => (vec![], Some(error.to_string())),
            Ok(calls) => (
                calls
                    .iter()
                    .map(|call| serde_json::json!({"name": call.name, "arguments": call.arguments}))
                    .collect(),
                None,
            ),
        };
        Ok(serde_json::json!({
            "start": format!("{:?}", self.turn.start).to_lowercase(),
            "prompt_tokens": self.turn.prompt_tokens, "reused_tokens": self.turn.reused_tokens,
            "generated_tokens": self.tokens.len(),
            "ttft_seconds": first, "first_visible_seconds": self.first_visible,
            "decode_tokens_per_second": (self.tokens.len() - 1) as f64 / (last - first),
            "generation_seconds": last, "tokens": self.tokens, "token_times_seconds": self.times,
            "answer": self.answer, "thinking": self.thinking,
            "tool_calls": tool_calls, "parse_error": parse_error,
        }))
    }
}

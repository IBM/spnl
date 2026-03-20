use futures::StreamExt;

use async_openai::types::{
    chat::CreateChatCompletionResponse, completions::CreateCompletionResponse,
};
use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use crate::{
    SpnlResult,
    generate::GenerateOptions,
    ir::{Bulk, GenerateMetadata, Map, Message::Assistant, Query, Repeat, to_string},
};

pub enum Spec {
    Map(Map),
    Repeat(Repeat),
}

impl Spec {
    fn n(&self) -> usize {
        match self {
            Spec::Map(m) => m.inputs.len(),
            Spec::Repeat(r) => r.n.into(),
        }
    }

    fn metadata(&self) -> &GenerateMetadata {
        match self {
            Spec::Map(m) => &m.metadata,
            Spec::Repeat(r) => &r.generate.metadata,
        }
    }

    fn query(self) -> Query {
        match self {
            Spec::Map(m) => Query::Bulk(Bulk::Map(m)),
            Spec::Repeat(r) => Query::Bulk(Bulk::Repeat(r)),
        }
    }
}

const DATA_COLON: &[u8] = &[100, 97, 116, 97, 58, 32];

/// Call the /api/query/execute API, passing the given query `spec`
pub async fn generate(
    spec: Spec,
    m: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    let start_time = if options.time {
        Some(::std::time::Instant::now())
    } else {
        None
    };

    let client = reqwest::Client::new();

    // eprintln!("Sending query {:?}", to_string(&query)?);
    let pbs = if options.silent {
        None
    } else {
        super::progress::bars(spec.n(), spec.metadata(), &m, None)?
    };
    let mut response_strings = ::std::iter::repeat_n(String::new(), spec.n()).collect::<Vec<_>>();

    let is_map = matches!(spec, Spec::Map(_));
    let non_streaming = matches!(spec.metadata().max_tokens, Some(1));
    let response = client
        .post("http://localhost:8000/v1/query/execute")
        .query(&[("stream", if non_streaming { "false" } else { "true" })])
        .header("Content-Type", "text/plain")
        .body(to_string(&spec.query())?)
        .send()
        .await?;

    let mut stdout = stdout();
    let quiet = m.is_some() || options.time || options.silent;
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    // Timing tracking
    let mut ttft: Option<::std::time::Duration> = None;
    let mut token_count = 0u64;

    if non_streaming {
        // Non-streaming case. TODO: figure out how to share code
        // between Bulk::Map and Bulk::Repeat cases. The OpenAI data
        // structures for Completion are close but not identical to
        // those for ChatCompletion.
        response_strings = if is_map {
            // Non-streaming Bulk::Map case
            response
                .json::<CreateCompletionResponse>()
                .await?
                .choices
                .into_iter()
                .map(|choice| {
                    token_count += choice.text.len() as u64;
                    choice.text
                })
                .collect()
        } else {
            // Non-streaming Bulk::Repeat case.
            response
                .json::<CreateChatCompletionResponse>()
                .await?
                .choices
                .into_iter()
                .filter_map(|choice| {
                    if let Some(content) = choice.message.content {
                        token_count += content.len() as u64;
                        Some(content)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // For non-streaming, TTFT is essentially the total time
        if let Some(start) = start_time {
            ttft = Some(start.elapsed());
        }
    } else {
        // Streaming case — the server returns SSE (text/event-stream).
        // Each event is formatted as "data: {json}\n\n". Multiple events
        // may arrive in a single chunk, or a single event may be split
        // across chunks. The endpoint always returns completion-stream
        // format (not chat-completion format).
        let mut stream = response.error_for_status()?.bytes_stream();
        let mut buffer = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);

            // Process all complete SSE lines in the buffer.
            loop {
                // Find a complete "data: ...\n" line.
                let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') else {
                    break;
                };
                let line = buffer.drain(..=newline_pos).collect::<Vec<_>>();
                let line = line.strip_suffix(b"\n").unwrap_or(&line);
                let line = line.strip_suffix(b"\r").unwrap_or(line);

                // Skip empty lines (SSE event separators) and non-data lines.
                if line.is_empty() || !line.starts_with(DATA_COLON) {
                    continue;
                }
                let json_data = &line[DATA_COLON.len()..];

                // Skip [DONE] sentinel.
                if json_data == b"[DONE]" {
                    continue;
                }

                // Parse as completion stream response (both map and repeat
                // use the same completions endpoint).
                if let Ok(res) = serde_json::from_slice::<CreateCompletionResponse>(json_data) {
                    for choice in res.choices.iter() {
                        let idx: usize = choice.index.try_into()?;

                        // Track TTFT (time to first token)
                        if ttft.is_none()
                            && !choice.text.is_empty()
                            && let Some(start) = start_time
                        {
                            ttft = Some(start.elapsed());
                        }

                        token_count += choice.text.len() as u64;

                        if !quiet {
                            stdout.write_all(b"\x1b[32m").await?; // green
                            stdout.write_all(choice.text.as_bytes()).await?;
                            stdout.flush().await?;
                            stdout.write_all(b"\x1b[0m").await?; // reset color
                        } else if let Some(ref pbs) = pbs
                            && idx < pbs.len()
                        {
                            pbs[idx].inc(choice.text.len() as u64);
                        }
                        if idx < response_strings.len() {
                            response_strings[idx] += choice.text.as_str();
                        }
                    }
                }
            }
        }
    }

    let response = response_strings
        .into_iter()
        .map(|s| Query::Message(Assistant(s)))
        .collect::<Vec<_>>();

    // Report timing metrics (unless in silent mode)
    if let Some(start) = start_time
        && !options.silent
    {
        let total_time = start.elapsed();
        let task = super::timing::TaskTiming {
            ttft,
            total_duration: total_time,
            token_count,
        };
        super::timing::print_timing_metrics(&[task]);
    }

    if response.len() == 1 {
        Ok(response.into_iter().next().unwrap())
    } else {
        Ok(Query::Par(response))
    }
}

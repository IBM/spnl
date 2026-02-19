use futures::StreamExt;

use async_openai::types::{
    chat::{CreateChatCompletionResponse, CreateChatCompletionStreamResponse},
    completions::CreateCompletionResponse,
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

/// Call the /api/query/{prepare|execute} API, passing the given query `spec`
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

    let exec = if let Some(true) = options.prepare {
        "prepare"
    } else {
        "execute"
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
        .post(format!("http://localhost:8000/v1/query/{exec}"))
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
        response_strings = if let Some(true) = options.prepare {
            vec!["prepared".to_string()]
        } else if is_map {
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
        // Streaming case
        let mut stream = response.error_for_status()?.bytes_stream();
        let mut buffer = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.extend_from_slice(if chunk.starts_with(DATA_COLON) {
                // Hack support for text/event-stream
                // (i.e. Server-Sent Events a.k.a SSE). Each
                // serialized data event is prefixed with "data:
                // ". SSE includes several other non-data events
                // (e.g. begin, end) that we can safely ignore.
                &chunk[DATA_COLON.len()..]
            } else {
                &chunk
            });

            // Process complete JSON objects as they arrive. TODO:
            // figure out how to share code between Bulk::Map and
            // Bulk::Repeat cases.
            if is_map {
                if let Ok(res) = serde_json::from_slice::<CreateCompletionResponse>(&buffer) {
                    buffer.clear();
                    for choice in res.choices.iter() {
                        let idx: usize = choice.index.try_into()?;

                        // Track TTFT (time to first token)
                        if ttft.is_none()
                            && !choice.text.is_empty()
                            && let Some(start) = start_time
                        {
                            ttft = Some(start.elapsed());
                        }

                        // Count tokens (approximate by characters for now)
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
            } else if let Ok(res) =
                serde_json::from_slice::<CreateChatCompletionStreamResponse>(&buffer)
            {
                buffer.clear();
                for choice in res.choices.iter() {
                    if let Some(ref content) = choice.delta.content {
                        let idx: usize = choice.index.try_into()?;

                        // Track TTFT (time to first token)
                        if ttft.is_none()
                            && !content.is_empty()
                            && let Some(start) = start_time
                        {
                            ttft = Some(start.elapsed());
                        }

                        // Count tokens (approximate by characters for now)
                        token_count += content.len() as u64;

                        if !quiet {
                            stdout.write_all(b"\x1b[32m").await?; // green
                            stdout.write_all(content.as_bytes()).await?;
                            stdout.flush().await?;
                            stdout.write_all(b"\x1b[0m").await?; // reset color
                        } else if let Some(ref pbs) = pbs
                            && idx < pbs.len()
                        {
                            pbs[idx].inc(content.len() as u64);
                        }
                        if idx < response_strings.len() {
                            response_strings[idx] += content.as_str();
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

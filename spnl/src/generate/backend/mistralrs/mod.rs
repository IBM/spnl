//! mistral.rs backend for SPNL
//!
//! This backend provides inference using the mistral.rs library, which offers:
//! - Support for multiple model architectures (Llama, Mistral, Qwen, Phi, etc.)
//! - GGUF and quantized model support
//! - Flash Attention on supported hardware
//! - Built-in streaming and batching

use indicatif::MultiProgress;
use mistralrs::{RequestBuilder, Response, TextMessageRole};
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;

use crate::{
    SpnlResult,
    generate::GenerateOptions,
    ir::{Map, Message::*, Query, Repeat},
};

mod loader;
use loader::ModelPool;

#[cfg(feature = "rag")]
pub mod embed;

// Global model pool - initialized once and reused across all requests
static MODEL_POOL: OnceLock<ModelPool> = OnceLock::new();

fn get_model_pool() -> &'static ModelPool {
    MODEL_POOL.get_or_init(ModelPool::new)
}

/// Get the maximum number of parallel tasks from environment variable
/// Defaults to 2 if not set or invalid
fn get_max_parallel() -> usize {
    std::env::var("SPNL_NUM_PARALLEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}

/// Convert collected timing data into TaskTiming structs
fn prepare_timing_metrics(
    all_ttft: &[::std::time::Duration],
    all_task_durations: &[::std::time::Duration],
    all_token_counts: &[u64],
) -> Vec<super::timing::TaskTiming> {
    all_ttft
        .iter()
        .enumerate()
        .map(|(i, &ttft_duration)| {
            let task_duration = all_task_durations.get(i).copied().unwrap_or(ttft_duration);
            let task_tokens = all_token_counts.get(i).copied().unwrap_or(0);

            super::timing::TaskTiming {
                ttft: Some(ttft_duration),
                total_duration: task_duration,
                token_count: task_tokens,
            }
        })
        .collect()
}

/// Generate completions for multiple inputs (Map operation)
pub async fn generate_completion(
    spec: Map,
    mp: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    use tokio::io::AsyncWriteExt;

    let n_prompts = spec.inputs.len();

    let quiet = mp.is_some() || options.time || options.silent;

    // Create progress bars if in quiet mode (but not if silent)
    let pbs = if options.silent {
        None
    } else {
        super::progress::bars(n_prompts, &spec.metadata, &mp, None)?
    };

    // Print "Assistant: " prefix if not in quiet mode
    if !quiet {
        let mut stdout = tokio::io::stdout();
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    // Extract parameters from metadata
    let model_name = spec.metadata.model.clone();
    let temperature = spec.metadata.temperature.unwrap_or(0.6);
    let max_tokens = spec.metadata.max_tokens.unwrap_or(2048);

    // Get or load the model
    let pool = get_model_pool();
    let model = pool.get_or_load(&model_name, options.silent).await?;

    // Timing tracking
    let start_time = if options.time {
        Some(::std::time::Instant::now())
    } else {
        None
    };

    // Create semaphore for concurrency control
    let max_parallel = get_max_parallel();
    let semaphore = Arc::new(Semaphore::new(max_parallel));

    // Process each input in parallel with bounded concurrency
    let mut tasks = Vec::new();
    for (idx, input) in spec.inputs.iter().enumerate() {
        let progress_bar = pbs.as_ref().and_then(|v| v.get(idx).cloned());
        let input = input.clone();
        let model = model.clone();
        let track_timing = options.time;
        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            // Acquire permit before processing
            let _permit = sem.acquire().await.unwrap();

            // Create a simple completion request using the higher-level API
            let request = RequestBuilder::new()
                .add_message(TextMessageRole::User, input)
                .set_sampler_temperature(temperature as f64)
                .set_sampler_max_len(max_tokens as usize);

            // Stream the response
            let mut stream = model.stream_chat_request(request).await?;
            let mut full_text = String::new();
            let mut stdout = tokio::io::stdout();

            // Timing tracking per task
            let mut ttft: Option<::std::time::Duration> = None;
            let mut token_count = 0u64;
            let task_start = if track_timing {
                Some(::std::time::Instant::now())
            } else {
                None
            };

            while let Some(response) = stream.next().await {
                match response {
                    Response::Chunk(chunk) => {
                        // Extract the text from the chunk
                        if let Some(text) =
                            chunk.choices.first().and_then(|c| c.delta.content.as_ref())
                        {
                            // Track TTFT (time to first token)
                            if ttft.is_none()
                                && !text.is_empty()
                                && let Some(start) = task_start
                            {
                                ttft = Some(start.elapsed());
                            }

                            // Count tokens (approximate by characters for now)
                            token_count += text.len() as u64;

                            full_text.push_str(text);

                            // Update progress bar by the number of characters in this chunk
                            if let Some(pb) = &progress_bar {
                                pb.inc(text.len() as u64);
                            }

                            // Stream to stdout if NOT in quiet mode
                            if !quiet {
                                stdout
                                    .write_all(format!("\x1b[32m{}\x1b[0m", text).as_bytes())
                                    .await?;
                                stdout.flush().await?;
                            }
                        }
                    }
                    Response::Done(done) => {
                        // Final response - if we haven't collected any chunks, use this
                        if full_text.is_empty()
                            && let Some(content) = done
                                .choices
                                .first()
                                .and_then(|c| c.message.content.as_ref())
                        {
                            full_text = content.to_string();

                            // Update progress bar for non-streaming response
                            if let Some(pb) = &progress_bar {
                                pb.inc(content.len() as u64);
                            }

                            token_count += content.len() as u64;
                        }
                        break;
                    }
                    Response::ValidationError(e) => {
                        return Err(anyhow::anyhow!("Validation error: {:?}", e));
                    }
                    Response::InternalError(e) => {
                        return Err(anyhow::anyhow!("Internal error: {}", e));
                    }
                    Response::ModelError(e, _) => {
                        return Err(anyhow::anyhow!("Model error: {}", e));
                    }
                    _ => {
                        // Other response types, continue
                    }
                }
            }

            // Calculate task duration
            let task_duration = task_start.map(|start| start.elapsed());

            // Print newline after streaming output (only if not quiet)
            if !quiet && !full_text.is_empty() {
                stdout.write_all(b"\n").await?;
            }

            // Finish progress bar
            if let Some(pb) = &progress_bar {
                pb.finish_and_clear();
            }

            Ok::<_, anyhow::Error>((
                Query::Message(Assistant(full_text)),
                ttft,
                task_duration,
                token_count,
            ))
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;
    let mut final_results = Vec::new();
    let mut all_ttft = Vec::new();
    let mut all_task_durations = Vec::new();
    let mut all_token_counts = Vec::new();

    for result in results {
        let (query, ttft, task_duration, token_count) = result??;
        final_results.push(query);
        if let Some(ttft_val) = ttft {
            all_ttft.push(ttft_val);
        }
        if let Some(duration) = task_duration {
            all_task_durations.push(duration);
        }
        all_token_counts.push(token_count);
    }

    // Print final newline if not in quiet mode
    if !quiet {
        let mut stdout = tokio::io::stdout();
        stdout.write_all(b"\n").await?;
    }

    // Report timing metrics (unless in silent mode)
    if start_time.is_some() && !options.silent {
        let tasks = prepare_timing_metrics(&all_ttft, &all_task_durations, &all_token_counts);
        super::timing::print_timing_metrics(&tasks);
    }

    Ok(Query::Par(final_results))
}

/// Unload all models from the global pool, releasing GPU memory.
/// Call between benchmark runs to avoid accumulating models in VRAM.
pub async fn unload_all_models() {
    get_model_pool().unload_all().await
}

/// Generate multiple completions for the same input (Repeat operation)
pub async fn generate_chat(
    spec: Repeat,
    mp: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    use tokio::io::AsyncWriteExt;

    let n_usize = spec.n as usize;

    let quiet = mp.is_some() || options.time || options.silent;

    // Create progress bars if in quiet mode (but not if silent)
    let pbs = if options.silent {
        None
    } else {
        super::progress::bars(n_usize, &spec.generate.metadata, &mp, None)?
    };

    // Print "Assistant: " prefix if not in quiet mode
    if !quiet {
        let mut stdout = tokio::io::stdout();
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    // Extract the input from the generate spec
    let input_query = spec.generate.input.clone();

    // Extract parameters
    let model_name = spec.generate.metadata.model.clone();
    let temperature = spec.generate.metadata.temperature.unwrap_or(0.6);
    let max_tokens = spec.generate.metadata.max_tokens.unwrap_or(2048);

    // Get or load the model
    let pool = get_model_pool();
    let model = pool.get_or_load(&model_name, options.silent).await?;

    // Create semaphore for concurrency control
    let max_parallel = get_max_parallel();
    let semaphore = Arc::new(Semaphore::new(max_parallel));

    // Timing tracking
    let start_time = if options.time {
        Some(::std::time::Instant::now())
    } else {
        None
    };

    // Generate n completions in parallel with bounded concurrency
    let mut tasks = Vec::new();
    for idx in 0..n_usize {
        let progress_bar = pbs.as_ref().and_then(|v| v.get(idx).cloned());
        let input_query = input_query.clone();
        let model = model.clone();
        let track_timing = options.time;
        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            // Acquire permit before processing
            let _permit = sem.acquire().await.unwrap();

            // Build request with messages from the query
            let mut request_builder = RequestBuilder::new()
                .set_sampler_temperature(temperature as f64)
                .set_sampler_max_len(max_tokens as usize);

            // Add messages from the query
            add_messages_from_query(&mut request_builder, &input_query)?;

            // Stream the response
            let mut stream = model.stream_chat_request(request_builder).await?;
            let mut full_text = String::new();
            let mut stdout = tokio::io::stdout();

            // Timing tracking per task
            let mut ttft: Option<::std::time::Duration> = None;
            let mut token_count = 0u64;
            let task_start = if track_timing {
                Some(::std::time::Instant::now())
            } else {
                None
            };

            while let Some(response) = stream.next().await {
                match response {
                    Response::Chunk(chunk) => {
                        // Extract the text from the chunk
                        if let Some(text) =
                            chunk.choices.first().and_then(|c| c.delta.content.as_ref())
                        {
                            // Track TTFT (time to first token)
                            if ttft.is_none()
                                && !text.is_empty()
                                && let Some(start) = task_start
                            {
                                ttft = Some(start.elapsed());
                            }

                            // Count tokens (approximate by characters for now)
                            token_count += text.len() as u64;

                            full_text.push_str(text);

                            // Update progress bar by the number of characters in this chunk
                            if let Some(pb) = &progress_bar {
                                pb.inc(text.len() as u64);
                            }

                            // Stream to stdout if NOT in quiet mode
                            if !quiet {
                                stdout
                                    .write_all(format!("\x1b[32m{}\x1b[0m", text).as_bytes())
                                    .await?;
                                stdout.flush().await?;
                            }
                        }
                    }
                    Response::Done(done) => {
                        // Final response - if we haven't collected any chunks, use this
                        if full_text.is_empty()
                            && let Some(content) = done
                                .choices
                                .first()
                                .and_then(|c| c.message.content.as_ref())
                        {
                            full_text = content.to_string();

                            // Update progress bar for non-streaming response
                            if let Some(pb) = &progress_bar {
                                pb.inc(content.len() as u64);
                            }

                            token_count += content.len() as u64;
                        }
                        break;
                    }
                    Response::ValidationError(e) => {
                        return Err(anyhow::anyhow!("Validation error: {:?}", e));
                    }
                    Response::InternalError(e) => {
                        return Err(anyhow::anyhow!("Internal error: {}", e));
                    }
                    Response::ModelError(e, _) => {
                        return Err(anyhow::anyhow!("Model error: {}", e));
                    }
                    _ => {
                        // Other response types, continue
                    }
                }
            }

            // Calculate task duration
            let task_duration = task_start.map(|start| start.elapsed());

            // Print newline after streaming output (only if not quiet)
            if !quiet && !full_text.is_empty() {
                stdout.write_all(b"\n").await?;
            }

            // Finish progress bar
            if let Some(pb) = &progress_bar {
                pb.finish_and_clear();
            }

            Ok::<_, anyhow::Error>((
                Query::Message(Assistant(full_text)),
                ttft,
                task_duration,
                token_count,
            ))
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;
    let mut final_results = Vec::new();
    let mut all_ttft = Vec::new();
    let mut all_task_durations = Vec::new();
    let mut all_token_counts = Vec::new();

    for result in results {
        let (query, ttft, task_duration, token_count) = result??;
        final_results.push(query);
        if let Some(ttft_val) = ttft {
            all_ttft.push(ttft_val);
        }
        if let Some(duration) = task_duration {
            all_task_durations.push(duration);
        }
        all_token_counts.push(token_count);
    }

    // Print final newline if not in quiet mode
    if !quiet {
        let mut stdout = tokio::io::stdout();
        stdout.write_all(b"\n").await?;
    }

    // Report timing metrics (unless in silent mode)
    if start_time.is_some() && !options.silent {
        let tasks = prepare_timing_metrics(&all_ttft, &all_task_durations, &all_token_counts);
        super::timing::print_timing_metrics(&tasks);
    }

    Ok(Query::Par(final_results))
}

/// Add messages from a Query to a RequestBuilder.
/// When `in_plus` is true, messages are tagged with a `pic_plus` metadata field
/// so the engine can identify Plus blocks after tokenization.
fn add_messages_from_query(builder: &mut RequestBuilder, query: &Query) -> anyhow::Result<()> {
    add_messages_from_query_inner(builder, query, false)
}

fn add_messages_from_query_inner(
    builder: &mut RequestBuilder,
    query: &Query,
    in_plus: bool,
) -> anyhow::Result<()> {
    match query {
        Query::Message(msg) => {
            let (role, content) = match msg {
                User(content) => (TextMessageRole::User, content),
                Assistant(content) => (TextMessageRole::Assistant, content),
                System(content) => (TextMessageRole::System, content),
            };
            if in_plus {
                // Tag this message so the engine knows it's a Plus block.
                // Uses add_message_with_metadata to attach an in-band flag
                // that add_request.rs will extract and strip before templating.
                *builder = builder
                    .clone()
                    .add_message(role, format!("\x00PIC_PLUS\x00{}", content));
            } else {
                *builder = builder.clone().add_message(role, content.clone());
            }
        }
        Query::Plus(queries) => {
            for q in queries {
                add_messages_from_query_inner(builder, q, true)?;
            }
        }
        Query::Seq(queries) | Query::Par(queries) | Query::Cross(queries) => {
            for q in queries {
                add_messages_from_query_inner(builder, q, in_plus)?;
            }
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported query type for chat generation"
            ));
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "local"))]
mod tests {
    use super::*;
    use mistralrs::{MessageContent, RequestLike};

    /// Extract the text content of a message (the "content" key, Left variant).
    fn content_str(msg: &indexmap::IndexMap<String, MessageContent>) -> &str {
        match msg.get("content").unwrap() {
            either::Either::Left(s) => s.as_str(),
            _ => panic!("expected Left(String) content"),
        }
    }

    #[test]
    fn plain_user_message() {
        let query = Query::Message(User("hello".to_string()));
        let mut builder = RequestBuilder::new();
        add_messages_from_query_inner(&mut builder, &query, false).unwrap();
        let msgs = builder.messages_ref();
        assert_eq!(msgs.len(), 1);
        assert_eq!(content_str(&msgs[0]), "hello");
    }

    #[test]
    fn plus_tags_messages() {
        let query = Query::Plus(vec![Query::Message(User("doc1".to_string()))]);
        let mut builder = RequestBuilder::new();
        add_messages_from_query(&mut builder, &query).unwrap();
        let msgs = builder.messages_ref();
        assert_eq!(msgs.len(), 1);
        assert!(
            content_str(&msgs[0]).starts_with("\x00PIC_PLUS\x00"),
            "expected PIC_PLUS prefix, got: {:?}",
            content_str(&msgs[0])
        );
    }

    #[test]
    fn seq_inside_plus_all_tagged() {
        let query = Query::Plus(vec![Query::Seq(vec![
            Query::Message(User("a".to_string())),
            Query::Message(User("b".to_string())),
        ])]);
        let mut builder = RequestBuilder::new();
        add_messages_from_query(&mut builder, &query).unwrap();
        let msgs = builder.messages_ref();
        assert_eq!(msgs.len(), 2);
        for msg in msgs {
            assert!(
                content_str(msg).starts_with("\x00PIC_PLUS\x00"),
                "expected PIC_PLUS prefix, got: {:?}",
                content_str(msg)
            );
        }
    }

    #[test]
    fn plus_inside_cross() {
        let query = Query::Cross(vec![
            Query::Message(User("q".to_string())),
            Query::Plus(vec![Query::Message(User("doc".to_string()))]),
        ]);
        let mut builder = RequestBuilder::new();
        add_messages_from_query(&mut builder, &query).unwrap();
        let msgs = builder.messages_ref();
        assert_eq!(msgs.len(), 2);
        // First message: plain (no prefix)
        assert_eq!(content_str(&msgs[0]), "q");
        // Second message: tagged with PIC_PLUS
        assert_eq!(content_str(&msgs[1]), "\x00PIC_PLUS\x00doc");
    }

    #[test]
    fn unsupported_query_type() {
        use crate::ir::{Generate, GenerateMetadata};
        let query = Query::Generate(Generate {
            metadata: GenerateMetadata {
                model: "test".to_string(),
                max_tokens: Some(1),
                temperature: Some(0.0),
            },
            input: Box::new(Query::Message(User("hi".to_string()))),
        });
        let mut builder = RequestBuilder::new();
        let result = add_messages_from_query(&mut builder, &query);
        assert!(result.is_err());
    }
}

// Made with Bob

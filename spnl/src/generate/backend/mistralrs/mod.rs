//! mistral.rs backend for SPNL
//!
//! This backend provides inference using the mistral.rs library, which offers:
//! - Support for multiple model architectures (Llama, Mistral, Qwen, Phi, etc.)
//! - GGUF and quantized model support
//! - Flash Attention on supported hardware
//! - Built-in streaming and batching

use indicatif::MultiProgress;
use mistralrs::{RequestBuilder, Response, TextMessageRole};
use std::sync::OnceLock;

use crate::{
    SpnlResult,
    generate::GenerateOptions,
    ir::{Map, Message::*, Query, Repeat},
};

mod loader;
use loader::ModelPool;

// Global model pool - initialized once and reused across all requests
static MODEL_POOL: OnceLock<ModelPool> = OnceLock::new();

fn get_model_pool() -> &'static ModelPool {
    MODEL_POOL.get_or_init(ModelPool::new)
}

/// Generate completions for multiple inputs (Map operation)
pub async fn generate_completion(
    spec: Map,
    mp: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    use tokio::io::AsyncWriteExt;

    if let Some(true) = options.prepare {
        return Err(anyhow::anyhow!(
            "Prepare mode not supported for mistralrs backend"
        ));
    }

    let n_prompts = spec.inputs.len();

    // Determine if we're in quiet mode (same logic as Candle backend)
    let start_time = match (n_prompts, options.time.as_ref()) {
        (1, Some(crate::WhatToTime::Gen1))
        | (_, Some(crate::WhatToTime::Gen))
        | (_, Some(crate::WhatToTime::All)) => Some(::std::time::Instant::now()),
        _ => None,
    };
    let quiet = mp.is_some() || start_time.is_some();

    // Create progress bars if in quiet mode
    let pbs = super::progress::bars(n_prompts, &spec.metadata, &mp, None)?;

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
    let model = pool.get_or_load(&model_name).await?;

    // Process each input in parallel
    let mut tasks = Vec::new();
    for (idx, input) in spec.inputs.iter().enumerate() {
        let progress_bar = pbs.as_ref().and_then(|v| v.get(idx).cloned());
        let input = input.clone();
        let model = model.clone();

        let task = tokio::spawn(async move {
            // Create a simple completion request using the higher-level API
            let request = RequestBuilder::new()
                .add_message(TextMessageRole::User, input)
                .set_sampler_temperature(temperature as f64)
                .set_sampler_max_len(max_tokens as usize);

            // Stream the response
            let mut stream = model.stream_chat_request(request).await?;
            let mut full_text = String::new();
            let mut stdout = tokio::io::stdout();

            while let Some(response) = stream.next().await {
                match response {
                    Response::Chunk(chunk) => {
                        // Extract the text from the chunk
                        if let Some(text) =
                            chunk.choices.first().and_then(|c| c.delta.content.as_ref())
                        {
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

            // Print newline after streaming output (only if not quiet)
            if !quiet && !full_text.is_empty() {
                stdout.write_all(b"\n").await?;
            }

            // Finish progress bar
            if let Some(pb) = &progress_bar {
                pb.finish_and_clear();
            }

            Ok::<_, anyhow::Error>(Query::Message(Assistant(full_text)))
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;
    let mut final_results = Vec::new();
    for result in results {
        final_results.push(result??);
    }

    // Print final newline if not in quiet mode
    if !quiet {
        let mut stdout = tokio::io::stdout();
        stdout.write_all(b"\n").await?;
    }

    Ok(Query::Par(final_results))
}

/// Generate multiple completions for the same input (Repeat operation)
pub async fn generate_chat(
    spec: Repeat,
    mp: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    use tokio::io::AsyncWriteExt;

    if let Some(true) = options.prepare {
        return Err(anyhow::anyhow!(
            "Prepare mode not supported for mistralrs backend"
        ));
    }

    let n_usize = spec.n as usize;

    // Determine if we're in quiet mode (same logic as Candle backend)
    let start_time = match (n_usize, options.time.as_ref()) {
        (1, Some(crate::WhatToTime::Gen1))
        | (_, Some(crate::WhatToTime::Gen))
        | (_, Some(crate::WhatToTime::All)) => Some(::std::time::Instant::now()),
        _ => None,
    };
    let quiet = mp.is_some() || start_time.is_some();

    // Create progress bars if in quiet mode
    let pbs = super::progress::bars(n_usize, &spec.generate.metadata, &mp, None)?;

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
    let model = pool.get_or_load(&model_name).await?;

    // Generate n completions in parallel
    let mut tasks = Vec::new();
    for idx in 0..n_usize {
        let progress_bar = pbs.as_ref().and_then(|v| v.get(idx).cloned());
        let input_query = input_query.clone();
        let model = model.clone();

        let task = tokio::spawn(async move {
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

            while let Some(response) = stream.next().await {
                match response {
                    Response::Chunk(chunk) => {
                        // Extract the text from the chunk
                        if let Some(text) =
                            chunk.choices.first().and_then(|c| c.delta.content.as_ref())
                        {
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

            // Print newline after streaming output (only if not quiet)
            if !quiet && !full_text.is_empty() {
                stdout.write_all(b"\n").await?;
            }

            // Finish progress bar
            if let Some(pb) = &progress_bar {
                pb.finish_and_clear();
            }

            Ok::<_, anyhow::Error>(Query::Message(Assistant(full_text)))
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;
    let mut final_results = Vec::new();
    for result in results {
        final_results.push(result??);
    }

    // Print final newline if not in quiet mode
    if !quiet {
        let mut stdout = tokio::io::stdout();
        stdout.write_all(b"\n").await?;
    }

    Ok(Query::Par(final_results))
}

/// Add messages from a Query to a RequestBuilder
fn add_messages_from_query(builder: &mut RequestBuilder, query: &Query) -> anyhow::Result<()> {
    match query {
        Query::Message(msg) => {
            let (role, content) = match msg {
                User(content) => (TextMessageRole::User, content),
                Assistant(content) => (TextMessageRole::Assistant, content),
                System(content) => (TextMessageRole::System, content),
            };
            *builder = builder.clone().add_message(role, content.clone());
        }
        Query::Seq(queries)
        | Query::Par(queries)
        | Query::Plus(queries)
        | Query::Cross(queries) => {
            for q in queries {
                add_messages_from_query(builder, q)?;
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

// Made with Bob

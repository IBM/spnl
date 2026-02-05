mod loader;
mod model_pool;

pub mod gemma2;
pub mod gemma3;
pub mod llama;
pub mod model;
pub mod quantized_qwen2;
pub mod quantized_qwen3;
pub mod quantized_qwen3_moe;
pub mod qwen2;
pub mod qwen3;
pub mod qwen3_moe;

use model_pool::ModelPool;
use std::sync::OnceLock;

// Global model pool - initialized once and reused across all requests
static MODEL_POOL: OnceLock<ModelPool> = OnceLock::new();

fn get_model_pool() -> &'static ModelPool {
    MODEL_POOL.get_or_init(ModelPool::new)
}

use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};
use tokio::sync::mpsc;

use super::shared::tokenize_with_chat_template;
use crate::{
    SpnlResult,
    generate::GenerateOptions,
    ir::{Map, Message::*, Query, Repeat},
};

// Get maximum number of parallel inference operations from environment or use default
fn get_candle_num_parallel() -> usize {
    std::env::var("CANDLE_NUM_PARALLEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

// Work item for the queue
struct WorkItem {
    idx: usize,
    input: Query,
    temperature: f64,
    max_tokens: usize,
    model_path: String,
    progress_bar: Option<indicatif::ProgressBar>,
    quiet: bool,
}

// Result from a worker
struct WorkResult {
    idx: usize,
    generated: String,
}

pub use gemma2::{Gemma2GenericConfig, Gemma2ModelWrapper};
pub use gemma3::{Gemma3GenericConfig, Gemma3ModelWrapper};
pub use llama::{LlamaGenericConfig, LlamaModelWrapper};
pub use model::{CandleModel, TokenCallback};
pub use qwen2::{Qwen2GenericConfig, Qwen2ModelWrapper};
pub use qwen3::{Qwen3GenericConfig, Qwen3ModelWrapper};
pub use qwen3_moe::{Qwen3MoeGenericConfig, Qwen3MoeModelWrapper};

pub async fn generate_completion(
    spec: Map,
    m: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    if let Some(true) = options.prepare {
        return Err(anyhow::anyhow!(
            "Prepare mode not supported for candle backend"
        ));
    }

    let n_prompts = spec.inputs.len();
    let mut stdout = stdout();

    // Extract max tokens
    let mt = spec
        .metadata
        .max_tokens
        .map(|mt| match mt {
            0 => 2048,
            _ => mt as usize,
        })
        .unwrap_or(2048);

    let start_time = match (mt, &options.time) {
        (1, Some(crate::WhatToTime::Gen1))
        | (_, Some(crate::WhatToTime::Gen))
        | (_, Some(crate::WhatToTime::All)) => Some(::std::time::Instant::now()),
        _ => None,
    };
    let quiet = m.is_some() || start_time.is_some();

    let pbs = super::progress::bars(n_prompts, &spec.metadata, &m, Some(1))?;

    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    // Create work queue and result channel
    let num_workers = get_candle_num_parallel();
    let (work_tx, work_rx) = mpsc::channel::<WorkItem>(n_prompts);
    let (result_tx, mut result_rx) = mpsc::channel::<WorkResult>(n_prompts);

    // Create a shared work receiver using Arc and Mutex
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));

    // Spawn worker tasks
    let workers: Vec<_> = (0..num_workers)
        .map(|_| {
            let work_rx = std::sync::Arc::clone(&work_rx);
            let result_tx = result_tx.clone();

            tokio::task::spawn_blocking(move || {
                // Get the model pool reference
                let pool = get_model_pool();

                loop {
                    let item = {
                        let mut rx = work_rx.lock().unwrap();
                        rx.blocking_recv()
                    };

                    let item = match item {
                        Some(item) => item,
                        None => break, // Channel closed
                    };

                    // Get model from pool (will load if not cached)
                    let cached_model = match pool.get_or_load(&item.model_path) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("Failed to get model from pool: {}", e);
                            continue;
                        }
                    };

                    // Lock the cached model for this generation
                    let mut cached = match cached_model.lock() {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Failed to lock model: {}", e);
                            continue;
                        }
                    };

                    let (tokenizer, device, dtype) = cached.resources();

                    // Tokenize the input with chat template
                    let tokens = match tokenize_with_chat_template(
                        &item.input,
                        &tokenizer,
                        &item.model_path,
                    ) {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            eprintln!("Tokenization with chat template failed: {}", e);
                            continue;
                        }
                    };

                    // Create streaming callback if not in quiet mode
                    let mut callback: Option<TokenCallback> = if !item.quiet {
                        Some(Box::new(move |token: &str| {
                            let mut stdout = std::io::stdout();
                            use std::io::Write;
                            write!(stdout, "\x1b[32m{}\x1b[0m", token)?;
                            stdout.flush()?;
                            Ok(())
                        }))
                    } else {
                        None
                    };

                    let config = model::GenerateConfig {
                        device: &device,
                        dtype,
                        max_tokens: item.max_tokens,
                        temperature: item.temperature,
                        top_p: None,         // TODO: Add to GenerateOptions if needed
                        _top_k: None,        // TODO: Add to GenerateOptions if needed
                        repeat_penalty: 1.1, // Default repeat penalty
                        seed: 299792458,     // Default seed (speed of light in m/s)
                        tokenizer: &tokenizer,
                        progress_bar: item.progress_bar.as_ref(),
                    };

                    match cached.generate(&tokens, config, callback.as_mut()) {
                        Ok(generated) => {
                            let _ = result_tx.blocking_send(WorkResult {
                                idx: item.idx,
                                generated,
                            });
                        }
                        Err(e) => {
                            eprintln!("Generation failed: {}", e);
                        }
                    }
                }
            })
        })
        .collect();

    // Send work items to queue
    for (idx, prompt) in spec.inputs.iter().enumerate() {
        let temperature = spec.metadata.temperature.unwrap_or(0.7) as f64;
        let progress_bar = pbs.as_ref().and_then(|pbs| pbs.get(idx)).cloned();

        work_tx
            .send(WorkItem {
                idx,
                input: Query::Message(User(prompt.clone())),
                temperature,
                max_tokens: mt,
                model_path: spec.metadata.model.clone(),
                progress_bar,
                quiet,
            })
            .await?;
    }

    // Close the work queue
    drop(work_tx);
    drop(result_tx);

    // Collect results
    let mut response_strings = vec![String::new(); n_prompts];
    while let Some(result) = result_rx.recv().await {
        response_strings[result.idx] = result.generated;
    }

    // Wait for all workers to finish
    for worker in workers {
        let _ = worker.await;
    }

    if !quiet {
        stdout.write_all(b"\n").await?;
    }

    let response = response_strings
        .into_iter()
        .map(|s| Query::Message(Assistant(s)))
        .collect::<Vec<_>>();

    if let Some(start_time) = start_time {
        println!("GenerateTime {} ns", start_time.elapsed().as_nanos())
    }

    if response.len() == 1 {
        Ok(response.into_iter().next().unwrap())
    } else {
        Ok(Query::Par(response))
    }
}

pub async fn generate_chat(
    spec: Repeat,
    m: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    if let Some(true) = options.prepare {
        return Err(anyhow::anyhow!(
            "Prepare mode not supported for candle backend"
        ));
    }

    let mut stdout = stdout();

    // Extract max tokens
    let mt = spec
        .generate
        .metadata
        .max_tokens
        .map(|mt| match mt {
            0 => 2048,
            _ => mt as usize,
        })
        .unwrap_or(2048);

    let start_time = match (mt, &options.time) {
        (1, Some(crate::WhatToTime::Gen1))
        | (_, Some(crate::WhatToTime::Gen))
        | (_, Some(crate::WhatToTime::All)) => Some(::std::time::Instant::now()),
        _ => None,
    };
    let quiet = m.is_some() || start_time.is_some();

    let n_usize: usize = spec.n.into();
    let pbs = super::progress::bars(n_usize, &spec.generate.metadata, &m, Some(1))?;

    // Store the input query for chat template processing (dereference the Box)
    let input_query = *spec.generate.input.clone();

    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    // Create work queue and result channel
    let num_workers = get_candle_num_parallel();
    let (work_tx, work_rx) = mpsc::channel::<WorkItem>(n_usize);
    let (result_tx, mut result_rx) = mpsc::channel::<WorkResult>(n_usize);

    // Create a shared work receiver using Arc and Mutex
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));

    // Spawn worker tasks
    let workers: Vec<_> = (0..num_workers)
        .map(|_| {
            let work_rx = std::sync::Arc::clone(&work_rx);
            let result_tx = result_tx.clone();

            tokio::task::spawn_blocking(move || {
                // Get the model pool reference
                let pool = get_model_pool();

                loop {
                    let item = {
                        let mut rx = work_rx.lock().unwrap();
                        rx.blocking_recv()
                    };

                    let item = match item {
                        Some(item) => item,
                        None => break, // Channel closed
                    };

                    // Get model from pool (will load if not cached)
                    let cached_model = match pool.get_or_load(&item.model_path) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("Failed to get model from pool: {}", e);
                            continue;
                        }
                    };

                    // Lock the cached model for this generation
                    let mut cached = match cached_model.lock() {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Failed to lock model: {}", e);
                            continue;
                        }
                    };

                    let (tokenizer, device, dtype) = cached.resources();

                    // Tokenize the input with chat template
                    let tokens = match tokenize_with_chat_template(
                        &item.input,
                        &tokenizer,
                        &item.model_path,
                    ) {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            eprintln!("Tokenization with chat template failed: {}", e);
                            continue;
                        }
                    };

                    // Create streaming callback if not in quiet mode
                    let mut callback: Option<TokenCallback> = if !item.quiet {
                        Some(Box::new(move |token: &str| {
                            let mut stdout = std::io::stdout();
                            use std::io::Write;
                            write!(stdout, "\x1b[32m{}\x1b[0m", token)?;
                            stdout.flush()?;
                            Ok(())
                        }))
                    } else {
                        None
                    };

                    let config = model::GenerateConfig {
                        device: &device,
                        dtype,
                        max_tokens: item.max_tokens,
                        temperature: item.temperature,
                        top_p: None,         // TODO: Add to GenerateOptions if needed
                        _top_k: None,        // TODO: Add to GenerateOptions if needed
                        repeat_penalty: 1.1, // Default repeat penalty
                        seed: 299792458,     // Default seed (speed of light in m/s)
                        tokenizer: &tokenizer,
                        progress_bar: item.progress_bar.as_ref(),
                    };

                    match cached.generate(&tokens, config, callback.as_mut()) {
                        Ok(generated) => {
                            let _ = result_tx.blocking_send(WorkResult {
                                idx: item.idx,
                                generated,
                            });
                        }
                        Err(e) => {
                            eprintln!("Generation failed: {}", e);
                        }
                    }
                }
            })
        })
        .collect();

    // Send work items to queue
    let temperature = spec.generate.metadata.temperature.unwrap_or(0.7) as f64;
    for idx in 0..n_usize {
        let progress_bar = pbs.as_ref().and_then(|pbs| pbs.get(idx)).cloned();

        work_tx
            .send(WorkItem {
                idx,
                input: input_query.clone(),
                temperature,
                max_tokens: mt,
                model_path: spec.generate.metadata.model.clone(),
                progress_bar,
                quiet,
            })
            .await?;
    }

    // Close the work queue
    drop(work_tx);
    drop(result_tx);

    // Collect results
    let mut response_strings = vec![String::new(); n_usize];
    while let Some(result) = result_rx.recv().await {
        response_strings[result.idx] = result.generated;
    }

    // Wait for all workers to finish
    for worker in workers {
        let _ = worker.await;
    }

    if !quiet {
        stdout.write_all(b"\n").await?;
    }

    let response = response_strings
        .into_iter()
        .map(|s| Query::Message(Assistant(s)))
        .collect::<Vec<_>>();

    if let Some(start_time) = start_time {
        println!("GenerateTime {} ns", start_time.elapsed().as_nanos())
    }

    if response.len() == 1 {
        Ok(response.into_iter().next().unwrap())
    } else {
        Ok(Query::Par(response))
    }
}

// Made with Bob

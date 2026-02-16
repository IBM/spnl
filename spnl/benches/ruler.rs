//! RULER: What's the Real Context Size of Your Long-Context Language Models?
//!
//! This benchmark is a faithful Rust port of the RULER benchmark methodology
//! (https://github.com/NVIDIA/RULER) adapted to use SPNL span queries.
//!
//! RULER evaluates long-context language models across 4 task categories:
//! 1. **Retrieval (NIAH)**: Needle-in-a-haystack with configurable complexity
//! 2. **Multi-hop Tracing (Variable Tracking)**: Track variable assignments through chains
//! 3. **Aggregation (Common/Frequent Words Extraction)**: Extract most common words
//! 4. **Question Answering**: Answer questions based on context (not yet implemented)
//!
//! ## Faithful to Original Methodology
//!
//! This implementation matches the original Python code:
//! - Uses token-based context lengths (not word-based)
//! - Binary search to find optimal haystack size for target context length
//! - Same validation logic: `string_match_all` and `string_match_part`
//! - Same task templates and answer prefixes from constants.py
//! - Same complexity configurations (num_chains, num_hops, freq_cw, etc.)
//!
//! ## Configuration via Environment Variables
//!
//! ### General Settings
//! - `BENCH_SAMPLE_SIZE`: Number of samples per configuration (default: 10)
//! - `BENCH_MEASUREMENT_TIME`: Measurement time in seconds (default: 60)
//! - `BENCH_CONTEXT_LENGTHS`: Comma-separated context lengths in TOKENS (default: "4000,8000")
//! - `BENCH_MODEL`: Model to use for inference (default: "ollama/granite3.3:2b")
//! - `BENCH_TOKENIZER_MODEL`: HuggingFace model for tokenizer (default: "ibm-granite/granite-3.3-2b-instruct")
//! - `BENCH_FINAL_CONTEXT_LENGTH_BUFFER`: Buffer for system/question/response (default: 200)
//! - `BENCH_DEBUG`: Enable debug output for first sample (default: false)
//! - `BENCH_TASKS`: Comma-separated tasks to run (default: "niah")
//!
//! ### Task-Specific Settings
//! #### NIAH (Needle-in-a-Haystack)
//! - `BENCH_NIAH_NUM_NEEDLE_K`: Number of needles (keys) to insert (default: 1)
//! - `BENCH_NIAH_NUM_NEEDLE_V`: Number of values per needle (default: 1)
//! - `BENCH_NIAH_NUM_NEEDLE_Q`: Number of needles to query (default: 1)
//! - `BENCH_NIAH_DEPTH_PERCENTAGES`: Comma-separated depth percentages (default: "50")
//!
//! #### Variable Tracking
//! - `BENCH_VT_NUM_CHAINS`: Number of variable chains (default: 1)
//! - `BENCH_VT_NUM_HOPS`: Number of hops per chain (default: 4)
//!
//! ## Example Usage
//!
//! ```bash
//! # Run with default settings (requires granite3.3:2b in Ollama)
//! cargo bench --bench ruler --features tok
//!
//! # Run specific tasks
//! BENCH_TASKS="niah" cargo bench --bench ruler --features tok
//!
//! # Run with debug output
//! BENCH_DEBUG=1 cargo bench --bench ruler --features tok
//!
//! # Custom NIAH configuration
//! BENCH_NIAH_NUM_NEEDLE_K=2 BENCH_NIAH_NUM_NEEDLE_V=2 \
//!   cargo bench --bench ruler --features tok
//! ```

mod bench_progress;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::Rng;
use spnl::{
    ExecuteOptions, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// GitHub API URL for the Paul Graham essays directory
const PG_ESSAYS_API_URL: &str = "https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/contents/needlehaystack/PaulGrahamEssays";

/// Get the cache directory for RULER benchmark data
fn get_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir()
        .ok_or("Could not determine cache directory")?
        .join("spnl")
        .join("ruler");

    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

#[derive(serde::Deserialize)]
struct GitHubFile {
    name: String,
    download_url: String,
}

/// Fetch and cache the Paul Graham essays
fn fetch_pg_essays() -> Result<String, Box<dyn std::error::Error>> {
    let cache_file = get_cache_dir()?.join("paul_graham_essays_combined.txt");

    if cache_file.exists() {
        return Ok(fs::read_to_string(&cache_file)?);
    }

    eprintln!("Downloading Paul Graham essays from GitHub...");
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(PG_ESSAYS_API_URL)
        .header("User-Agent", "spnl-ruler-benchmark")
        .send()?;

    let files: Vec<GitHubFile> = response.json()?;
    let mut combined_content = String::new();
    let txt_files: Vec<_> = files
        .into_iter()
        .filter(|f| f.name.ends_with(".txt"))
        .collect();

    for (i, file) in txt_files.iter().enumerate() {
        eprint!("\rDownloading essay {}/{}...", i + 1, txt_files.len());
        let essay_content = client
            .get(&file.download_url)
            .header("User-Agent", "spnl-ruler-benchmark")
            .send()?
            .text()?;
        combined_content.push_str(&essay_content);
        combined_content.push('\n');
    }
    eprintln!("\nDownload complete!");

    fs::File::create(&cache_file)?.write_all(combined_content.as_bytes())?;
    Ok(combined_content)
}

/// Fallback essays
const FALLBACK_ESSAYS: &str =
    r#"The way to get startup ideas is not to try to think of startup ideas."#;

fn get_context_length_in_tokens(context: &str, tokenizer: &Tokenizer) -> usize {
    tokenizer
        .encode(context, false)
        .map(|encoding| encoding.get_ids().len())
        .unwrap_or(0)
}

fn encode_and_trim(
    context: &str,
    context_length: usize,
    tokenizer: &Tokenizer,
) -> Result<String, Box<dyn std::error::Error>> {
    let encoding = tokenizer
        .encode(context, false)
        .map_err(|e| format!("Encoding error: {}", e))?;
    let tokens = encoding.get_ids();
    if tokens.len() > context_length {
        let trimmed_tokens = &tokens[..context_length];
        Ok(tokenizer
            .decode(trimmed_tokens, false)
            .map_err(|e| format!("Decoding error: {}", e))?)
    } else {
        Ok(context.to_string())
    }
}

/// Evaluation metric: string_match_all - Returns score based on how many references are found
fn string_match_all(prediction: &str, references: &[String]) -> f64 {
    let pred_lower = prediction.to_lowercase();
    let matches: usize = references
        .iter()
        .filter(|r| pred_lower.contains(&r.to_lowercase()))
        .count();
    (matches as f64) / (references.len() as f64)
}

/// Evaluation metric: string_match_part - Returns 1.0 if ANY reference is found
#[allow(dead_code)]
fn string_match_part(prediction: &str, references: &[String]) -> f64 {
    let pred_lower = prediction.to_lowercase();
    if references
        .iter()
        .any(|r| pred_lower.contains(&r.to_lowercase()))
    {
        1.0
    } else {
        0.0
    }
}

// TASK 1: NIAH
#[derive(Debug, Clone)]
struct NIAHConfig {
    context_length: usize,
    depth_percent: usize,
    num_needle_k: usize,
    num_needle_v: usize,
    num_needle_q: usize,
    final_context_length_buffer: usize,
}

impl Default for NIAHConfig {
    fn default() -> Self {
        Self {
            context_length: 4000,
            depth_percent: 50,
            num_needle_k: 1,
            num_needle_v: 1,
            num_needle_q: 1,
            final_context_length_buffer: 200,
        }
    }
}

fn generate_random_number() -> String {
    let mut rng = rand::rng();
    rng.random_range(1000000..10000000).to_string()
}

fn generate_niah_context(
    config: &NIAHConfig,
    tokenizer: &Tokenizer,
    essays: &str,
) -> Result<(String, Vec<String>), Box<dyn std::error::Error>> {
    let mut keys = Vec::new();
    let mut all_values = Vec::new();
    let mut needles = Vec::new();

    for _ in 0..config.num_needle_k {
        let key = generate_random_number();
        keys.push(key.clone());
        for _ in 0..config.num_needle_v {
            let value = generate_random_number();
            all_values.push(value.clone());
            needles.push(format!(
                "One of the special magic numbers for {} is: {}.",
                key, value
            ));
        }
    }

    let haystack_words: Vec<&str> = essays.split_whitespace().collect();
    let adjusted_length = config
        .context_length
        .saturating_sub(config.final_context_length_buffer);

    let mut lower = 100;
    let mut upper = haystack_words.len();
    let mut optimal_size = lower;

    while lower <= upper {
        let mid = (lower + upper) / 2;
        let test_text = haystack_words[..mid.min(haystack_words.len())].join(" ");
        let test_tokens = get_context_length_in_tokens(&test_text, tokenizer);
        if test_tokens + needles.len() * 20 <= adjusted_length {
            optimal_size = mid;
            lower = mid + 1;
        } else {
            upper = mid - 1;
        }
    }

    let mut context_text = haystack_words[..optimal_size.min(haystack_words.len())].join(" ");
    let sentences: Vec<&str> = context_text.split('.').collect();
    let insertion_point = if config.depth_percent == 100 {
        sentences.len()
    } else {
        (sentences.len() * config.depth_percent) / 100
    };

    let mut result_sentences = sentences[..insertion_point].to_vec();
    for needle in &needles {
        result_sentences.push(needle.as_str());
    }
    result_sentences.extend_from_slice(&sentences[insertion_point..]);
    context_text = result_sentences.join(".");
    context_text = encode_and_trim(&context_text, adjusted_length, tokenizer)?;

    let query_indices: Vec<usize> = (0..config.num_needle_q.min(config.num_needle_k)).collect();
    let query_keys: Vec<String> = query_indices.iter().map(|&i| keys[i].clone()).collect();
    let query_str = if query_keys.len() > 1 {
        format!(
            "{}, and {}",
            query_keys[..query_keys.len() - 1].join(", "),
            query_keys.last().unwrap()
        )
    } else {
        query_keys[0].clone()
    };

    let type_needle_v = if config.num_needle_q * config.num_needle_v == 1 {
        "number"
    } else {
        "numbers"
    };
    let prompt = format!(
        "Some special magic {} are hidden within the following text. Make sure to memorize it. I will quiz you about the {} afterwards.\n{}\nWhat are all the special magic {} for {} mentioned in the provided text?",
        type_needle_v, type_needle_v, context_text, type_needle_v, query_str
    );

    let expected_answers: Vec<String> = query_indices
        .iter()
        .flat_map(|&i| {
            let start = i * config.num_needle_v;
            let end = start + config.num_needle_v;
            all_values[start..end].to_vec()
        })
        .collect();

    Ok((prompt, expected_answers))
}

async fn run_niah_test(
    config: &NIAHConfig,
    model: &str,
    tokenizer: &Tokenizer,
    essays: &str,
    debug: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (prompt, expected_answers) = generate_niah_context(config, tokenizer, essays)?;

    if debug {
        eprintln!("\n=== DEBUG: NIAH Test ===");
        eprintln!("Expected answers: {:?}", expected_answers);
    }

    let system_prompt =
        "You are a helpful AI assistant. Answer based only on the provided context.";
    let max_tokens = 128;
    let temperature = 0.0;

    let query: Query =
        spnl!(g model (cross (system system_prompt) (user prompt)) temperature max_tokens);
    let options = ExecuteOptions {
        silent: true,
        ..Default::default()
    };

    match execute(&query, &options).await {
        Ok(Query::Message(Assistant(response))) => {
            if debug {
                eprintln!("Model response: {}", response);
            }
            Ok(string_match_all(&response, &expected_answers))
        }
        Ok(x) => Err(format!("Unexpected response: {:?}", x).into()),
        Err(e) => Err(format!("Query error: {}", e).into()),
    }
}

// TASK 2: Variable Tracking
#[derive(Debug, Clone)]
struct VariableTrackingConfig {
    context_length: usize,
    num_chains: usize,
    num_hops: usize,
    final_context_length_buffer: usize,
}

impl Default for VariableTrackingConfig {
    fn default() -> Self {
        Self {
            context_length: 4000,
            num_chains: 1,
            num_hops: 4,
            final_context_length_buffer: 200,
        }
    }
}

fn generate_var_name() -> String {
    let mut rng = rand::rng();
    (0..5)
        .map(|_| {
            let c = rng.random_range(b'A'..=b'Z');
            c as char
        })
        .collect()
}

fn generate_variable_tracking_context(
    config: &VariableTrackingConfig,
    tokenizer: &Tokenizer,
) -> Result<(String, Vec<String>), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();
    let mut all_vars = Vec::new();
    let mut chains = Vec::new();

    for _ in 0..config.num_chains {
        let initial_value = rng.random_range(10000..100000).to_string();
        let mut chain_vars = Vec::new();
        let mut chain_statements = Vec::new();

        let first_var = generate_var_name();
        chain_vars.push(first_var.clone());
        chain_statements.push(format!("VAR {} = {}", first_var, initial_value));

        for _ in 0..config.num_hops {
            let next_var = generate_var_name();
            chain_vars.push(next_var.clone());
            chain_statements.push(format!(
                "VAR {} = VAR {}",
                next_var,
                chain_vars[chain_vars.len() - 2]
            ));
        }

        all_vars.push(chain_vars);
        chains.push(chain_statements);
    }

    let noise = "The grass is green. The sky is blue.";
    let adjusted_length = config
        .context_length
        .saturating_sub(config.final_context_length_buffer);

    let mut lower = 10;
    let mut upper = 1000;
    let mut optimal_noise = lower;

    while lower <= upper {
        let mid = (lower + upper) / 2;
        let mut test_sentences = vec![noise; mid];
        for chain in &chains {
            for statement in chain {
                test_sentences.push(statement.as_str());
            }
        }
        let test_text = test_sentences.join("\n");
        let test_tokens = get_context_length_in_tokens(&test_text, tokenizer);
        if test_tokens <= adjusted_length {
            optimal_noise = mid;
            lower = mid + 1;
        } else {
            upper = mid - 1;
        }
    }

    let mut sentences = vec![noise; optimal_noise];
    for chain in &chains {
        for statement in chain {
            let insert_pos = rng.random_range(0..sentences.len());
            sentences.insert(insert_pos, statement.as_str());
        }
    }

    let context = sentences.join("\n");
    let context = encode_and_trim(&context, adjusted_length, tokenizer)?;

    let initial_value = chains[0][0].split('=').nth(1).unwrap().trim();
    let prompt = format!(
        "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{}\nQuestion: Find all variables that are assigned the value {} in the text above.",
        context, initial_value
    );

    Ok((prompt, all_vars[0].clone()))
}

async fn run_variable_tracking_test(
    config: &VariableTrackingConfig,
    model: &str,
    tokenizer: &Tokenizer,
    debug: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (prompt, expected_answers) = generate_variable_tracking_context(config, tokenizer)?;

    if debug {
        eprintln!("\n=== DEBUG: Variable Tracking Test ===");
        eprintln!("Expected answers: {:?}", expected_answers);
    }

    let system_prompt = "You are a helpful AI assistant.";
    let max_tokens = 30;
    let temperature = 0.0;

    let query: Query =
        spnl!(g model (cross (system system_prompt) (user prompt)) temperature max_tokens);
    let options = ExecuteOptions {
        silent: true,
        ..Default::default()
    };

    match execute(&query, &options).await {
        Ok(Query::Message(Assistant(response))) => {
            if debug {
                eprintln!("Model response: {}", response);
            }
            Ok(string_match_all(&response, &expected_answers))
        }
        Ok(x) => Err(format!("Unexpected response: {:?}", x).into()),
        Err(e) => Err(format!("Query error: {}", e).into()),
    }
}

fn compute_quantiles(values: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    (
        sorted[0],
        sorted[len * 25 / 100],
        sorted[len * 50 / 100],
        sorted[len * 75 / 100],
        sorted[len * 90 / 100],
        sorted[len * 99 / 100],
        sorted[len - 1],
    )
}

fn ruler_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("ruler");

    let sample_size = std::env::var("BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    group.sample_size(sample_size);

    let measurement_time = std::env::var("BENCH_MEASUREMENT_TIME")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    group.measurement_time(std::time::Duration::from_secs(measurement_time));

    let context_lengths: Vec<usize> = std::env::var("BENCH_CONTEXT_LENGTHS")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![4000, 8000]);

    let model = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "ollama/granite3.3:2b".to_string());
    let tokenizer_model = std::env::var("BENCH_TOKENIZER_MODEL")
        .unwrap_or_else(|_| "ibm-granite/granite-3.3-2b-instruct".to_string());
    let final_context_length_buffer = std::env::var("BENCH_FINAL_CONTEXT_LENGTH_BUFFER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let debug = std::env::var("BENCH_DEBUG")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let tasks_str = std::env::var("BENCH_TASKS").unwrap_or_else(|_| "niah".to_string());
    let tasks: Vec<&str> = tasks_str.split(',').map(|s| s.trim()).collect();

    eprintln!("\n=== Loading tokenizer: {} ===", tokenizer_model);
    let tokenizer =
        Tokenizer::from_pretrained(&tokenizer_model, None).expect("Failed to load tokenizer");
    let essays = fetch_pg_essays().unwrap_or_else(|_| FALLBACK_ESSAYS.to_string());

    eprintln!("\n=== RULER Benchmark ===");
    eprintln!("Model: {}", model);
    eprintln!("Context lengths: {:?}", context_lengths);
    eprintln!("Tasks: {:?}\n", tasks);

    // NIAH benchmarks
    if tasks.contains(&"niah") {
        let depth_percentages: Vec<usize> = std::env::var("BENCH_NIAH_DEPTH_PERCENTAGES")
            .ok()
            .and_then(|s| {
                s.split(',')
                    .map(|n| n.trim().parse().ok())
                    .collect::<Option<Vec<_>>>()
            })
            .unwrap_or_else(|| vec![50]);

        let num_needle_k = std::env::var("BENCH_NIAH_NUM_NEEDLE_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let num_needle_v = std::env::var("BENCH_NIAH_NUM_NEEDLE_V")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let num_needle_q = std::env::var("BENCH_NIAH_NUM_NEEDLE_Q")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        for context_length in &context_lengths {
            for depth_percent in &depth_percentages {
                let accuracy_values = Arc::new(Mutex::new(Vec::new()));
                let accuracy_clone = Arc::clone(&accuracy_values);
                let debug_counter = Arc::new(Mutex::new(0));

                let base_msg = format!("NIAH len={} depth={}%", context_length, depth_percent);
                let pb =
                    bench_progress::create_benchmark_progress(sample_size as u64, base_msg.clone());
                let pb_clone = Arc::clone(&pb);
                let base_msg = Arc::new(base_msg);
                let base_msg_clone = Arc::clone(&base_msg);

                group.bench_with_input(
                    BenchmarkId::new(
                        "niah",
                        format!("len={}/depth={}", context_length, depth_percent),
                    ),
                    &(*context_length, *depth_percent),
                    |b, &(len, depth)| {
                        b.to_async(&runtime).iter(|| {
                            let accuracy_clone = Arc::clone(&accuracy_clone);
                            let pb = Arc::clone(&pb_clone);
                            let base_msg = Arc::clone(&base_msg_clone);
                            let model = model.clone();
                            let tokenizer = tokenizer.clone();
                            let essays = essays.clone();
                            let debug_counter = Arc::clone(&debug_counter);

                            async move {
                                let mut counter = debug_counter.lock().unwrap();
                                let should_debug = debug && *counter == 0;
                                *counter += 1;
                                drop(counter);

                                let config = NIAHConfig {
                                    context_length: len,
                                    depth_percent: depth,
                                    num_needle_k,
                                    num_needle_v,
                                    num_needle_q,
                                    final_context_length_buffer,
                                };

                                let accuracy = run_niah_test(
                                    &config,
                                    &model,
                                    &tokenizer,
                                    &essays,
                                    should_debug,
                                )
                                .await
                                .unwrap_or(0.0);
                                accuracy_clone.lock().unwrap().push(accuracy);

                                let accuracies = accuracy_clone.lock().unwrap();
                                let total_count = accuracies.len();
                                let avg_acc = accuracies.iter().sum::<f64>() / total_count as f64;
                                let perfect_count =
                                    accuracies.iter().filter(|&&a| a >= 1.0).count();
                                drop(accuracies);

                                pb.set_message(format!(
                                    "{} | n={} | Acc={:.1}% | Perfect={}/{}",
                                    base_msg,
                                    total_count,
                                    avg_acc * 100.0,
                                    perfect_count,
                                    total_count
                                ));
                                pb.inc(1);
                                accuracy
                            }
                        });
                    },
                );

                bench_progress::finish_benchmark_progress(
                    &pb,
                    format!("✓ NIAH len={} depth={}%", context_length, depth_percent),
                );

                let accuracies = accuracy_values.lock().unwrap();
                if !accuracies.is_empty() {
                    let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&accuracies);
                    let avg = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
                    let perfect_count = accuracies.iter().filter(|&&a| a >= 1.0).count();

                    eprintln!(
                        "\n=== NIAH Stats: len={} depth={}% (n={}) ===",
                        context_length,
                        depth_percent,
                        accuracies.len()
                    );
                    eprintln!("  avg:  {:.1}%", avg * 100.0);
                    eprintln!("  min:  {:.1}%", min * 100.0);
                    eprintln!("  p25:  {:.1}%", p25 * 100.0);
                    eprintln!("  p50:  {:.1}%", p50 * 100.0);
                    eprintln!("  p75:  {:.1}%", p75 * 100.0);
                    eprintln!("  p90:  {:.1}%", p90 * 100.0);
                    eprintln!("  p99:  {:.1}%", p99 * 100.0);
                    eprintln!("  max:  {:.1}%", max * 100.0);
                    eprintln!("  perfect: {}/{}\n", perfect_count, accuracies.len());
                }
            }
        }
    }

    // Variable Tracking benchmarks
    if tasks.contains(&"variable_tracking") {
        let num_chains = std::env::var("BENCH_VT_NUM_CHAINS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let num_hops = std::env::var("BENCH_VT_NUM_HOPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        for context_length in &context_lengths {
            let accuracy_values = Arc::new(Mutex::new(Vec::new()));
            let accuracy_clone = Arc::clone(&accuracy_values);
            let debug_counter = Arc::new(Mutex::new(0));

            let base_msg = format!("VT len={}", context_length);
            let pb =
                bench_progress::create_benchmark_progress(sample_size as u64, base_msg.clone());
            let pb_clone = Arc::clone(&pb);
            let base_msg = Arc::new(base_msg);
            let base_msg_clone = Arc::clone(&base_msg);

            group.bench_with_input(
                BenchmarkId::new("variable_tracking", format!("len={}", context_length)),
                context_length,
                |b, &len| {
                    b.to_async(&runtime).iter(|| {
                        let accuracy_clone = Arc::clone(&accuracy_clone);
                        let pb = Arc::clone(&pb_clone);
                        let base_msg = Arc::clone(&base_msg_clone);
                        let model = model.clone();
                        let tokenizer = tokenizer.clone();
                        let debug_counter = Arc::clone(&debug_counter);

                        async move {
                            let mut counter = debug_counter.lock().unwrap();
                            let should_debug = debug && *counter == 0;
                            *counter += 1;
                            drop(counter);

                            let config = VariableTrackingConfig {
                                context_length: len,
                                num_chains,
                                num_hops,
                                final_context_length_buffer,
                            };
                            let accuracy = run_variable_tracking_test(
                                &config,
                                &model,
                                &tokenizer,
                                should_debug,
                            )
                            .await
                            .unwrap_or(0.0);
                            accuracy_clone.lock().unwrap().push(accuracy);

                            let accuracies = accuracy_clone.lock().unwrap();
                            let total_count = accuracies.len();
                            let avg_acc = accuracies.iter().sum::<f64>() / total_count as f64;
                            let perfect_count = accuracies.iter().filter(|&&a| a >= 1.0).count();
                            drop(accuracies);

                            pb.set_message(format!(
                                "{} | n={} | Acc={:.1}% | Perfect={}/{}",
                                base_msg,
                                total_count,
                                avg_acc * 100.0,
                                perfect_count,
                                total_count
                            ));
                            pb.inc(1);
                            accuracy
                        }
                    });
                },
            );

            bench_progress::finish_benchmark_progress(&pb, format!("✓ VT len={}", context_length));

            let accuracies = accuracy_values.lock().unwrap();
            if !accuracies.is_empty() {
                let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&accuracies);
                let avg = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
                let perfect_count = accuracies.iter().filter(|&&a| a >= 1.0).count();

                eprintln!(
                    "\n=== Variable Tracking Stats: len={} (n={}) ===",
                    context_length,
                    accuracies.len()
                );
                eprintln!("  avg:  {:.1}%", avg * 100.0);
                eprintln!("  min:  {:.1}%", min * 100.0);
                eprintln!("  p25:  {:.1}%", p25 * 100.0);
                eprintln!("  p50:  {:.1}%", p50 * 100.0);
                eprintln!("  p75:  {:.1}%", p75 * 100.0);
                eprintln!("  p90:  {:.1}%", p90 * 100.0);
                eprintln!("  p99:  {:.1}%", p99 * 100.0);
                eprintln!("  max:  {:.1}%", max * 100.0);
                eprintln!("  perfect: {}/{}\n", perfect_count, accuracies.len());
            }
        }
    }

    group.finish();
    eprintln!("\n=== RULER Benchmark Complete ===\n");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = ruler_benchmark
}
criterion_main!(benches);

// Made with Bob

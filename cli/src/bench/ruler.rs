use super::{
    FALLBACK_ESSAYS, compute_quantiles, create_benchmark_progress, encode_and_trim,
    fetch_pg_essays, finish_benchmark_progress, get_context_length_in_tokens,
};
use criterion::{BenchmarkId, Criterion};
use rand::Rng;
use spnl::{
    ExecuteOptions, SpnlError, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct RulerArgs {
    /// Generative model
    #[arg(
        short,
        long,
        default_value = "ollama/granite3.3:2b",
        env = "BENCH_MODEL"
    )]
    pub model: String,

    /// HuggingFace tokenizer model
    #[arg(
        long,
        default_value = "ibm-granite/granite-3.3-2b-instruct",
        env = "BENCH_TOKENIZER_MODEL"
    )]
    pub tokenizer_model: String,

    /// Sample size per configuration
    #[arg(long, default_value_t = 10, env = "BENCH_SAMPLE_SIZE")]
    pub sample_size: usize,

    /// Measurement time in seconds
    #[arg(long, default_value_t = 60, env = "BENCH_MEASUREMENT_TIME")]
    pub measurement_time: u64,

    /// Comma-separated context lengths in tokens
    #[arg(long, default_value = "4000,8000", value_parser = super::parse_csv_usize, env = "BENCH_CONTEXT_LENGTHS")]
    pub context_lengths: Vec<usize>,

    /// Comma-separated tasks to run (niah, variable_tracking)
    #[arg(long, default_value = "niah", env = "BENCH_TASKS")]
    pub tasks: String,

    /// Token buffer for system/question/response
    #[arg(long, default_value_t = 200, env = "BENCH_FINAL_CONTEXT_LENGTH_BUFFER")]
    pub context_length_buffer: usize,

    /// Enable debug output for first sample
    #[arg(long)]
    pub debug: bool,

    // -- NIAH-specific --
    /// Number of needles (keys) to insert
    #[arg(long, default_value_t = 1, env = "BENCH_NIAH_NUM_NEEDLE_K")]
    pub niah_num_needle_k: usize,

    /// Number of values per needle
    #[arg(long, default_value_t = 1, env = "BENCH_NIAH_NUM_NEEDLE_V")]
    pub niah_num_needle_v: usize,

    /// Number of needles to query
    #[arg(long, default_value_t = 1, env = "BENCH_NIAH_NUM_NEEDLE_Q")]
    pub niah_num_needle_q: usize,

    /// Comma-separated depth percentages for NIAH
    #[arg(long, default_value = "50", value_parser = super::parse_csv_usize, env = "BENCH_NIAH_DEPTH_PERCENTAGES")]
    pub niah_depth_percentages: Vec<usize>,

    // -- Variable Tracking-specific --
    /// Number of variable chains
    #[arg(long, default_value_t = 1, env = "BENCH_VT_NUM_CHAINS")]
    pub vt_num_chains: usize,

    /// Number of hops per chain
    #[arg(long, default_value_t = 4, env = "BENCH_VT_NUM_HOPS")]
    pub vt_num_hops: usize,
}

// ---------------------------------------------------------------------------
// Evaluation metrics
// ---------------------------------------------------------------------------

fn string_match_all(prediction: &str, references: &[String]) -> f64 {
    let pred_lower = prediction.to_lowercase();
    let matches: usize = references
        .iter()
        .filter(|r| pred_lower.contains(&r.to_lowercase()))
        .count();
    (matches as f64) / (references.len() as f64)
}

// ---------------------------------------------------------------------------
// NIAH task
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct NIAHConfig {
    context_length: usize,
    depth_percent: usize,
    num_needle_k: usize,
    num_needle_v: usize,
    num_needle_q: usize,
    final_context_length_buffer: usize,
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

// ---------------------------------------------------------------------------
// Variable Tracking task
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VariableTrackingConfig {
    context_length: usize,
    num_chains: usize,
    num_hops: usize,
    final_context_length_buffer: usize,
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

// ---------------------------------------------------------------------------
// Criterion benchmark function
// ---------------------------------------------------------------------------

fn ruler_benchmark(c: &mut Criterion, args: &RulerArgs) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("ruler");

    group.sample_size(args.sample_size);
    group.measurement_time(std::time::Duration::from_secs(args.measurement_time));

    let model = &args.model;
    let final_context_length_buffer = args.context_length_buffer;
    let debug = args.debug;
    let tasks: Vec<&str> = args.tasks.split(',').map(|s| s.trim()).collect();

    eprintln!("\n=== Loading tokenizer: {} ===", args.tokenizer_model);
    let tokenizer =
        Tokenizer::from_pretrained(&args.tokenizer_model, None).expect("Failed to load tokenizer");
    let essays = fetch_pg_essays("ruler").unwrap_or_else(|_| FALLBACK_ESSAYS.to_string());

    eprintln!("\n=== RULER Benchmark ===");
    eprintln!("Model: {}", model);
    eprintln!("Context lengths: {:?}", args.context_lengths);
    eprintln!("Tasks: {:?}\n", tasks);

    // NIAH benchmarks
    if tasks.contains(&"niah") {
        let num_needle_k = args.niah_num_needle_k;
        let num_needle_v = args.niah_num_needle_v;
        let num_needle_q = args.niah_num_needle_q;

        for &context_length in &args.context_lengths {
            for &depth_percent in &args.niah_depth_percentages {
                let accuracy_values = Arc::new(Mutex::new(Vec::new()));
                let accuracy_clone = Arc::clone(&accuracy_values);
                let debug_counter = Arc::new(Mutex::new(0));

                let base_msg = format!("NIAH len={} depth={}%", context_length, depth_percent);
                let pb = create_benchmark_progress(args.sample_size as u64, base_msg.clone());
                let pb_clone = Arc::clone(&pb);
                let base_msg = Arc::new(base_msg);
                let base_msg_clone = Arc::clone(&base_msg);

                group.bench_with_input(
                    BenchmarkId::new(
                        "niah",
                        format!("len={}/depth={}", context_length, depth_percent),
                    ),
                    &(context_length, depth_percent),
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
                                let should_debug = {
                                    let mut counter = debug_counter.lock().unwrap();
                                    let v = debug && *counter == 0;
                                    *counter += 1;
                                    v
                                };

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

                finish_benchmark_progress(
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
        let num_chains = args.vt_num_chains;
        let num_hops = args.vt_num_hops;

        for &context_length in &args.context_lengths {
            let accuracy_values = Arc::new(Mutex::new(Vec::new()));
            let accuracy_clone = Arc::clone(&accuracy_values);
            let debug_counter = Arc::new(Mutex::new(0));

            let base_msg = format!("VT len={}", context_length);
            let pb = create_benchmark_progress(args.sample_size as u64, base_msg.clone());
            let pb_clone = Arc::clone(&pb);
            let base_msg = Arc::new(base_msg);
            let base_msg_clone = Arc::clone(&base_msg);

            group.bench_with_input(
                BenchmarkId::new("variable_tracking", format!("len={}", context_length)),
                &context_length,
                |b, &len| {
                    b.to_async(&runtime).iter(|| {
                        let accuracy_clone = Arc::clone(&accuracy_clone);
                        let pb = Arc::clone(&pb_clone);
                        let base_msg = Arc::clone(&base_msg_clone);
                        let model = model.clone();
                        let tokenizer = tokenizer.clone();
                        let debug_counter = Arc::clone(&debug_counter);

                        async move {
                            let should_debug = {
                                let mut counter = debug_counter.lock().unwrap();
                                let v = debug && *counter == 0;
                                *counter += 1;
                                v
                            };

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

            finish_benchmark_progress(&pb, format!("✓ VT len={}", context_length));

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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: RulerArgs) -> Result<(), SpnlError> {
    let mut criterion = Criterion::default();
    ruler_benchmark(&mut criterion, &args);
    criterion.final_summary();
    Ok(())
}

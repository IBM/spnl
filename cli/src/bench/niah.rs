use super::{
    FALLBACK_ESSAYS, compute_quantiles, create_benchmark_progress, encode_and_trim,
    fetch_pg_essays, finish_benchmark_progress, get_context_length_in_tokens,
};
use criterion::{BenchmarkId, Criterion};
use spnl::{
    ExecuteOptions, SpnlError, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct NiahArgs {
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
    #[arg(
        long,
        default_value = "1000,2000,4000,8000",
        value_delimiter = ',',
        env = "BENCH_CONTEXT_LENGTHS"
    )]
    pub context_lengths: Vec<usize>,

    /// Comma-separated depth percentages (0-100)
    #[arg(
        long,
        default_value = "0,25,50,75,100",
        value_delimiter = ',',
        env = "BENCH_DEPTH_PERCENTAGES"
    )]
    pub depth_percentages: Vec<usize>,

    /// Comma-separated chunk sizes (0 = no chunking)
    #[arg(
        long,
        default_value = "0,2,4",
        value_delimiter = ',',
        env = "BENCH_CHUNK_SIZES"
    )]
    pub chunk_sizes: Vec<usize>,

    /// Token buffer for system/question/response
    #[arg(long, default_value_t = 200, env = "BENCH_FINAL_CONTEXT_LENGTH_BUFFER")]
    pub context_length_buffer: usize,

    /// Enable debug output for first sample
    #[arg(long)]
    pub debug: bool,
}

// ---------------------------------------------------------------------------
// Needle config and context generation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct NeedleConfig {
    context_length: usize,
    depth_percent: usize,
    needle: String,
    question: String,
    expected_answer: String,
    final_context_length_buffer: usize,
}

impl Default for NeedleConfig {
    fn default() -> Self {
        Self {
            context_length: 2000,
            depth_percent: 50,
            needle: "\nThe special magic number mentioned in the context is 73.\n".to_string(),
            question: "What is the special magic number mentioned in the context?".to_string(),
            expected_answer: "73".to_string(),
            final_context_length_buffer: 200,
        }
    }
}

fn read_context_files(
    max_context_length: usize,
    tokenizer: &Tokenizer,
) -> Result<String, Box<dyn std::error::Error>> {
    let essays_text = fetch_pg_essays("niah").unwrap_or_else(|e| {
        eprintln!("Warning: Failed to fetch Paul Graham essays: {}", e);
        eprintln!("Using fallback essays...");
        FALLBACK_ESSAYS.to_string()
    });

    let mut context = String::new();
    while get_context_length_in_tokens(&context, tokenizer) < max_context_length {
        context.push_str(&essays_text);
        context.push(' ');
    }

    Ok(context)
}

fn insert_needle(
    context: &str,
    needle: &str,
    depth_percent: usize,
    context_length: usize,
    final_context_length_buffer: usize,
    tokenizer: &Tokenizer,
) -> Result<String, Box<dyn std::error::Error>> {
    let tokens_needle = tokenizer
        .encode(needle, false)
        .map_err(|e| format!("Encoding needle error: {}", e))?;
    let needle_tokens = tokens_needle.get_ids();

    let tokens_context = tokenizer
        .encode(context, false)
        .map_err(|e| format!("Encoding context error: {}", e))?;
    let mut context_tokens = tokens_context.get_ids().to_vec();

    let adjusted_context_length = context_length.saturating_sub(final_context_length_buffer);

    if context_tokens.len() + needle_tokens.len() > adjusted_context_length {
        context_tokens.truncate(adjusted_context_length.saturating_sub(needle_tokens.len()));
    }

    let new_context_tokens = if depth_percent == 100 {
        [context_tokens.as_slice(), needle_tokens].concat()
    } else {
        let mut insertion_point = (context_tokens.len() * depth_percent) / 100;

        let period_tokens = tokenizer
            .encode(".", false)
            .map_err(|e| format!("Encoding period error: {}", e))?;
        let period_token_ids = period_tokens.get_ids();

        while insertion_point > 0 {
            let tokens_before = &context_tokens[..insertion_point];
            if tokens_before.is_empty() {
                break;
            }
            if period_token_ids.contains(&tokens_before[tokens_before.len() - 1]) {
                break;
            }
            insertion_point -= 1;
        }

        [
            &context_tokens[..insertion_point],
            needle_tokens,
            &context_tokens[insertion_point..],
        ]
        .concat()
    };

    Ok(tokenizer
        .decode(&new_context_tokens, false)
        .map_err(|e| format!("Decoding error: {}", e))?)
}

fn generate_context(
    config: &NeedleConfig,
    tokenizer: &Tokenizer,
    max_context_length: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let context = read_context_files(max_context_length, tokenizer)?;
    let context = encode_and_trim(&context, config.context_length, tokenizer)?;
    insert_needle(
        &context,
        &config.needle,
        config.depth_percent,
        config.context_length,
        config.final_context_length_buffer,
        tokenizer,
    )
}

fn evaluate_needle_retrieval(response: &str, expected_answer: &str, debug: bool) -> f64 {
    let response_lower = response.to_lowercase();
    let expected_lower = expected_answer.to_lowercase();

    if debug {
        eprintln!("\n=== DEBUG: Needle Retrieval ===");
        eprintln!("Expected answer: {}", expected_answer);
        eprintln!("Model response: {}", response);
        eprintln!(
            "Response contains expected? {}",
            response_lower.contains(&expected_lower)
        );
    }

    if response_lower.contains(&expected_lower) {
        return 1.0;
    }

    if let Ok(expected_num) = expected_answer.parse::<i32>() {
        for word in response.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| !c.is_numeric());
            if let Ok(num) = cleaned.parse::<i32>()
                && num == expected_num
            {
                if debug {
                    eprintln!("Found number match: {}", num);
                }
                return 1.0;
            }
        }
    }

    if debug {
        eprintln!("=== No match found ===\n");
    }

    0.0
}

// ---------------------------------------------------------------------------
// Core test
// ---------------------------------------------------------------------------

async fn run_niah_test(
    config: &NeedleConfig,
    model: &str,
    temperature: f32,
    tokenizer: &Tokenizer,
    max_context_length: usize,
    debug: bool,
    chunk: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let context_with_needle = generate_context(config, tokenizer, max_context_length)?;

    if debug {
        eprintln!("\n=== DEBUG: Context Generation ===");
        eprintln!("Context length (chars): {}", context_with_needle.len());
        eprintln!(
            "Context length (tokens): {}",
            get_context_length_in_tokens(&context_with_needle, tokenizer)
        );
        eprintln!("Needle: {}", config.needle);
        eprintln!("Question: {}", config.question);
        eprintln!(
            "Context preview (first 500 chars): {}...",
            &context_with_needle[..500.min(context_with_needle.len())]
        );
        eprintln!(
            "Context preview (last 200 chars): ...{}",
            &context_with_needle[context_with_needle.len().saturating_sub(200)..]
        );
        eprintln!(
            "Context contains needle? {}",
            context_with_needle.contains(&config.needle)
        );
        if let Some(pos) = context_with_needle.find(&config.needle) {
            eprintln!("Needle found at character position: {}", pos);
            eprintln!(
                "Needle position as % of context: {:.1}%",
                (pos as f64 / context_with_needle.len() as f64) * 100.0
            );
        }
    }

    let system_prompt = "You are a helpful AI assistant. Answer the question based only on the information provided in the context. Be concise and direct.";
    let question = &config.question;
    let max_tokens = 300;

    let query: Query = if chunk > 0 {
        let encoding = tokenizer
            .encode(context_with_needle.as_str(), false)
            .map_err(|e| format!("Encoding error: {}", e))?;
        let tokens = encoding.get_ids();
        let chunk_size_tokens = tokens.len().div_ceil(chunk);

        let chunks: Vec<Query> = tokens
            .chunks(chunk_size_tokens)
            .map(|chunk_tokens| tokenizer.decode(chunk_tokens, false).unwrap_or_default())
            .map(|chunk_text| {
                spnl!(
                    g model
                        (cross
                            (system system_prompt)
                            (user chunk_text)
                            (user question)
                        )
                        temperature
                        max_tokens
                )
            })
            .collect();

        if chunks.len() == 1 {
            chunks[0].clone()
        } else {
            spnl!(
                g model
                    (cross
                        (system system_prompt)
                        (plus chunks)
                        (user "Based on the above responses, what is the final answer to the question? Be concise and direct.")
                    )
                    temperature
                    max_tokens
            )
        }
    } else {
        spnl!(
            g model
                (cross
                    (system system_prompt)
                    (user context_with_needle)
                    (user question)
                )
                temperature
                max_tokens
        )
    };

    let options = ExecuteOptions {
        silent: true,
        ..Default::default()
    };

    if debug {
        eprintln!("=== Executing query... ===");
        eprintln!("=== Query structure: ===");
        eprintln!("{:#?}", query);
    }

    match execute(&query, &options).await {
        Ok(Query::Message(Assistant(response))) => {
            if debug {
                eprintln!("=== Got response from model ===");
            }
            let score = evaluate_needle_retrieval(&response, &config.expected_answer, debug);
            Ok(score)
        }
        Ok(x) => {
            if debug {
                eprintln!("=== ERROR: Unexpected non-string response: {:?} ===", x);
            }
            Err(format!("Unexpected non-string response: {:?}", x).into())
        }
        Err(e) => {
            if debug {
                eprintln!("=== ERROR executing query: {} ===", e);
            }
            Err(format!("Query execution error: {}", e).into())
        }
    }
}

// ---------------------------------------------------------------------------
// Criterion benchmark function
// ---------------------------------------------------------------------------

fn niah_benchmark(c: &mut Criterion, args: &NiahArgs) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("needle_in_haystack");

    group.sample_size(args.sample_size);
    group.measurement_time(std::time::Duration::from_secs(args.measurement_time));

    let model = &args.model;
    let final_context_length_buffer = args.context_length_buffer;
    let temperature = 0.0;
    let debug = args.debug;
    let debug_counter = Arc::new(Mutex::new(0));

    eprintln!("\n=== Loading tokenizer: {} ===", args.tokenizer_model);
    eprintln!("=== Using model for inference: {} ===", model);
    let tokenizer =
        Tokenizer::from_pretrained(&args.tokenizer_model, None).expect("Failed to load tokenizer");
    let max_context_length = *args.context_lengths.iter().max().unwrap_or(&8000);

    eprintln!("\n=== Needle In A Haystack Benchmark ===");
    eprintln!("Model: {}", model);
    eprintln!("Context lengths (tokens): {:?}", args.context_lengths);
    eprintln!("Depth percentages: {:?}", args.depth_percentages);
    eprintln!("Chunk sizes: {:?}", args.chunk_sizes);
    eprintln!("Sample size: {}", args.sample_size);
    eprintln!("Temperature: {}", temperature);
    eprintln!(
        "Final context length buffer: {}\n",
        final_context_length_buffer
    );

    for &chunk_size in &args.chunk_sizes {
        for &context_length in &args.context_lengths {
            for &depth_percent in &args.depth_percentages {
                let accuracy_values = Arc::new(Mutex::new(Vec::new()));
                let accuracy_clone = Arc::clone(&accuracy_values);

                let base_msg = format!(
                    "chunk={} len={} depth={}%",
                    chunk_size, context_length, depth_percent
                );
                let pb = create_benchmark_progress(args.sample_size as u64, base_msg.clone());
                let pb_clone = Arc::clone(&pb);
                let base_msg = Arc::new(base_msg);
                let base_msg_clone = Arc::clone(&base_msg);

                let bench_id = format!(
                    "chunk={}/len={}/depth={}",
                    chunk_size, context_length, depth_percent
                );

                group.bench_with_input(
                    BenchmarkId::new("retrieval", bench_id),
                    &(context_length, depth_percent, chunk_size),
                    |b, &(len, depth, chunk)| {
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
                                let config = NeedleConfig {
                                    context_length: len,
                                    depth_percent: depth,
                                    final_context_length_buffer,
                                    ..Default::default()
                                };

                                let accuracy = run_niah_test(
                                    &config,
                                    &model,
                                    temperature,
                                    &tokenizer,
                                    max_context_length,
                                    should_debug,
                                    chunk,
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

                let finish_msg = format!(
                    "✓ chunk={} len={} depth={}%",
                    chunk_size, context_length, depth_percent
                );
                finish_benchmark_progress(&pb, finish_msg);

                let accuracies = accuracy_values.lock().unwrap();
                if !accuracies.is_empty() {
                    let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&accuracies);
                    let avg = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
                    let perfect_count = accuracies.iter().filter(|&&a| a >= 1.0).count();

                    eprintln!(
                        "\n=== Accuracy Stats: chunk={} len={} depth={}% (n={}) ===",
                        chunk_size,
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

    group.finish();
    eprintln!("\n=== Benchmark Complete ===\n");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: NiahArgs) -> Result<(), SpnlError> {
    let mut criterion = Criterion::default();
    niah_benchmark(&mut criterion, &args);
    criterion.final_summary();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_exact_substring_match() {
        assert!(
            (evaluate_needle_retrieval("The answer is 73.", "73", false) - 1.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_case_insensitive_match() {
        assert!(
            (evaluate_needle_retrieval("HELLO world", "hello", false) - 1.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_numeric_match_in_tokens() {
        // Number not as substring but parsed from a token
        assert!(
            (evaluate_needle_retrieval("The number is (73).", "73", false) - 1.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_no_match_returns_zero() {
        assert!(
            (evaluate_needle_retrieval("Nothing relevant here", "73", false)).abs() < f64::EPSILON
        );
    }
}

//! Needle In A Haystack (NIAH) Benchmark
//!
//! This benchmark is a faithful Rust port of the LLMTest_NeedleInAHaystack test methodology
//! (https://github.com/gkamradt/LLMTest_NeedleInAHaystack) adapted to use SPNL span queries.
//!
//! The test evaluates an LLM's ability to retrieve specific information ("needle")
//! from within a large context ("haystack") by:
//! 1. Loading Paul Graham essays as context
//! 2. Encoding to tokens and trimming to exact context length
//! 3. Inserting needle at sentence boundaries at various depths (0-100%)
//! 4. Asking the model to retrieve the needle
//! 5. Measuring retrieval accuracy
//!
//! ## Faithful to Original Methodology
//!
//! This implementation matches the original Python code:
//! - Uses token-based context lengths (not word-based)
//! - Encodes context to tokens, trims to exact token count
//! - Inserts needle at sentence boundaries (backs up to find period token)
//! - Accounts for 200-token buffer for system/question/response
//! - Works in token space, then decodes back to text
//!
//! ## Configuration via Environment Variables
//!
//! - `BENCH_SAMPLE_SIZE`: Number of samples per configuration (default: 10)
//! - `BENCH_MEASUREMENT_TIME`: Measurement time in seconds (default: 60)
//! - `BENCH_CONTEXT_LENGTHS`: Comma-separated context lengths in TOKENS (default: "1000,2000,4000,8000")
//! - `BENCH_DEPTH_PERCENTAGES`: Comma-separated depth percentages (default: "0,25,50,75,100")
//! - `BENCH_CHUNK_SIZES`: Comma-separated chunk counts for map-reduce (default: "0,2,4")
//! - `BENCH_MODEL`: Model to use for inference (default: "ollama/granite3.3:2b")
//! - `BENCH_TOKENIZER_MODEL`: HuggingFace model for tokenizer (default: "ibm-granite/granite-3.3-2b-instruct")
//! - `BENCH_FINAL_CONTEXT_LENGTH_BUFFER`: Buffer for system/question/response (default: 200)
//! - `BENCH_DEBUG`: Enable debug output for first sample (default: false)
//!
//! ## Example Usage
//!
//! ```bash
//! # Run with default settings (requires granite3.3:2b in Ollama)
//! cargo bench --bench niah --features tok
//!
//! # Use a different Ollama model
//! BENCH_MODEL="ollama/llama3.2:3b" BENCH_TOKENIZER_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
//!   cargo bench --bench niah --features tok
//!
//! # Run with debug output
//! BENCH_DEBUG=1 cargo bench --bench niah --features tok
//!
//! # Custom configuration
//! BENCH_SAMPLE_SIZE=20 BENCH_CONTEXT_LENGTHS="2000,4000,8000" \
//!   cargo bench --bench niah --features tok
//!
//! # Test with map-reduce chunking (split context into 2 or 4 chunks)
//! BENCH_CHUNK_SIZES="2,4" cargo bench --bench niah --features tok
//!
//! # Filter to run only chunk=0 (non-chunked) benchmarks
//! cargo bench --bench niah --features tok -- "chunk=0"
//!
//! # Filter to run only chunk=2 benchmarks
//! cargo bench --bench niah --features tok -- "chunk=2"
//!
//! # Run only chunked benchmarks (chunk > 0)
//! cargo bench --bench niah --features tok -- "chunk=(2|4)"
//!
//! # Compare all chunk sizes for same configuration
//! cargo bench --bench niah --features tok -- "len=2000/depth=50"
//! ```

mod bench_progress;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
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

/// Get the cache directory for Paul Graham essays
fn get_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir()
        .ok_or("Could not determine cache directory")?
        .join("spnl")
        .join("niah");

    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

#[derive(serde::Deserialize)]
struct GitHubFile {
    name: String,
    download_url: String,
}

/// Fetch and cache the Paul Graham essays (all 49 essay files)
fn fetch_pg_essays() -> Result<String, Box<dyn std::error::Error>> {
    let cache_file = get_cache_dir()?.join("paul_graham_essays_combined.txt");

    // Check if cached file exists
    if cache_file.exists() {
        // eprintln!("Loading Paul Graham essays from cache: {}", cache_file.display());
        return Ok(fs::read_to_string(&cache_file)?);
    }

    // Download the essays
    eprintln!("Downloading Paul Graham essays from GitHub...");
    eprintln!("This is a one-time download (49 essay files, ~728KB total)...");

    // Get list of files from GitHub API
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(PG_ESSAYS_API_URL)
        .header("User-Agent", "spnl-niah-benchmark")
        .send()?;

    let files: Vec<GitHubFile> = response.json()?;

    // Filter for .txt files and download each one
    let mut combined_content = String::new();
    let txt_files: Vec<_> = files
        .into_iter()
        .filter(|f| f.name.ends_with(".txt"))
        .collect();

    eprintln!("Found {} essay files to download...", txt_files.len());

    for (i, file) in txt_files.iter().enumerate() {
        eprint!("\rDownloading essay {}/{}...", i + 1, txt_files.len());
        let essay_content = client
            .get(&file.download_url)
            .header("User-Agent", "spnl-niah-benchmark")
            .send()?
            .text()?;
        combined_content.push_str(&essay_content);
        combined_content.push('\n'); // Add separator between essays
    }
    eprintln!("\nDownload complete!");

    // Cache the combined content
    let mut cache_file_handle = fs::File::create(&cache_file)?;
    cache_file_handle.write_all(combined_content.as_bytes())?;

    eprintln!("Cached Paul Graham essays to: {}", cache_file.display());

    Ok(combined_content)
}

/// Fallback essays in case download fails
const FALLBACK_ESSAYS: &str = r#"The way to get startup ideas is not to try to think of startup ideas. It's to look for problems, preferably problems you have yourself. The very best startup ideas tend to have three things in common: they're something the founders themselves want, that they themselves can build, and that few others realize are worth doing. Microsoft, Apple, Yahoo, Google, and Facebook all began this way.

One of the biggest things holding people back from doing great work is the fear of making something lame. And this fear is not an irrational one. Many things that are new are bad. But the way to get good ideas is to get lots of ideas. The way to get lots of ideas is to lower your standards. If you don't lower your standards, you won't get any ideas at all.

The most important quality in a startup founder is determination. Not intelligence—determination. This is a little depressing. It would be nice if intelligence were the most important quality, since that's what we're usually judged by. But determination is more important, because intelligence without determination is like a car without an engine."#;

/// Configuration for a single needle-in-haystack test
#[derive(Debug, Clone)]
struct NeedleConfig {
    /// Total context length in tokens
    context_length: usize,
    /// Position of needle as percentage through context (0 to 100)
    depth_percent: usize,
    /// The needle fact to insert (should include surrounding newlines for separation)
    needle: String,
    /// Question to ask about the needle
    question: String,
    /// Expected answer
    expected_answer: String,
    /// Buffer for system message, question, and response
    final_context_length_buffer: usize,
}

impl Default for NeedleConfig {
    fn default() -> Self {
        Self {
            context_length: 2000,
            depth_percent: 50,
            // Note: Newlines around needle match original Python implementation
            needle: "\nThe special magic number mentioned in the context is 73.\n".to_string(),
            question: "What is the special magic number mentioned in the context?".to_string(),
            expected_answer: "73".to_string(),
            final_context_length_buffer: 200,
        }
    }
}

/// Read and prepare context files (repeating until we have enough for max context length)
fn read_context_files(
    max_context_length: usize,
    tokenizer: &Tokenizer,
) -> Result<String, Box<dyn std::error::Error>> {
    let essays_text = fetch_pg_essays().unwrap_or_else(|e| {
        eprintln!("Warning: Failed to fetch Paul Graham essays: {}", e);
        eprintln!("Using fallback essays...");
        FALLBACK_ESSAYS.to_string()
    });

    let mut context = String::new();

    // Keep adding essays until we have enough tokens
    while get_context_length_in_tokens(&context, tokenizer) < max_context_length {
        context.push_str(&essays_text);
        context.push(' ');
    }

    Ok(context)
}

/// Get the number of tokens in a context
fn get_context_length_in_tokens(context: &str, tokenizer: &Tokenizer) -> usize {
    tokenizer
        .encode(context, false)
        .map(|encoding| encoding.get_ids().len())
        .unwrap_or(0)
}

/// Encode and trim context to exact token length
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

/// Insert needle at specified depth, backing up to sentence boundary
/// This matches the original Python implementation exactly
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

    // Reduce context length by buffer
    let adjusted_context_length = context_length.saturating_sub(final_context_length_buffer);

    // If context + needle are longer than adjusted length, trim context
    if context_tokens.len() + needle_tokens.len() > adjusted_context_length {
        context_tokens.truncate(adjusted_context_length.saturating_sub(needle_tokens.len()));
    }

    let new_context_tokens = if depth_percent == 100 {
        // Needle at the end
        [context_tokens.as_slice(), needle_tokens].concat()
    } else {
        // Calculate insertion point
        let mut insertion_point = (context_tokens.len() * depth_percent) / 100;

        // Back up to find a sentence boundary (period token)
        // This matches the Python: while tokens_new_context and tokens_new_context[-1] not in period_tokens
        let period_tokens = tokenizer
            .encode(".", false)
            .map_err(|e| format!("Encoding period error: {}", e))?;
        let period_token_ids = period_tokens.get_ids();

        // Search backwards for a period token at the END of tokens_new_context
        // Check the last token in the slice before insertion point
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

        // Insert needle at sentence boundary
        [
            &context_tokens[..insertion_point],
            needle_tokens,
            &context_tokens[insertion_point..],
        ]
        .concat()
    };

    // Decode back to text
    Ok(tokenizer
        .decode(&new_context_tokens, false)
        .map_err(|e| format!("Decoding error: {}", e))?)
}

/// Generate context with needle inserted at specified depth
fn generate_context(
    config: &NeedleConfig,
    tokenizer: &Tokenizer,
    max_context_length: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    // Read context files
    let context = read_context_files(max_context_length, tokenizer)?;

    // Trim to exact context length
    let context = encode_and_trim(&context, config.context_length, tokenizer)?;

    // Insert needle at specified depth
    insert_needle(
        &context,
        &config.needle,
        config.depth_percent,
        config.context_length,
        config.final_context_length_buffer,
        tokenizer,
    )
}

/// Evaluate if the response contains the expected answer
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

    // Check for exact match or substring match
    if response_lower.contains(&expected_lower) {
        return 1.0;
    }

    // Check for the number in various formats
    if let Ok(expected_num) = expected_answer.parse::<i32>() {
        // Look for the number in the response
        for word in response.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| !c.is_numeric());
            if let Ok(num) = cleaned.parse::<i32>() {
                if num == expected_num {
                    if debug {
                        eprintln!("Found number match: {}", num);
                    }
                    return 1.0;
                }
            }
        }
    }

    if debug {
        eprintln!("=== No match found ===\n");
    }

    0.0
}

/// Run a single needle-in-haystack test
async fn run_niah_test(
    config: &NeedleConfig,
    model: &str,
    temperature: f32,
    tokenizer: &Tokenizer,
    max_context_length: usize,
    debug: bool,
    chunk: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Generate context with needle inserted
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

        // Find where the needle actually is
        if let Some(pos) = context_with_needle.find(&config.needle) {
            eprintln!("Needle found at character position: {}", pos);
            eprintln!(
                "Needle position as % of context: {:.1}%",
                (pos as f64 / context_with_needle.len() as f64) * 100.0
            );
        }
    }

    // Create SPNL query
    let system_prompt = "You are a helpful AI assistant. Answer the question based only on the information provided in the context. Be concise and direct.";
    let question = &config.question;
    let max_tokens = 300; // Match original Python implementation

    let query: Query = if chunk > 0 {
        // Split context into token-based chunks for map-reduce
        let encoding = tokenizer
            .encode(context_with_needle.as_str(), false)
            .map_err(|e| format!("Encoding error: {}", e))?;
        let tokens = encoding.get_ids();

        // Calculate chunk size in tokens
        let chunk_size_tokens = (tokens.len() + chunk - 1) / chunk; // Round up division

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
            // Reduce step: combine answers from all chunks
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
        // Original non-chunked query
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

    // Execute query
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

/// Compute quantiles for a set of values
fn compute_quantiles(values: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();

    let min = sorted[0];
    let p25 = sorted[len * 25 / 100];
    let p50 = sorted[len * 50 / 100];
    let p75 = sorted[len * 75 / 100];
    let p90 = sorted[len * 90 / 100];
    let p99 = sorted[len * 99 / 100];
    let max = sorted[len - 1];

    (min, p25, p50, p75, p90, p99, max)
}

/// Main benchmark function
fn niah_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("needle_in_haystack");

    // Configure sample size
    let sample_size = std::env::var("BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    group.sample_size(sample_size);

    // Configure measurement time
    let measurement_time = std::env::var("BENCH_MEASUREMENT_TIME")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    group.measurement_time(std::time::Duration::from_secs(measurement_time));

    // Read configuration from environment variables
    let context_lengths: Vec<usize> = std::env::var("BENCH_CONTEXT_LENGTHS")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![1000, 2000, 4000, 8000]);

    let depth_percentages: Vec<usize> = std::env::var("BENCH_DEPTH_PERCENTAGES")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![0, 25, 50, 75, 100]);

    let chunk_sizes: Vec<usize> = std::env::var("BENCH_CHUNK_SIZES")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![0, 2, 4]); // default: no chunking, 2-way, 4-way

    let model = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "ollama/granite3.3:2b".to_string());

    // Tokenizer model (HuggingFace format) - separate from inference model
    let tokenizer_model = std::env::var("BENCH_TOKENIZER_MODEL")
        .unwrap_or_else(|_| "ibm-granite/granite-3.3-2b-instruct".to_string());

    let final_context_length_buffer = std::env::var("BENCH_FINAL_CONTEXT_LENGTH_BUFFER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let temperature = 0.0;

    // Check for debug mode - only debug first sample
    let debug = std::env::var("BENCH_DEBUG")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let debug_counter = Arc::new(Mutex::new(0));

    // Load tokenizer from HuggingFace
    eprintln!("\n=== Loading tokenizer: {} ===", tokenizer_model);
    eprintln!("=== Using model for inference: {} ===", model);
    let tokenizer =
        Tokenizer::from_pretrained(&tokenizer_model, None).expect("Failed to load tokenizer");
    let max_context_length = *context_lengths.iter().max().unwrap_or(&8000);

    eprintln!("\n=== Needle In A Haystack Benchmark ===");
    eprintln!("Model: {}", model);
    eprintln!("Context lengths (tokens): {:?}", context_lengths);
    eprintln!("Depth percentages: {:?}", depth_percentages);
    eprintln!("Chunk sizes: {:?}", chunk_sizes);
    eprintln!("Sample size: {}", sample_size);
    eprintln!("Temperature: {}", temperature);
    eprintln!(
        "Final context length buffer: {}\n",
        final_context_length_buffer
    );

    // Run benchmarks for each combination of context length, depth, and chunk size
    for chunk_size in &chunk_sizes {
        for context_length in &context_lengths {
            for depth_percent in &depth_percentages {
                let accuracy_values = Arc::new(Mutex::new(Vec::new()));
                let accuracy_clone = Arc::clone(&accuracy_values);

                // Create progress bar
                let base_msg = format!(
                    "chunk={} len={} depth={}%",
                    chunk_size, context_length, depth_percent
                );
                let pb =
                    bench_progress::create_benchmark_progress(sample_size as u64, base_msg.clone());
                let pb_clone = Arc::clone(&pb);
                let base_msg = Arc::new(base_msg);
                let base_msg_clone = Arc::clone(&base_msg);

                let bench_id = format!(
                    "chunk={}/len={}/depth={}",
                    chunk_size, context_length, depth_percent
                );

                group.bench_with_input(
                    BenchmarkId::new("retrieval", bench_id),
                    &(*context_length, *depth_percent, *chunk_size),
                    |b, &(len, depth, chunk)| {
                        b.to_async(&runtime).iter(|| {
                            let accuracy_clone = Arc::clone(&accuracy_clone);
                            let pb = Arc::clone(&pb_clone);
                            let base_msg = Arc::clone(&base_msg_clone);
                            let model = model.clone();
                            let tokenizer = tokenizer.clone();
                            let debug_counter = Arc::clone(&debug_counter);

                            async move {
                                // Only debug first sample
                                let mut counter = debug_counter.lock().unwrap();
                                let should_debug = debug && *counter == 0;
                                *counter += 1;
                                drop(counter);
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

                                // Collect metrics
                                accuracy_clone.lock().unwrap().push(accuracy);

                                // Update progress bar
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

                // Finish progress bar
                let finish_msg = format!(
                    "✓ chunk={} len={} depth={}%",
                    chunk_size, context_length, depth_percent
                );
                bench_progress::finish_benchmark_progress(&pb, finish_msg);

                // Print statistics
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

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = niah_benchmark
}
criterion_main!(benches);

// Made with Bob

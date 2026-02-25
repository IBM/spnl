use super::{compute_quantiles_with_avg, create_benchmark_progress, finish_benchmark_progress};
use spnl::{
    ExecuteOptions, SpnlError, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Semaphore, mpsc};

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct RagcsvArgs {
    /// Path to CSV file
    #[arg(short, long, env = "RAGCSV_FILE")]
    pub file: String,

    /// Generative model
    #[arg(
        short,
        long,
        default_value = "ollama/granite3.3:2b",
        env = "RAGCSV_MODEL"
    )]
    pub model: String,

    /// Grading model (defaults to --model)
    #[arg(long, env = "RAGCSV_GRADING_MODEL")]
    pub grading_model: Option<String>,

    /// Concurrency level
    #[arg(long, default_value_t = 1, env = "RAGCSV_CONCURRENCY")]
    pub concurrency: usize,

    /// Limit number of rows to process
    #[arg(long, env = "RAGCSV_LIMIT")]
    pub limit: Option<usize>,

    /// Max tokens for primary query
    #[arg(long, default_value_t = 512, env = "RAGCSV_MAX_TOKENS")]
    pub max_tokens: i32,

    /// Enable debug output for first row
    #[arg(long)]
    pub debug: bool,

    /// Comma-separated LLM-judge metrics to run: accuracy,faithfulness,relevancy,all
    #[arg(long, default_value = "all", env = "RAGCSV_METRICS")]
    pub metrics: String,
}

// ---------------------------------------------------------------------------
// Metric flags
// ---------------------------------------------------------------------------

struct MetricFlags {
    accuracy: bool,
    faithfulness: bool,
    relevancy: bool,
}

impl MetricFlags {
    fn from_arg(arg: &str) -> Self {
        let tokens: HashSet<&str> = arg.split(',').map(|s| s.trim()).collect();
        if tokens.contains("all") {
            return Self {
                accuracy: true,
                faithfulness: true,
                relevancy: true,
            };
        }
        Self {
            accuracy: tokens.contains("accuracy"),
            faithfulness: tokens.contains("faithfulness"),
            relevancy: tokens.contains("relevancy"),
        }
    }
}

// ---------------------------------------------------------------------------
// CSV types
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct EvalRow {
    index: usize,
    expected: String,
    fragments: Vec<Fragment>,
    question: String,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct Fragment {
    page_content: String,
    metadata: FragmentMetadata,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct FragmentMetadata {
    #[serde(default)]
    title: String,
}

struct RowMetrics {
    #[allow(dead_code)]
    row_index: usize,
    accuracy: f64,
    faithfulness: f64,
    relevancy: f64,
    token_f1: f64,
    exact_match: f64,
    bleu: f64,
    total_time_ms: f64,
}

// ---------------------------------------------------------------------------
// Python repr → JSON conversion
// ---------------------------------------------------------------------------

fn python_repr_to_json(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let c = bytes[i] as char;

        match c {
            '\'' => {
                out.push('"');
                i += 1;
                while i < len {
                    let sc = bytes[i] as char;
                    match sc {
                        '\\' if i + 1 < len => {
                            let next = bytes[i + 1] as char;
                            if next == '\'' {
                                out.push('\'');
                                i += 2;
                            } else {
                                out.push('\\');
                                out.push(next);
                                i += 2;
                            }
                        }
                        '\'' => {
                            out.push('"');
                            i += 1;
                            break;
                        }
                        '"' => {
                            out.push('\\');
                            out.push('"');
                            i += 1;
                        }
                        _ => {
                            out.push(sc);
                            i += 1;
                        }
                    }
                }
            }
            '"' => {
                out.push('"');
                i += 1;
                while i < len {
                    let sc = bytes[i] as char;
                    match sc {
                        '\\' if i + 1 < len => {
                            out.push('\\');
                            out.push(bytes[i + 1] as char);
                            i += 2;
                        }
                        '"' => {
                            out.push('"');
                            i += 1;
                            break;
                        }
                        _ => {
                            out.push(sc);
                            i += 1;
                        }
                    }
                }
            }
            'N' if input[i..].starts_with("None") => {
                out.push_str("null");
                i += 4;
            }
            'T' if input[i..].starts_with("True") => {
                out.push_str("true");
                i += 4;
            }
            'F' if input[i..].starts_with("False") => {
                out.push_str("false");
                i += 5;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// CSV loading
// ---------------------------------------------------------------------------

fn load_csv(path: &str, limit: Option<usize>) -> Vec<EvalRow> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|e| panic!("Failed to open CSV at {path}: {e}"));

    let mut rows = Vec::new();
    for (idx, result) in rdr.records().enumerate() {
        if let Some(limit) = limit
            && idx >= limit
        {
            break;
        }
        let record = result.unwrap_or_else(|e| panic!("CSV parse error at row {idx}: {e}"));

        let expected = record.get(0).unwrap_or("").to_string();
        let fragments_raw = record.get(1).unwrap_or("[]").to_string();
        let question = record.get(4).unwrap_or("").to_string();

        let fragments_json = python_repr_to_json(&fragments_raw);
        let fragments: Vec<Fragment> = serde_json::from_str(&fragments_json).unwrap_or_else(|e| {
            if idx == 0 {
                eprintln!(
                    "Warning: failed to parse fragments for row {idx}: {e}\n  raw: {}",
                    &fragments_raw[..fragments_raw.len().min(200)]
                );
            }
            vec![]
        });

        rows.push(EvalRow {
            index: idx,
            expected,
            fragments,
            question,
        });
    }

    rows
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

fn build_primary_query(
    model: &str,
    question: &str,
    fragments: &[Fragment],
    max_tokens: i32,
) -> Query {
    let model = model.to_string();
    let system_prompt =
        "You are a helpful assistant. Answer the question based only on the provided Documents."
            .to_string();

    let doc_messages: Vec<Query> = fragments
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let text = format!("Document {idx}: {}", f.page_content);
            spnl!(user text)
        })
        .collect();

    let question = question.to_string();
    let temperature: f32 = 0.0;

    spnl!(
        g model
            (cross
                (system system_prompt)
                (plus doc_messages)
                (user question)
            )
            temperature
            max_tokens
    )
}

fn build_grading_query(model: &str, expected: &str, actual: &str) -> Query {
    let model = model.to_string();
    let system_prompt = "You are an accuracy evaluator. Compare the expected answer to the actual answer and return ONLY a single integer 0-100 representing accuracy percentage. 100 means perfectly correct, 0 means completely wrong.".to_string();
    let user_prompt = format!(
        "Expected answer: {expected}\n\nActual answer: {actual}\n\nAccuracy score (0-100):"
    );
    let temperature: f32 = 0.0;
    let max_tokens: i32 = 16;

    spnl!(
        g model
            (cross
                (system system_prompt)
                (user user_prompt)
            )
            temperature
            max_tokens
    )
}

fn parse_accuracy(response: &str) -> f64 {
    let trimmed = response.trim();
    trimmed
        .split(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 100.0))
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Non-LLM string metrics
// ---------------------------------------------------------------------------

fn normalize_tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn token_f1(expected: &str, actual: &str) -> f64 {
    let expected_tokens = normalize_tokens(expected);
    let actual_tokens = normalize_tokens(actual);
    if expected_tokens.is_empty() && actual_tokens.is_empty() {
        return 100.0;
    }
    if expected_tokens.is_empty() || actual_tokens.is_empty() {
        return 0.0;
    }

    let expected_counts: HashMap<&str, usize> =
        expected_tokens.iter().fold(HashMap::new(), |mut m, t| {
            *m.entry(t.as_str()).or_insert(0) += 1;
            m
        });
    let actual_counts: HashMap<&str, usize> =
        actual_tokens.iter().fold(HashMap::new(), |mut m, t| {
            *m.entry(t.as_str()).or_insert(0) += 1;
            m
        });

    let mut common = 0usize;
    for (tok, &count) in &actual_counts {
        common += count.min(*expected_counts.get(tok).unwrap_or(&0));
    }

    if common == 0 {
        return 0.0;
    }

    let precision = common as f64 / actual_tokens.len() as f64;
    let recall = common as f64 / expected_tokens.len() as f64;
    let f1 = 2.0 * precision * recall / (precision + recall);
    f1 * 100.0
}

fn exact_match(expected: &str, actual: &str) -> f64 {
    let norm_expected: String = normalize_tokens(expected).join(" ");
    let norm_actual: String = normalize_tokens(actual).join(" ");
    if norm_expected == norm_actual {
        100.0
    } else {
        0.0
    }
}

fn bleu_1(expected: &str, actual: &str) -> f64 {
    let ref_tokens = normalize_tokens(expected);
    let hyp_tokens = normalize_tokens(actual);
    if ref_tokens.is_empty() || hyp_tokens.is_empty() {
        return 0.0;
    }

    let ref_counts: HashMap<&str, usize> = ref_tokens.iter().fold(HashMap::new(), |mut m, t| {
        *m.entry(t.as_str()).or_insert(0) += 1;
        m
    });

    let mut clipped = 0usize;
    let mut hyp_counts: HashMap<&str, usize> = HashMap::new();
    for tok in &hyp_tokens {
        *hyp_counts.entry(tok.as_str()).or_insert(0) += 1;
    }
    for (tok, &count) in &hyp_counts {
        clipped += count.min(*ref_counts.get(tok).unwrap_or(&0));
    }

    let precision = clipped as f64 / hyp_tokens.len() as f64;

    // Brevity penalty
    let bp = if hyp_tokens.len() >= ref_tokens.len() {
        1.0
    } else {
        (1.0 - ref_tokens.len() as f64 / hyp_tokens.len() as f64).exp()
    };

    bp * precision * 100.0
}

// ---------------------------------------------------------------------------
// LLM-judge: faithfulness & relevancy
// ---------------------------------------------------------------------------

fn build_faithfulness_query(model: &str, answer: &str, fragments: &[Fragment]) -> Query {
    let model = model.to_string();
    let system_prompt = "You are a faithfulness evaluator. Determine whether the answer is grounded in the provided documents. Return ONLY a single integer 0-100. 100 means fully grounded in the documents, 0 means completely fabricated.".to_string();
    let docs: String = fragments
        .iter()
        .enumerate()
        .map(|(i, f)| format!("Document {i}: {}", f.page_content))
        .collect::<Vec<_>>()
        .join("\n\n");
    let user_prompt =
        format!("Documents:\n{docs}\n\nAnswer: {answer}\n\nFaithfulness score (0-100):");
    let temperature: f32 = 0.0;
    let max_tokens: i32 = 16;

    spnl!(
        g model
            (cross
                (system system_prompt)
                (user user_prompt)
            )
            temperature
            max_tokens
    )
}

fn build_relevancy_query(model: &str, question: &str, answer: &str) -> Query {
    let model = model.to_string();
    let system_prompt = "You are a relevancy evaluator. Determine whether the answer addresses the question. Return ONLY a single integer 0-100. 100 means the answer fully addresses the question, 0 means it is completely off-topic.".to_string();
    let user_prompt =
        format!("Question: {question}\n\nAnswer: {answer}\n\nRelevancy score (0-100):");
    let temperature: f32 = 0.0;
    let max_tokens: i32 = 16;

    spnl!(
        g model
            (cross
                (system system_prompt)
                (user user_prompt)
            )
            temperature
            max_tokens
    )
}

// ---------------------------------------------------------------------------
// Document reuse analysis
// ---------------------------------------------------------------------------

fn print_document_reuse_report(rows: &[EvalRow]) {
    // Build flat sequence of documents in order across all rows.
    let mut doc_sequence: Vec<&str> = Vec::new();
    let mut doc_titles: HashMap<&str, &str> = HashMap::new();

    for row in rows {
        for frag in &row.fragments {
            doc_sequence.push(&frag.page_content);
            if !frag.metadata.title.is_empty() {
                doc_titles.insert(&frag.page_content, &frag.metadata.title);
            }
        }
    }

    if doc_sequence.is_empty() {
        return;
    }

    // Token count per slot and prefix sums for fast range queries.
    let token_counts: Vec<usize> = doc_sequence
        .iter()
        .map(|d| normalize_tokens(d).len())
        .collect();
    let mut prefix_tokens: Vec<usize> = Vec::with_capacity(token_counts.len() + 1);
    prefix_tokens.push(0);
    for &c in &token_counts {
        prefix_tokens.push(prefix_tokens.last().unwrap() + c);
    }

    // Track ordered positions of each document.
    let mut doc_positions: HashMap<&str, Vec<usize>> = HashMap::new();
    for (pos, doc) in doc_sequence.iter().enumerate() {
        doc_positions.entry(doc).or_default().push(pos);
    }

    let total_slots = doc_sequence.len();
    let unique_count = doc_positions.len();

    // Documents that appear more than once, sorted by use count descending.
    let mut reused: Vec<(&str, &Vec<usize>)> = doc_positions
        .iter()
        .filter(|(_, positions)| positions.len() > 1)
        .map(|(doc, positions)| (*doc, positions))
        .collect();
    reused.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    // Reuse distance = total tokens of intervening documents between
    // consecutive occurrences of the same document.
    let intervening_tokens = |i: usize, j: usize| -> usize {
        // Tokens in slots (i+1)..j  (exclusive of both endpoints)
        prefix_tokens[j] - prefix_tokens[i + 1]
    };

    let mut all_distances: Vec<usize> = Vec::new();
    for (_, positions) in &reused {
        for w in positions.windows(2) {
            all_distances.push(intervening_tokens(w[0], w[1]));
        }
    }

    eprintln!("\n=== RAGCSV Document Reuse ===");
    eprintln!("  Total document slots:  {total_slots}");
    eprintln!("  Unique documents:      {unique_count}");
    eprintln!(
        "  Reused (>1 use):       {} ({:.1}% of unique)",
        reused.len(),
        if unique_count > 0 {
            reused.len() as f64 / unique_count as f64 * 100.0
        } else {
            0.0
        }
    );

    if !reused.is_empty() {
        let max_uses = reused[0].1.len();
        let avg_uses =
            reused.iter().map(|(_, p)| p.len()).sum::<usize>() as f64 / reused.len() as f64;
        eprintln!("  Avg uses (reused):     {avg_uses:.1}");
        eprintln!("  Max uses:              {max_uses}");

        if !all_distances.is_empty() {
            let avg_dist = all_distances.iter().sum::<usize>() as f64 / all_distances.len() as f64;
            let min_dist = *all_distances.iter().min().unwrap();
            let max_dist = *all_distances.iter().max().unwrap();
            eprintln!("  Avg reuse distance:    {avg_dist:.1} tokens");
            eprintln!("  Min reuse distance:    {min_dist} tokens");
            eprintln!("  Max reuse distance:    {max_dist} tokens");
        }

        eprintln!("  Top reused:");
        for (doc, positions) in reused.iter().take(5) {
            let label = doc_titles.get(doc).copied().unwrap_or_else(|| {
                let end = doc
                    .char_indices()
                    .nth(60)
                    .map(|(i, _)| i)
                    .unwrap_or(doc.len());
                &doc[..end]
            });
            let dists: Vec<usize> = positions
                .windows(2)
                .map(|w| intervening_tokens(w[0], w[1]))
                .collect();
            let avg_d = if dists.is_empty() {
                0.0
            } else {
                dists.iter().sum::<usize>() as f64 / dists.len() as f64
            };
            eprintln!(
                "    {}x  avg_dist={:.0}tok  \"{}\"",
                positions.len(),
                avg_d,
                label
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting helper
// ---------------------------------------------------------------------------

fn print_quantile_report(name: &str, values: &[f64], unit: &str) {
    let (min, p25, p50, p75, p90, p99, max, avg) = compute_quantiles_with_avg(values);
    eprintln!("\n=== RAGCSV {name} (n={}) ===", values.len());
    eprintln!("  min:  {min:.1}{unit}");
    eprintln!("  p25:  {p25:.1}{unit}");
    eprintln!("  p50:  {p50:.1}{unit}");
    eprintln!("  p75:  {p75:.1}{unit}");
    eprintln!("  p90:  {p90:.1}{unit}");
    eprintln!("  p99:  {p99:.1}{unit}");
    eprintln!("  max:  {max:.1}{unit}");
    eprintln!("  avg:  {avg:.1}{unit}");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(args: RagcsvArgs) -> Result<(), SpnlError> {
    let grading_model = args
        .grading_model
        .clone()
        .unwrap_or_else(|| args.model.clone());
    let max_tokens = args.max_tokens;
    let debug = args.debug;
    let flags = MetricFlags::from_arg(&args.metrics);

    let rows = load_csv(&args.file, args.limit);
    let total = rows.len();
    eprintln!("Loaded {total} rows from {}", args.file);
    eprintln!(
        "Model: {} | Grading: {} | Concurrency: {} | Max tokens: {}",
        args.model, grading_model, args.concurrency, max_tokens
    );
    eprintln!("Metrics: {}", args.metrics);

    print_document_reuse_report(&rows);

    if total == 0 {
        eprintln!("No rows to process.");
        return Ok(());
    }

    let semaphore = Arc::new(Semaphore::new(args.concurrency));
    let (tx, mut rx) = mpsc::channel::<RowMetrics>(total);

    let options = ExecuteOptions {
        silent: true,
        ..Default::default()
    };

    let flags = Arc::new(flags);

    for row in rows {
        let sem = Arc::clone(&semaphore);
        let tx = tx.clone();
        let model = args.model.clone();
        let grading_model = grading_model.clone();
        let flags = Arc::clone(&flags);
        let options = ExecuteOptions {
            silent: options.silent,
            ..Default::default()
        };

        tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            let row_idx = row.index;
            let start = Instant::now();

            let query = build_primary_query(&model, &row.question, &row.fragments, max_tokens);

            if debug && row_idx == 0 {
                eprintln!("\n=== DEBUG: Row 0 Query ===\n{query:?}");
            }

            let actual = match execute(&query, &options).await {
                Ok(Query::Message(Assistant(s))) => s,
                Ok(other) => {
                    eprintln!("Row {row_idx}: unexpected response type: {other:?}");
                    String::new()
                }
                Err(e) => {
                    eprintln!("Row {row_idx}: primary query error: {e}");
                    String::new()
                }
            };

            let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

            if debug && row_idx == 0 {
                eprintln!("=== DEBUG: Row 0 Response ===\n{actual}");
                eprintln!("=== DEBUG: Row 0 Expected ===\n{}", row.expected);
            }

            // Non-LLM metrics (zero cost, always run)
            let tf1 = token_f1(&row.expected, &actual);
            let em = exact_match(&row.expected, &actual);
            let bl = bleu_1(&row.expected, &actual);

            // LLM-judge metrics (gated by flags, run concurrently)
            let acc_fut = async {
                if flags.accuracy {
                    let q = build_grading_query(&grading_model, &row.expected, &actual);
                    match execute(&q, &options).await {
                        Ok(Query::Message(Assistant(s))) => {
                            let acc = parse_accuracy(&s);
                            if debug && row_idx == 0 {
                                eprintln!("=== DEBUG: Row 0 Grading Response ===\n{s}");
                                eprintln!("=== DEBUG: Row 0 Parsed Accuracy === {acc}%");
                            }
                            acc
                        }
                        Ok(_) => 0.0,
                        Err(e) => {
                            eprintln!("Row {row_idx}: accuracy grading error: {e}");
                            0.0
                        }
                    }
                } else {
                    -1.0
                }
            };

            let faith_fut = async {
                if flags.faithfulness {
                    let q = build_faithfulness_query(&grading_model, &actual, &row.fragments);
                    match execute(&q, &options).await {
                        Ok(Query::Message(Assistant(s))) => parse_accuracy(&s),
                        Ok(_) => 0.0,
                        Err(e) => {
                            eprintln!("Row {row_idx}: faithfulness grading error: {e}");
                            0.0
                        }
                    }
                } else {
                    -1.0
                }
            };

            let rel_fut = async {
                if flags.relevancy {
                    let q = build_relevancy_query(&grading_model, &row.question, &actual);
                    match execute(&q, &options).await {
                        Ok(Query::Message(Assistant(s))) => parse_accuracy(&s),
                        Ok(_) => 0.0,
                        Err(e) => {
                            eprintln!("Row {row_idx}: relevancy grading error: {e}");
                            0.0
                        }
                    }
                } else {
                    -1.0
                }
            };

            let (accuracy, faithfulness, relevancy) = tokio::join!(acc_fut, faith_fut, rel_fut);

            let _ = tx
                .send(RowMetrics {
                    row_index: row_idx,
                    accuracy,
                    faithfulness,
                    relevancy,
                    token_f1: tf1,
                    exact_match: em,
                    bleu: bl,
                    total_time_ms,
                })
                .await;
        });
    }

    drop(tx);

    let pb = create_benchmark_progress(total as u64, "RAGCSV Eval");
    let mut metrics: Vec<RowMetrics> = Vec::with_capacity(total);
    let mut accuracy_sum = 0.0;
    let mut pass_count = 0usize;

    while let Some(m) = rx.recv().await {
        if m.accuracy >= 0.0 {
            accuracy_sum += m.accuracy;
            if m.accuracy >= 75.0 {
                pass_count += 1;
            }
        }
        metrics.push(m);

        let n = metrics.len();
        let avg_acc = if flags.accuracy {
            accuracy_sum / n as f64
        } else {
            0.0
        };
        pb.set_position(n as u64);
        pb.set_message(format!(
            "{n}/{total} | Avg Acc={avg_acc:.1}% | Pass(>=75%)={pass_count}/{n}"
        ));
    }

    finish_benchmark_progress(
        &pb,
        format!(
            "Done {}/{total} | Avg Acc={:.1}% | Pass(>=75%)={pass_count}/{total}",
            metrics.len(),
            accuracy_sum / metrics.len().max(1) as f64
        ),
    );

    if metrics.is_empty() {
        eprintln!("\nNo results collected.");
        return Ok(());
    }

    // --- Quantile reports for each metric ---

    // LLM-judge metrics (only if enabled, filter out -1.0 sentinels)
    let acc_values: Vec<f64> = metrics
        .iter()
        .map(|m| m.accuracy)
        .filter(|&v| v >= 0.0)
        .collect();
    if !acc_values.is_empty() {
        print_quantile_report("Accuracy", &acc_values, "%");
        eprintln!("  pass (>=75%): {pass_count}/{}", acc_values.len());
    }

    let faith_values: Vec<f64> = metrics
        .iter()
        .map(|m| m.faithfulness)
        .filter(|&v| v >= 0.0)
        .collect();
    if !faith_values.is_empty() {
        print_quantile_report("Faithfulness", &faith_values, "%");
    }

    let rel_values: Vec<f64> = metrics
        .iter()
        .map(|m| m.relevancy)
        .filter(|&v| v >= 0.0)
        .collect();
    if !rel_values.is_empty() {
        print_quantile_report("Relevancy", &rel_values, "%");
    }

    // Non-LLM metrics (always present)
    let f1_values: Vec<f64> = metrics.iter().map(|m| m.token_f1).collect();
    print_quantile_report("Token F1", &f1_values, "%");

    let em_values: Vec<f64> = metrics.iter().map(|m| m.exact_match).collect();
    print_quantile_report("Exact Match", &em_values, "%");

    let bleu_values: Vec<f64> = metrics.iter().map(|m| m.bleu).collect();
    print_quantile_report("BLEU-1", &bleu_values, "%");

    // Latency
    let times: Vec<f64> = metrics.iter().map(|m| m.total_time_ms).collect();
    print_quantile_report("Total Time", &times, "ms");

    // --- Summary table ---
    let n = metrics.len() as f64;
    eprintln!("\n=== RAGCSV Summary (n={}) ===", metrics.len());
    if !acc_values.is_empty() {
        eprintln!(
            "  Accuracy:      {:.1}%",
            acc_values.iter().sum::<f64>() / acc_values.len() as f64
        );
    }
    if !faith_values.is_empty() {
        eprintln!(
            "  Faithfulness:  {:.1}%",
            faith_values.iter().sum::<f64>() / faith_values.len() as f64
        );
    }
    if !rel_values.is_empty() {
        eprintln!(
            "  Relevancy:     {:.1}%",
            rel_values.iter().sum::<f64>() / rel_values.len() as f64
        );
    }
    eprintln!("  Token F1:      {:.1}%", f1_values.iter().sum::<f64>() / n);
    eprintln!("  Exact Match:   {:.1}%", em_values.iter().sum::<f64>() / n);
    eprintln!(
        "  BLEU-1:        {:.1}%",
        bleu_values.iter().sum::<f64>() / n
    );
    eprintln!("  Avg Latency:   {:.0}ms", times.iter().sum::<f64>() / n);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- python_repr_to_json ----

    #[test]
    fn python_repr_single_quoted_strings() {
        assert_eq!(python_repr_to_json("{'a': 'b'}"), r#"{"a": "b"}"#);
    }

    #[test]
    fn python_repr_none_true_false() {
        assert_eq!(
            python_repr_to_json("{'x': None, 'y': True, 'z': False}"),
            r#"{"x": null, "y": true, "z": false}"#
        );
    }

    #[test]
    fn python_repr_embedded_double_quotes() {
        // A single-quoted Python string containing a double quote should escape it
        assert_eq!(
            python_repr_to_json(r#"{'a': 'he said "hi"'}"#),
            r#"{"a": "he said \"hi\""}"#
        );
    }

    #[test]
    fn python_repr_escaped_single_quotes() {
        // Python: 'it\'s' -> JSON: "it's"
        assert_eq!(python_repr_to_json(r"{'a': 'it\'s'}"), r#"{"a": "it's"}"#);
    }

    // ---- parse_accuracy ----

    #[test]
    fn parse_accuracy_numeric_string() {
        assert!((parse_accuracy("85") - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_accuracy_text_with_number() {
        assert!((parse_accuracy("The accuracy is 72 percent") - 72.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_accuracy_clamps_above_100() {
        assert!((parse_accuracy("150") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_accuracy_no_number_returns_zero() {
        assert!((parse_accuracy("no numbers here")).abs() < f64::EPSILON);
    }

    // ---- normalize_tokens ----

    #[test]
    fn normalize_tokens_splits_punctuation() {
        let tokens = normalize_tokens("Hello, world!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn normalize_tokens_case_folding() {
        let tokens = normalize_tokens("ABC def GHI");
        assert_eq!(tokens, vec!["abc", "def", "ghi"]);
    }

    #[test]
    fn normalize_tokens_empty_filtering() {
        let tokens = normalize_tokens("  ...  ");
        assert!(tokens.is_empty());
    }

    // ---- token_f1 ----

    #[test]
    fn token_f1_identical() {
        assert!((token_f1("the cat sat", "the cat sat") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn token_f1_no_overlap() {
        assert!(token_f1("alpha beta", "gamma delta").abs() < f64::EPSILON);
    }

    #[test]
    fn token_f1_partial_overlap() {
        let score = token_f1("the cat sat on the mat", "the cat");
        assert!(score > 0.0 && score < 100.0);
    }

    #[test]
    fn token_f1_both_empty() {
        assert!((token_f1("", "") - 100.0).abs() < f64::EPSILON);
    }

    // ---- exact_match ----

    #[test]
    fn exact_match_identical_after_normalization() {
        assert!((exact_match("Hello, World!", "hello world") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn exact_match_different() {
        assert!(exact_match("foo", "bar").abs() < f64::EPSILON);
    }

    // ---- bleu_1 ----

    #[test]
    fn bleu_1_identical() {
        assert!((bleu_1("the cat sat", "the cat sat") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bleu_1_no_overlap() {
        assert!(bleu_1("alpha beta", "gamma delta").abs() < f64::EPSILON);
    }

    #[test]
    fn bleu_1_brevity_penalty() {
        // hyp is shorter than ref, so brevity penalty applies
        let long_ref = "the quick brown fox jumps over the lazy dog";
        let short_hyp = "the fox";
        let score = bleu_1(long_ref, short_hyp);
        assert!(score > 0.0 && score < 100.0);
    }

    // ---- MetricFlags::from_arg ----

    #[test]
    fn metric_flags_all() {
        let f = MetricFlags::from_arg("all");
        assert!(f.accuracy && f.faithfulness && f.relevancy);
    }

    #[test]
    fn metric_flags_single() {
        let f = MetricFlags::from_arg("accuracy");
        assert!(f.accuracy);
        assert!(!f.faithfulness);
        assert!(!f.relevancy);
    }

    #[test]
    fn metric_flags_comma_separated() {
        let f = MetricFlags::from_arg("accuracy,relevancy");
        assert!(f.accuracy);
        assert!(!f.faithfulness);
        assert!(f.relevancy);
    }
}

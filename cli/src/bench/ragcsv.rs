use super::{compute_quantiles_with_avg, create_benchmark_progress, finish_benchmark_progress};
use spnl::{
    ExecuteOptions, SpnlError, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
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
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(args: RagcsvArgs) -> Result<(), SpnlError> {
    let grading_model = args
        .grading_model
        .clone()
        .unwrap_or_else(|| args.model.clone());
    let max_tokens = args.max_tokens;
    let debug = args.debug;

    let rows = load_csv(&args.file, args.limit);
    let total = rows.len();
    eprintln!("Loaded {total} rows from {}", args.file);
    eprintln!(
        "Model: {} | Grading: {} | Concurrency: {} | Max tokens: {}",
        args.model, grading_model, args.concurrency, max_tokens
    );

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

    for row in rows {
        let sem = Arc::clone(&semaphore);
        let tx = tx.clone();
        let model = args.model.clone();
        let grading_model = grading_model.clone();
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

            let grading_query = build_grading_query(&grading_model, &row.expected, &actual);
            let accuracy = match execute(&grading_query, &options).await {
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
                    eprintln!("Row {row_idx}: grading query error: {e}");
                    0.0
                }
            };

            let _ = tx
                .send(RowMetrics {
                    row_index: row_idx,
                    accuracy,
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
        accuracy_sum += m.accuracy;
        if m.accuracy >= 75.0 {
            pass_count += 1;
        }
        metrics.push(m);

        let n = metrics.len();
        let avg_acc = accuracy_sum / n as f64;
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

    let accuracies: Vec<f64> = metrics.iter().map(|m| m.accuracy).collect();
    let (min, p25, p50, p75, p90, p99, max, avg) = compute_quantiles_with_avg(&accuracies);

    eprintln!("\n=== RAGCSV Eval Accuracy (n={}) ===", metrics.len());
    eprintln!("  min:  {min:.1}%");
    eprintln!("  p25:  {p25:.1}%");
    eprintln!("  p50:  {p50:.1}%");
    eprintln!("  p75:  {p75:.1}%");
    eprintln!("  p90:  {p90:.1}%");
    eprintln!("  p99:  {p99:.1}%");
    eprintln!("  max:  {max:.1}%");
    eprintln!("  avg:  {avg:.1}%");
    eprintln!("  pass (>=75%): {pass_count}/{}", metrics.len());

    let times: Vec<f64> = metrics.iter().map(|m| m.total_time_ms).collect();
    let (tmin, t25, t50, t75, t90, t99, tmax, tavg) = compute_quantiles_with_avg(&times);

    eprintln!("\n=== RAGCSV Eval Total Time (n={}) ===", metrics.len());
    eprintln!("  min:  {tmin:.0}ms");
    eprintln!("  p25:  {t25:.0}ms");
    eprintln!("  p50:  {t50:.0}ms");
    eprintln!("  p75:  {t75:.0}ms");
    eprintln!("  p90:  {t90:.0}ms");
    eprintln!("  p99:  {t99:.0}ms");
    eprintln!("  max:  {tmax:.0}ms");
    eprintln!("  avg:  {tavg:.0}ms");

    Ok(())
}

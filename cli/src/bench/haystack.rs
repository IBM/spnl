use super::{compute_quantiles, create_benchmark_progress, finish_benchmark_progress};
use criterion::{BenchmarkId, Criterion};
use petname::Generator;
use spnl::{
    ExecuteOptions, SpnlError, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
use std::sync::{Arc, Mutex};

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct HaystackArgs {
    /// Generative model
    #[arg(
        short,
        long,
        default_value = "ollama/granite3.3:2b",
        env = "BENCH_MODEL"
    )]
    pub model: String,

    /// Sample size per configuration
    #[arg(long, default_value_t = 100, env = "BENCH_SAMPLE_SIZE")]
    pub sample_size: usize,

    /// Measurement time in seconds
    #[arg(long, default_value_t = 320, env = "BENCH_MEASUREMENT_TIME")]
    pub measurement_time: u64,

    /// Comma-separated document counts for basic benchmark
    #[arg(long, default_value = "2,4,8", value_parser = super::parse_csv_usize, env = "BENCH_NUM_DOCS")]
    pub num_docs: Vec<usize>,

    /// Comma-separated document lengths (words of filler per doc)
    #[arg(long, default_value = "0,10,100,200,400,600,800,1000", value_parser = super::parse_csv_usize, env = "BENCH_DOC_LENGTH")]
    pub doc_lengths: Vec<usize>,

    /// Comma-separated chunk sizes for map-reduce benchmark
    #[arg(long, default_value = "2,4", value_parser = super::parse_csv_usize, env = "BENCH_CHUNK_SIZES")]
    pub chunk_sizes: Vec<usize>,

    /// Number of documents for the map-reduce benchmark
    #[arg(long, default_value_t = 8, env = "BENCH_MAP_REDUCE_NUM_DOCS")]
    pub map_reduce_num_docs: usize,

    /// Skip the basic (non-chunked) benchmark
    #[arg(long)]
    pub no_basic: bool,

    /// Skip the map-reduce benchmark
    #[arg(long)]
    pub no_map_reduce: bool,
}

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------

type GeneratedNames = Vec<String>;
#[derive(serde::Deserialize)]
struct Name {
    name: String,
}
type GeneratedNames2 = Vec<Name>;

fn ratio(n: usize, d: usize) -> f64 {
    (n as f64) / (d as f64)
}

fn score(expected: &[String], actual: &[String]) -> (f64, f64) {
    let true_positives = actual.iter().filter(|s| expected.contains(s)).count();
    let false_positives = actual.iter().filter(|s| !expected.contains(s)).count();
    let false_negatives = expected.iter().filter(|s| !actual.contains(s)).count();
    let precision = ratio(true_positives, true_positives + false_positives);
    let recall = ratio(true_positives, true_positives + false_negatives);
    (if precision.is_nan() { 0.0 } else { precision }, recall)
}

fn score_chain(expected: &[String], actual: &[String]) -> (f64, f64) {
    let do_not_want = &expected[0];
    let n = expected.len() - 1;
    (
        1.0 - ratio(actual[1..].iter().filter(|b| *b == do_not_want).count(), n),
        0.0,
    )
}

// ---------------------------------------------------------------------------
// Core test
// ---------------------------------------------------------------------------

async fn run_haystack_benchmark(
    model: &str,
    temperature: f32,
    length: usize,
    num_documents: usize,
    chain: bool,
    chunk: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let name_generator = petname::Petnames::default();
    let names: Vec<String> = (0..num_documents)
        .filter_map(|_| name_generator.generate_one(2, "-"))
        .collect();
    assert_eq!(names.len(), num_documents);

    let mut rng = rand::rng();
    let docs: Vec<Query> = if chain {
        names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                if idx == 0 {
                    format!(
                        "I am a cat, and my name is {name}. {}",
                        lipsum::lipsum_words_with_rng(&mut rng, length)
                    )
                } else {
                    format!(
                        "I am also a cat, and I have the same name as the previous cat! {}",
                        lipsum::lipsum_words_with_rng(&mut rng, length)
                    )
                }
            })
            .map(|text| spnl!(user text))
            .collect()
    } else {
        names
            .iter()
            .map(|name| {
                format!(
                    "I am a cat, and my name is {name}. {}",
                    lipsum::lipsum_words_with_rng(&mut rng, length)
                )
            })
            .map(|text| spnl!(user text))
            .collect()
    };

    let expected_names = if chain {
        let mut v = ::std::iter::repeat_n("".to_string(), num_documents).collect::<Vec<_>>();
        v[0] = names[0].clone();
        v
    } else {
        names
    };

    let system_prompt = r#"Your are an AI that responds to questions with a plain JSON array of strings such as ["a","b","c"] or ["x","y","z","w"] or ["hello","world"], no markdown or html or any other extra text"#;
    let user_prompt = "Tell me the names of the cats mentioned";

    let query: Query = if chunk > 0 {
        let chunks: Vec<Query> = docs
            .chunks(chunk)
            .map(|chunk| chunk.to_vec())
            .map(|chunk| {
                spnl!(
                g model
                    (cross (system system_prompt) (plus chunk) (user user_prompt))
                    temperature)
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
                     (user "Combine these arrays into one array")
                    )
                    temperature
            )
        }
    } else {
        spnl!(
            g model
                (cross
                 (system system_prompt)
                 (plus docs)
                 (user user_prompt)
                )
                temperature
        )
    };

    let options = ExecuteOptions {
        silent: true,
        ..Default::default()
    };
    match execute(&query, &options).await? {
        Query::Message(Assistant(ss)) => {
            let s = if let Some(idx) = ss.find("```json") {
                ss[idx + 7..ss.len() - 3].trim()
            } else {
                ss.trim()
            };

            let generated_names: GeneratedNames = serde_json::from_str::<GeneratedNames>(s)
                .unwrap_or_else(|_| {
                    let n2: GeneratedNames2 = serde_json::from_str(s).unwrap_or_else(|_| vec![]);
                    n2.into_iter().map(|n| n.name).collect()
                })
                .into_iter()
                .map(|s| s.to_lowercase())
                .collect();

            let (precision, recall) = if chain {
                score_chain(&expected_names, &generated_names)
            } else {
                score(&expected_names, &generated_names)
            };

            Ok((precision, recall))
        }
        x => Err(format!("Unexpected non-string response {x:?}").into()),
    }
}

// ---------------------------------------------------------------------------
// Criterion benchmark function
// ---------------------------------------------------------------------------

fn haystack_benchmark(c: &mut Criterion, args: &HaystackArgs) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("haystack");

    group.sample_size(args.sample_size);
    group.measurement_time(std::time::Duration::from_secs(args.measurement_time));

    let model = &args.model;

    // Basic haystack benchmark with different document counts
    if !args.no_basic {
        for &num_docs in &args.num_docs {
            for &doc_length in &args.doc_lengths {
                let precision_values = Arc::new(Mutex::new(Vec::new()));
                let recall_values = Arc::new(Mutex::new(Vec::new()));

                let precision_clone = Arc::clone(&precision_values);
                let recall_clone = Arc::clone(&recall_values);

                let base_msg = format!("basic docs={} len={}", num_docs, doc_length);
                let pb = create_benchmark_progress(args.sample_size as u64, base_msg.clone());
                let pb_clone = Arc::clone(&pb);
                let base_msg = Arc::new(base_msg);
                let base_msg_clone = Arc::clone(&base_msg);

                group.bench_with_input(
                    BenchmarkId::new("basic", format!("docs={}/len={}", num_docs, doc_length)),
                    &num_docs,
                    |b, &num_docs| {
                        b.to_async(&runtime).iter(|| {
                            let precision_clone = Arc::clone(&precision_clone);
                            let recall_clone = Arc::clone(&recall_clone);
                            let pb = Arc::clone(&pb_clone);
                            let base_msg = Arc::clone(&base_msg_clone);
                            let model = model.clone();
                            async move {
                                let temperature = 0.0;
                                let (precision, recall) = run_haystack_benchmark(
                                    &model,
                                    temperature,
                                    doc_length,
                                    num_docs,
                                    false,
                                    0,
                                )
                                .await
                                .unwrap();

                                precision_clone.lock().unwrap().push(precision);
                                recall_clone.lock().unwrap().push(recall);

                                let precisions = precision_clone.lock().unwrap();
                                let recalls = recall_clone.lock().unwrap();
                                let total_count = precisions.len();
                                let avg_p = precisions.iter().sum::<f64>() / total_count as f64;
                                let avg_r = recalls.iter().sum::<f64>() / total_count as f64;
                                let high_precision_count =
                                    precisions.iter().filter(|&&p| p >= 0.75).count();
                                let high_recall_count =
                                    recalls.iter().filter(|&&r| r >= 0.75).count();
                                drop(precisions);
                                drop(recalls);

                                pb.set_message(format!(
                                    "{} \x1b[1m|\x1b[0m n={} \x1b[1m|\x1b[0m P={:.1}% n≥75%={} \x1b[1m|\x1b[0m R={:.1}% n≥75%={}",
                                    base_msg,
                                    total_count,
                                    avg_p * 100.0,
                                    high_precision_count,
                                    avg_r * 100.0,
                                    high_recall_count
                                ));
                                pb.inc(1);

                                (precision, recall)
                            }
                        });
                    },
                );

                finish_benchmark_progress(
                    &pb,
                    format!("✓ basic docs={} len={}", num_docs, doc_length),
                );

                let precisions = precision_values.lock().unwrap();
                let recalls = recall_values.lock().unwrap();
                if !precisions.is_empty() {
                    let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&precisions);
                    eprintln!(
                        "\n=== Precision Quantiles for num_docs={} (n={}) ===",
                        num_docs,
                        precisions.len()
                    );
                    eprintln!("  min: {:.4}", min);
                    eprintln!("  p25: {:.4}", p25);
                    eprintln!("  p50: {:.4}", p50);
                    eprintln!("  p75: {:.4}", p75);
                    eprintln!("  p90: {:.4}", p90);
                    eprintln!("  p99: {:.4}", p99);
                    eprintln!("  max: {:.4}", max);

                    let (rmin, r25, r50, r75, r90, r99, rmax) = compute_quantiles(&recalls);
                    eprintln!(
                        "=== Recall Quantiles for num_docs={} (n={}) ===",
                        num_docs,
                        recalls.len()
                    );
                    eprintln!("  min: {:.4}", rmin);
                    eprintln!("  p25: {:.4}", r25);
                    eprintln!("  p50: {:.4}", r50);
                    eprintln!("  p75: {:.4}", r75);
                    eprintln!("  p90: {:.4}", r90);
                    eprintln!("  p99: {:.4}", r99);
                    eprintln!("  max: {:.4}\n", rmax);
                }
            }
        }
    }

    // Map-reduce benchmark with chunking
    if !args.no_map_reduce {
        let map_reduce_num_docs = args.map_reduce_num_docs;

        for &chunk_size in &args.chunk_sizes {
            for &doc_length in &args.doc_lengths {
                let precision_values = Arc::new(Mutex::new(Vec::new()));
                let recall_values = Arc::new(Mutex::new(Vec::new()));

                let precision_clone = Arc::clone(&precision_values);
                let recall_clone = Arc::clone(&recall_values);

                let base_msg = format!(
                    "map_reduce chunk={} docs={} len={}",
                    chunk_size, map_reduce_num_docs, doc_length
                );
                let pb = create_benchmark_progress(args.sample_size as u64, base_msg.clone());
                let pb_clone = Arc::clone(&pb);
                let base_msg = Arc::new(base_msg);
                let base_msg_clone = Arc::clone(&base_msg);

                group.bench_with_input(
                    BenchmarkId::new(
                        "map_reduce",
                        format!(
                            "chunk={}/docs={}/len={}",
                            chunk_size, map_reduce_num_docs, doc_length
                        ),
                    ),
                    &chunk_size,
                    |b, &chunk_size| {
                        b.to_async(&runtime).iter(|| {
                            let precision_clone = Arc::clone(&precision_clone);
                            let recall_clone = Arc::clone(&recall_clone);
                            let pb = Arc::clone(&pb_clone);
                            let base_msg = Arc::clone(&base_msg_clone);
                            let model = model.clone();
                            async move {
                                let temperature = 0.0;
                                let (precision, recall) = run_haystack_benchmark(
                                    &model,
                                    temperature,
                                    doc_length,
                                    map_reduce_num_docs,
                                    false,
                                    chunk_size,
                                )
                                .await
                                .unwrap();

                                precision_clone.lock().unwrap().push(precision);
                                recall_clone.lock().unwrap().push(recall);

                                let precisions = precision_clone.lock().unwrap();
                                let recalls = recall_clone.lock().unwrap();
                                let total_count = precisions.len();
                                let avg_p = precisions.iter().sum::<f64>() / total_count as f64;
                                let avg_r = recalls.iter().sum::<f64>() / total_count as f64;
                                let high_precision_count =
                                    precisions.iter().filter(|&&p| p >= 0.75).count();
                                let high_recall_count =
                                    recalls.iter().filter(|&&r| r >= 0.75).count();
                                drop(precisions);
                                drop(recalls);

                                pb.set_message(format!(
                                    "{} \x1b[1m|\x1b[0m n={} \x1b[1m|\x1b[0m P={:.1}% n≥75%={} \x1b[1m|\x1b[0m R={:.1}% n≥75%={}",
                                    base_msg,
                                    total_count,
                                    avg_p * 100.0,
                                    high_precision_count,
                                    avg_r * 100.0,
                                    high_recall_count
                                ));
                                pb.inc(1);

                                (precision, recall)
                            }
                        });
                    },
                );

                finish_benchmark_progress(
                    &pb,
                    format!(
                        "✓ map_reduce chunk={} docs={} len={}",
                        chunk_size, map_reduce_num_docs, doc_length
                    ),
                );

                let precisions = precision_values.lock().unwrap();
                let recalls = recall_values.lock().unwrap();
                if !precisions.is_empty() {
                    let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&precisions);
                    eprintln!(
                        "\n=== Precision Quantiles for chunk_size={} (n={}) ===",
                        chunk_size,
                        precisions.len()
                    );
                    eprintln!("  min: {:.4}", min);
                    eprintln!("  p25: {:.4}", p25);
                    eprintln!("  p50: {:.4}", p50);
                    eprintln!("  p75: {:.4}", p75);
                    eprintln!("  p90: {:.4}", p90);
                    eprintln!("  p99: {:.4}", p99);
                    eprintln!("  max: {:.4}", max);

                    let (rmin, r25, r50, r75, r90, r99, rmax) = compute_quantiles(&recalls);
                    eprintln!(
                        "=== Recall Quantiles for chunk_size={} (n={}) ===",
                        chunk_size,
                        recalls.len()
                    );
                    eprintln!("  min: {:.4}", rmin);
                    eprintln!("  p25: {:.4}", r25);
                    eprintln!("  p50: {:.4}", r50);
                    eprintln!("  p75: {:.4}", r75);
                    eprintln!("  p90: {:.4}", r90);
                    eprintln!("  p99: {:.4}", r99);
                    eprintln!("  max: {:.4}\n", rmax);
                }
            }
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: HaystackArgs) -> Result<(), SpnlError> {
    let mut criterion = Criterion::default();
    haystack_benchmark(&mut criterion, &args);
    criterion.final_summary();
    Ok(())
}

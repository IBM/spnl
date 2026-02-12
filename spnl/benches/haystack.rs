mod bench_progress;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use petname::Generator;
use spnl::{
    ExecuteOptions, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
use std::sync::{Arc, Mutex};

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
            // oof, be gracious here. sometimes the model wraps the
            // requested json array with markdown even though we asked
            // it not to
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

fn compute_quantiles(values: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
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

fn haystack_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("haystack");

    // Configure sample size (default 100 for meaningful quantile statistics)
    let sample_size = std::env::var("BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    group.sample_size(sample_size);

    // Configure measurement time (default 320s to allow 100 samples at ~3s per call)
    let measurement_time = std::env::var("BENCH_MEASUREMENT_TIME")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(320);
    group.measurement_time(std::time::Duration::from_secs(measurement_time));

    // Read configuration from environment variables
    let run_basic = std::env::var("BENCH_BASIC")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(true); // default to true

    let run_map_reduce = std::env::var("BENCH_MAP_REDUCE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(true); // default to true

    let num_docs_list: Vec<usize> = std::env::var("BENCH_NUM_DOCS")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![2, 4, 8]); // default

    let doc_lengths: Vec<usize> = std::env::var("BENCH_DOC_LENGTH")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![0, 10, 100, 200, 400, 600, 800, 1000]); // default

    let chunk_sizes: Vec<usize> = std::env::var("BENCH_CHUNK_SIZES")
        .ok()
        .and_then(|s| {
            s.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![2, 4]); // default

    let map_reduce_num_docs: usize = std::env::var("BENCH_MAP_REDUCE_NUM_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8); // default

    // Basic haystack benchmark with different document counts
    if run_basic {
        for num_docs in num_docs_list {
            for doc_length in &doc_lengths {
                let precision_values = Arc::new(Mutex::new(Vec::new()));
                let recall_values = Arc::new(Mutex::new(Vec::new()));

                let precision_clone = Arc::clone(&precision_values);
                let recall_clone = Arc::clone(&recall_values);

                // Create progress bar for this benchmark
                let base_msg = format!("basic docs={} len={}", num_docs, doc_length);
                let pb = bench_progress::create_benchmark_progress(
                    100, // sample_size
                    base_msg.clone(),
                );
                let pb_clone = Arc::clone(&pb);
                let base_msg = Arc::new(base_msg);
                let base_msg_clone = Arc::clone(&base_msg);

                group.bench_with_input(
                    BenchmarkId::new("basic", format!("docs={}/len={}", num_docs, doc_length)),
                    &num_docs,
                    |b, &num_docs| {
                        let doc_length = *doc_length;
                        b.to_async(&runtime).iter(|| {
                            let precision_clone = Arc::clone(&precision_clone);
                            let recall_clone = Arc::clone(&recall_clone);
                            let pb = Arc::clone(&pb_clone);
                            let base_msg = Arc::clone(&base_msg_clone);
                            async move {
                                let model = "ollama/granite3.3:2b";
                                let temperature = 0.0;
                                let length = doc_length;
                                let (precision, recall) = run_haystack_benchmark(
                                    model,
                                    temperature,
                                    length,
                                    num_docs,
                                    false,
                                    0,
                                )
                                .await
                                .unwrap();

                                // Collect metrics
                                precision_clone.lock().unwrap().push(precision);
                                recall_clone.lock().unwrap().push(recall);

                                /* println!(
                                    "{} {} {} {} {} {}",
                                    model, temperature, num_docs, length, precision, recall
                                ); */

                                // Update progress bar with running averages
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

                                bench_progress::update_progress_with_stats(
                                    &pb,
                                    &base_msg,
                                    avg_p,
                                    avg_r,
                                    high_precision_count,
                                    high_recall_count,
                                    total_count,
                                );
                                pb.inc(1);

                                (precision, recall)
                            }
                        });
                    },
                );

                // Finish progress bar
                bench_progress::finish_benchmark_progress(
                    &pb,
                    format!("✓ basic docs={} len={}", num_docs, doc_length),
                );

                // Print quantiles after benchmark completes
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
    if run_map_reduce {
        for chunk_size in chunk_sizes {
            for doc_length in &doc_lengths {
                let precision_values = Arc::new(Mutex::new(Vec::new()));
                let recall_values = Arc::new(Mutex::new(Vec::new()));

                let precision_clone = Arc::clone(&precision_values);
                let recall_clone = Arc::clone(&recall_values);

                // Create progress bar for this benchmark
                let base_msg = format!(
                    "map_reduce chunk={} docs={} len={}",
                    chunk_size, map_reduce_num_docs, doc_length
                );
                let pb = bench_progress::create_benchmark_progress(
                    100, // sample_size
                    base_msg.clone(),
                );
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
                        let doc_length = *doc_length;
                        b.to_async(&runtime).iter(|| {
                            let precision_clone = Arc::clone(&precision_clone);
                            let recall_clone = Arc::clone(&recall_clone);
                            let pb = Arc::clone(&pb_clone);
                            let base_msg = Arc::clone(&base_msg_clone);
                            async move {
                                let model = "ollama/granite3.3:2b";
                                let temperature = 0.0;
                                let num_docs = map_reduce_num_docs;
                                let length = doc_length;
                                let (precision, recall) = run_haystack_benchmark(
                                    model,
                                    temperature,
                                    length,
                                    num_docs,
                                    false,
                                    chunk_size,
                                )
                                .await
                                .unwrap();

                                // Collect metrics
                                precision_clone.lock().unwrap().push(precision);
                                recall_clone.lock().unwrap().push(recall);

                                /* println!(
                                    "{} {} {} {} {} {}",
                                    model, temperature, num_docs, length, precision, recall
                                ); */

                                // Update progress bar with running averages
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

                                bench_progress::update_progress_with_stats(
                                    &pb,
                                    &base_msg,
                                    avg_p,
                                    avg_r,
                                    high_precision_count,
                                    high_recall_count,
                                    total_count,
                                );
                                pb.inc(1);

                                (precision, recall)
                            }
                        });
                    },
                );

                // Finish progress bar
                bench_progress::finish_benchmark_progress(
                    &pb,
                    format!(
                        "✓ map_reduce chunk={} docs={} len={}",
                        chunk_size, map_reduce_num_docs, doc_length
                    ),
                );

                // Print quantiles after benchmark completes
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

// Configure Criterion to be quieter
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = haystack_benchmark
}
criterion_main!(benches);

// Made with Bob

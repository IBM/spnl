//! PIC (Position-Independent Caching) cross-request benchmark.
//!
//! Measures the **prefill latency** (TTFT) benefit of reusing cached Plus block
//! KV entries across requests that contain the same documents in different orders.
//! Also reports PIC hit/miss rates.
//!
//! # What is measured
//!
//! - **TTFT (time to first token)**: Approximated by setting `max_tokens=1` so
//!   wall-clock time ≈ prefill time. This is where PIC saves work — cached Plus
//!   blocks skip prefill entirely.
//! - **PIC hit rate**: How many reuse requests found all their Plus blocks
//!   in the content-based PIC.
//! - **Accuracy** (`-o accuracy`): Compares responses generated **with** Plus blocks
//!   (PIC block attention) vs **without** Plus (standard causal attention) for the
//!   same documents and question. Reports token F1 (word overlap) and optional
//!   LLM-judge semantic equivalence score (with `--grading-model`).
//!
//! # Options
//!
//! - **`--length`/`-l`**: Words per document for TTFT, or max tokens for accuracy (default: 200).
//! - **`--size`/`-s`**: Sweep across doc lengths, e.g. `-s xs,sm,m,lg,xl,xxl`.
//! - **`--model`/`-m`**: One or more models: `-m model1,model2`.
//! - **`--full`**: Sugar for `-s xs,sm,m,lg,xl,xxl -m llama3.2:1b,llama3.2:3b,llama3.1:8b,qwen2.5:0.5b,qwen2.5:14b`.
//!
//! Output is controlled by `-o/--output` (comma-separated): speedup, iqr, hitrate,
//! latency, json, accuracy. Multiple modes can be combined, e.g. `-o speedup,accuracy`.
//!
//! # Prerequisites
//!
//! PIC cross-request caching only works with the **local** (mistral.rs) backend.
//! The model must resolve to the local backend, either via:
//! - A `local/` prefix: `-m local/my-hf-model-id`
//! - A pretty name: `-m llama3.2:3b` (resolved to a HuggingFace GGUF model)
//!
//! Remote backends (`ollama/`, `openai/`, `gemini/`, `spnl/`) flatten Plus blocks
//! into ordinary messages and do not support PIC caching.
//!
//! ## Required features
//!
//! Build with `--features bench,local` (or `bench,metal` / `bench,cuda`):
//!
//! ```sh
//! cargo run -p spnl-cli --features bench,metal -- bench pic -m llama3.2:3b -s xs,sm,m
//! cargo run -p spnl-cli --features bench,metal -- bench pic --full
//! ```
//!
//! ## Required environment variables
//!
//! The PIC path is activated by sentinel token IDs that delimit Plus/Cross
//! blocks in the tokenized sequence. Set these to the token IDs your model uses
//! for the Plus and Cross sentinel tokens:
//!
//! ```sh
//! export SPNL_PIC_PLUS_TOKEN=128011   # example: a reserved special token
//! export SPNL_PIC_CROSS_TOKEN=128012   # example: a reserved special token
//! ```
//!
//! Without these env vars, cross-request cache reuse will not activate. The
//! benchmark will still run, but reuse requests will show no speedup and the
//! hit rate will be 0%.
//!
//! ## Verifying PIC is active
//!
//! Run with `RUST_LOG=info` to see per-request cache hit messages:
//!
//! ```text
//! INFO ... PIC hit: N Plus blocks reused
//! ```
//!
//! # Protocol
//!
//! For each model, for each doc-length configuration, for each trial:
//! 1. Generate fresh synthetic documents (unique per trial to avoid inter-trial hits)
//! 2. **No-cache** request (`max_tokens=1`): first time these docs are seen — full prefill
//! 3. N **reuse** requests (`max_tokens=1`): same docs shuffled into different orders
//!    - With PIC active, Plus block KVs are reused; only Cross tokens need prefill

use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use spnl::{
    ExecuteOptions, SpnlError, execute,
    ir::{Message::Assistant, Query},
    spnl,
};
use std::time::Instant;

/// Doc-length spectrum for size sweeps.
/// Each entry is (words_per_doc, short_name).
const FULL_SPECTRUM: &[(usize, &str)] = &[
    (10, "xs"),
    (50, "sm"),
    (200, "m"),
    (500, "lg"),
    (1000, "xl"),
    (2000, "xxl"),
];

/// All valid t-shirt size names, for help text.
const ALL_SIZES: &str = "xs,sm,m,lg,xl,xxl";

/// Default models for `--full` mode.
const FULL_MODELS: &[&str] = &[
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.1:8b",
    "qwen2.5:0.5b",
    "qwen2.5:14b",
];

/// Resolve size names into spectrum entries.
fn resolve_spectrum(sizes: &[String]) -> Option<Vec<(usize, String)>> {
    if sizes.is_empty() {
        return None;
    }
    let selected: Vec<(usize, String)> = sizes
        .iter()
        .filter_map(|s| {
            let s = s.trim();
            FULL_SPECTRUM
                .iter()
                .find(|(_, name)| *name == s)
                .map(|&(len, name)| (len, name.to_string()))
        })
        .collect();
    if selected.is_empty() {
        None
    } else {
        Some(selected)
    }
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum OutputMode {
    /// Speedup ratio: "3.75x"
    Speedup,
    /// PIC interquartile range (ms): "90,150"
    Iqr,
    /// PIC cache hit rate: "100%"
    Hitrate,
    /// Prefix and PIC p50 latency (ms): "450,120"
    Latency,
    /// All data as JSON
    Json,
    /// Accuracy: Plus (PIC) vs flat (causal) response comparison
    Accuracy,
}

#[derive(clap::Args, Debug, serde::Serialize)]
pub struct PicArgs {
    /// Generative model(s): -m model1,model2 or -m model1 -m model2
    #[arg(short, long, value_delimiter = ',', env = "BENCH_MODEL")]
    pub model: Vec<String>,

    /// Document sizes to sweep (comma-separated): -s xs,sm,m,lg,xl,xxl
    /// Sizes: xs(10w) sm(50w) m(200w) lg(500w) xl(1000w) xxl(2000w)
    #[arg(short, long, value_delimiter = ',')]
    pub size: Vec<String>,

    /// Number of documents (Plus blocks)
    #[arg(long, default_value_t = 4, env = "BENCH_PIC_NUM_DOCS")]
    pub num_docs: usize,

    /// Length parameter (used when no --size is specified).
    /// For TTFT modes: approximate words per document (default: 200).
    /// For accuracy mode: max tokens per response (default: 200).
    #[arg(short, long, default_value_t = 200, env = "BENCH_PIC_LENGTH")]
    pub length: usize,

    /// Number of reuse (reshuffled) requests after the initial no-cache request
    #[arg(long, default_value_t = 5, env = "BENCH_PIC_REUSE_ITERS")]
    pub reuse_iters: usize,

    /// Number of full trials (no-cache + reuse cycle) per doc-length
    #[arg(long, default_value_t = 3, env = "BENCH_PIC_TRIALS")]
    pub trials: usize,

    /// Full sweep: all sizes × default models
    /// Equivalent to -s xs,sm,m,lg,xl,xxl -m llama3.2:1b,llama3.2:3b,llama3.1:8b,qwen2.5:0.5b,qwen2.5:14b
    #[arg(long)]
    pub full: bool,

    /// Output mode(s): -o speedup,accuracy or -o latency -o accuracy
    /// Modes: speedup, iqr, hitrate, latency, json, accuracy
    #[arg(
        short = 'o',
        long,
        value_delimiter = ',',
        value_enum,
        default_value = "speedup"
    )]
    pub output: Vec<OutputMode>,

    /// Grading model for LLM-judge semantic equivalence scoring.
    /// If omitted, only token F1 is reported (no LLM judge).
    #[arg(long, env = "BENCH_PIC_GRADING_MODEL")]
    pub grading_model: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_documents(num_docs: usize, doc_length: usize) -> Vec<String> {
    let mut rng = rand::rng();
    (0..num_docs)
        .map(|i| {
            format!(
                "Document {}: The topic of this document is item-{}. {}",
                i,
                i,
                lipsum::lipsum_words_with_rng(&mut rng, doc_length)
            )
        })
        .collect()
}

/// Build a query with Plus-wrapped documents (PIC block attention).
fn build_query(model: &str, docs: &[String], question: &str, max_tokens: i32) -> Query {
    let model = model.to_string();
    let system_prompt =
        "You are a helpful assistant. Answer based on the provided documents.".to_string();
    let temperature: f32 = 0.0;

    let doc_messages: Vec<Query> = docs
        .iter()
        .map(|text| {
            let text = text.clone();
            spnl!(user text)
        })
        .collect();

    let question = question.to_string();

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

/// Build a query with docs as plain sequential messages (standard causal attention, no Plus).
fn build_query_flat(model: &str, docs: &[String], question: &str, max_tokens: i32) -> Query {
    use spnl::ir::{Generate, GenerateMetadata};

    let system_prompt =
        "You are a helpful assistant. Answer based on the provided documents.".to_string();

    let mut messages: Vec<Query> = Vec::new();
    messages.push(spnl!(system system_prompt));
    for text in docs {
        let text = text.clone();
        messages.push(spnl!(user text));
    }
    let question = question.to_string();
    messages.push(spnl!(user question));

    Query::Generate(Generate {
        metadata: GenerateMetadata {
            model: model.to_string(),
            max_tokens: Some(max_tokens),
            temperature: Some(0.0),
        },
        input: Box::new(Query::Cross(messages)),
    })
}

async fn timed_request(query: &Query, options: &ExecuteOptions) -> anyhow::Result<(f64, String)> {
    let start = Instant::now();
    let result = execute(query, options).await?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let text = match result {
        Query::Message(Assistant(s)) => s,
        _ => String::new(),
    };
    Ok((ms, text))
}

fn shuffle_docs(docs: &[String]) -> Vec<String> {
    let mut shuffled = docs.to_vec();
    let mut rng = rand::rng();
    shuffled.shuffle(&mut rng);
    shuffled
}

fn take_pic_stats() -> (u64, u64) {
    #[cfg(feature = "local")]
    {
        spnl::pic_stats::take_cache_stats()
    }
    #[cfg(not(feature = "local"))]
    {
        (0, 0)
    }
}

async fn unload_models() {
    #[cfg(feature = "local")]
    {
        spnl::model_pool::unload_all().await;
    }
}

// ---------------------------------------------------------------------------
// Shared run context
// ---------------------------------------------------------------------------

pub(crate) struct RunCtx<'a> {
    pub(crate) model: &'a str,
    pub(crate) num_docs: usize,
    pub(crate) doc_length: usize,
    pub(crate) label: &'a str,
    pub(crate) pb: &'a ProgressBar,
    pub(crate) step_prefix: &'a str,
}

// ---------------------------------------------------------------------------
// Single doc-length run — returns collected results
// ---------------------------------------------------------------------------

pub(crate) struct RunResult {
    label: String,
    doc_length: usize,
    pub(crate) nocache_ttfts: Vec<f64>,
    pub(crate) reuse_ttfts: Vec<f64>,
    reuse_hits: u64,
    reuse_misses: u64,
}

pub(crate) async fn run_one(
    ctx: &RunCtx<'_>,
    reuse_iters: usize,
    trials: usize,
) -> anyhow::Result<RunResult> {
    let RunCtx {
        model,
        num_docs,
        doc_length,
        label,
        pb,
        step_prefix,
    } = ctx;
    let options = ExecuteOptions {
        silent: true,
        ..Default::default()
    };
    let question = "Summarize the key topics from all the documents.";

    let mut nocache_ttfts = Vec::new();
    let mut reuse_ttfts = Vec::new();
    let mut total_hits: u64 = 0;
    let mut total_misses: u64 = 0;

    let _ = take_pic_stats();

    for trial in 0..trials {
        let docs = make_documents(*num_docs, *doc_length);

        // No-cache request: first time these docs are seen — full prefill
        let _ = take_pic_stats();
        let nocache_query = build_query(model, &docs, question, 1);
        let (nocache_ms, _) = timed_request(&nocache_query, &options).await?;
        nocache_ttfts.push(nocache_ms);
        pb.inc(1);
        pb.set_message(format!(
            "{step_prefix}{label} · trial {}/{trials} · Prefix · {nocache_ms:.0}ms",
            trial + 1,
        ));

        // Reuse requests: same docs reshuffled — PIC should serve Plus blocks from cache
        let _ = take_pic_stats();
        for reuse_i in 0..reuse_iters {
            let shuffled = shuffle_docs(&docs);
            let reuse_query = build_query(model, &shuffled, question, 1);
            let (reuse_ms, _) = timed_request(&reuse_query, &options).await?;
            reuse_ttfts.push(reuse_ms);
            pb.inc(1);
            pb.set_message(format!(
                "{step_prefix}{label} · trial {}/{trials} · PIC {}/{reuse_iters} · {reuse_ms:.0}ms",
                trial + 1,
                reuse_i + 1,
            ));
        }

        let (hits, misses) = take_pic_stats();
        total_hits += hits;
        total_misses += misses;
    }

    Ok(RunResult {
        label: label.to_string(),
        doc_length: *doc_length,
        nocache_ttfts,
        reuse_ttfts,
        reuse_hits: total_hits,
        reuse_misses: total_misses,
    })
}

// ---------------------------------------------------------------------------
// Accuracy trials — compare Plus (PIC attention) vs flat (standard causal)
// ---------------------------------------------------------------------------

use std::collections::HashMap;

struct AccuracyResult {
    label: String,
    doc_length: usize,
    trials: Vec<AccuracyTrial>,
}

struct AccuracyTrial {
    token_f1: f64,
    /// LLM-judge semantic equivalence score (0-100), None if no grading model
    llm_score: Option<f64>,
}

impl AccuracyResult {
    fn avg_token_f1(&self) -> f64 {
        if self.trials.is_empty() {
            return 0.0;
        }
        self.trials.iter().map(|t| t.token_f1).sum::<f64>() / self.trials.len() as f64
    }

    fn avg_llm_score(&self) -> Option<f64> {
        let scores: Vec<f64> = self.trials.iter().filter_map(|t| t.llm_score).collect();
        if scores.is_empty() {
            return None;
        }
        Some(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Normalize text into lowercase word tokens for comparison.
fn normalize_tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Token-level F1 between two texts (0-100). Measures word overlap regardless of order.
fn token_f1(reference: &str, candidate: &str) -> f64 {
    let ref_tokens = normalize_tokens(reference);
    let cand_tokens = normalize_tokens(candidate);
    if ref_tokens.is_empty() && cand_tokens.is_empty() {
        return 100.0;
    }
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return 0.0;
    }

    let ref_counts: HashMap<&str, usize> = ref_tokens.iter().fold(HashMap::new(), |mut m, t| {
        *m.entry(t.as_str()).or_insert(0) += 1;
        m
    });
    let cand_counts: HashMap<&str, usize> = cand_tokens.iter().fold(HashMap::new(), |mut m, t| {
        *m.entry(t.as_str()).or_insert(0) += 1;
        m
    });

    let mut common = 0usize;
    for (tok, &count) in &cand_counts {
        common += count.min(*ref_counts.get(tok).unwrap_or(&0));
    }

    if common == 0 {
        return 0.0;
    }

    let precision = common as f64 / cand_tokens.len() as f64;
    let recall = common as f64 / ref_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall) * 100.0
}

/// Build an LLM-judge query that scores semantic equivalence of two responses.
fn build_equivalence_query(model: &str, reference: &str, candidate: &str) -> Query {
    use spnl::ir::{Generate, GenerateMetadata};

    let system_prompt = "You are a semantic equivalence evaluator. You will be given two responses to the same question with the same documents. Determine whether they convey the same meaning. Return ONLY a single integer 0-100. 100 means semantically identical, 0 means completely different meaning.".to_string();
    let user_prompt = format!(
        "Response A (reference):\n{reference}\n\nResponse B (candidate):\n{candidate}\n\nSemantic equivalence score (0-100):"
    );

    Query::Generate(Generate {
        metadata: GenerateMetadata {
            model: model.to_string(),
            max_tokens: Some(16),
            temperature: Some(0.0),
        },
        input: Box::new(Query::Cross(vec![
            spnl!(system system_prompt),
            spnl!(user user_prompt),
        ])),
    })
}

/// Parse a numeric score from an LLM response.
fn parse_score(response: &str) -> f64 {
    response
        .trim()
        .split(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 100.0))
        .unwrap_or(0.0)
}

async fn run_accuracy(
    ctx: &RunCtx<'_>,
    grading_model: Option<&str>,
    accuracy_tokens: i32,
    trials: usize,
) -> anyhow::Result<AccuracyResult> {
    let RunCtx {
        model,
        num_docs,
        doc_length,
        label,
        pb,
        step_prefix,
    } = ctx;
    let options = ExecuteOptions {
        silent: true,
        ..Default::default()
    };
    let question = "Summarize the key topics from all the documents.";

    let mut trial_results = Vec::new();

    for trial in 0..trials {
        let docs = make_documents(*num_docs, *doc_length);

        // Flat query: standard causal attention (no Plus)
        let flat_query = build_query_flat(model, &docs, question, accuracy_tokens);
        let (_, flat_response) = timed_request(&flat_query, &options).await?;
        pb.inc(1);
        pb.set_message(format!(
            "{step_prefix}{label} · accuracy {}/{trials} · Prefix · {} chars",
            trial + 1,
            flat_response.len(),
        ));

        // Plus query: PIC block attention
        let _ = take_pic_stats(); // reset stats
        let plus_query = build_query(model, &docs, question, accuracy_tokens);
        let (_, plus_response) = timed_request(&plus_query, &options).await?;
        pb.inc(1);
        pb.set_message(format!(
            "{step_prefix}{label} · accuracy {}/{trials} · PIC · {} chars",
            trial + 1,
            plus_response.len(),
        ));

        // Token F1 (cheap, no LLM call)
        let f1 = token_f1(&flat_response, &plus_response);

        // LLM-judge (if grading model provided)
        let llm_score = if let Some(gm) = grading_model {
            let judge_query = build_equivalence_query(gm, &flat_response, &plus_response);
            match execute(&judge_query, &options).await {
                Ok(Query::Message(Assistant(s))) => {
                    pb.inc(1);
                    Some(parse_score(&s))
                }
                _ => {
                    pb.inc(1);
                    None
                }
            }
        } else {
            None
        };

        let score_str = llm_score
            .map(|s| format!(" llm={s:.0}"))
            .unwrap_or_default();
        pb.set_message(format!(
            "{step_prefix}{label} · accuracy {}/{trials} · f1={f1:.0}{score_str}",
            trial + 1,
        ));

        trial_results.push(AccuracyTrial {
            token_f1: f1,
            llm_score,
        });
    }

    Ok(AccuracyResult {
        label: label.to_string(),
        doc_length: *doc_length,
        trials: trial_results,
    })
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(args: PicArgs) -> Result<(), SpnlError> {
    // --- Resolve models and sizes (--full provides defaults) ---
    let mut models: Vec<String> = if !args.model.is_empty() {
        args.model.clone()
    } else if args.full {
        FULL_MODELS.iter().map(|s| s.to_string()).collect()
    } else {
        anyhow::bail!("At least one model is required (-m MODEL)");
    };
    // Deduplicate while preserving order
    {
        let mut seen = std::collections::HashSet::new();
        models.retain(|m| seen.insert(m.clone()));
    }

    let spectrum: Vec<(usize, String)> = if !args.size.is_empty() {
        resolve_spectrum(&args.size).ok_or_else(|| {
            anyhow::anyhow!(
                "No valid sizes in --size={}. Valid sizes: {}",
                args.size.join(","),
                ALL_SIZES
            )
        })?
    } else if args.full {
        FULL_SPECTRUM
            .iter()
            .map(|&(len, name)| (len, name.to_string()))
            .collect()
    } else {
        vec![(args.length, format!("{}w", args.length))]
    };

    // --- Prerequisite checks ---
    for model in &models {
        let is_remote = model.starts_with("ollama/")
            || model.starts_with("openai/")
            || model.starts_with("gemini/")
            || model.starts_with("spnl/");
        if is_remote {
            eprintln!("WARNING: Model '{}' uses a remote backend.", model);
            eprintln!(
                "         PIC cross-request caching only works with the local mistral.rs backend."
            );
            eprintln!();
        }
    }

    // --- Derive which protocols to run ---
    let wants_ttft = args.output.iter().any(|m| {
        matches!(
            m,
            OutputMode::Speedup
                | OutputMode::Iqr
                | OutputMode::Hitrate
                | OutputMode::Latency
                | OutputMode::Json
        )
    });
    let wants_accuracy = args.output.contains(&OutputMode::Accuracy);

    // --- Print config ---
    eprintln!("=== PIC Cross-Request Cache Benchmark ===");
    if models.len() == 1 {
        eprintln!("  Model:       {}", models[0]);
    } else {
        eprintln!("  Models:      {}", models.join(", "));
    }
    eprintln!("  Documents:   {} per request", args.num_docs);
    let size_names: Vec<String> = spectrum
        .iter()
        .map(|(len, name)| format!("{name}({len}w)"))
        .collect();
    eprintln!("  Doc sizes:   {}", size_names.join(", "));
    let output_labels: Vec<&str> = args
        .output
        .iter()
        .map(|m| match m {
            OutputMode::Speedup => "speedup",
            OutputMode::Iqr => "iqr (p25,p75 ms)",
            OutputMode::Hitrate => "hitrate",
            OutputMode::Latency => "latency (prefix,pic ms)",
            OutputMode::Json => "json",
            OutputMode::Accuracy => "accuracy",
        })
        .collect();
    eprintln!("  Output:      {}", output_labels.join(", "));
    if wants_ttft {
        eprintln!("  Reuse iters: {} per trial", args.reuse_iters);
        eprintln!("  Trials:      {} per doc-length", args.trials);
        eprintln!("  Max tokens:  1 (TTFT measurement)");
    }
    if wants_accuracy {
        let metric = match args.grading_model.as_deref() {
            Some(m) => format!("metric=token-f1+llm-judge({m})"),
            None => "metric=token-f1, llm-judge disabled (see --grading-model)".to_string(),
        };
        eprintln!(
            "  Accuracy:    {} tokens, {} trials, {metric}",
            args.length, args.trials
        );
    }
    eprintln!();

    // --- Progress bar ---
    let ttft_steps_per_len = if wants_ttft {
        args.trials * (1 + args.reuse_iters)
    } else {
        0
    };
    let accuracy_steps_per_trial = if args.grading_model.is_some() { 3 } else { 2 }; // flat + PIC + optional judge
    let accuracy_steps_per_len = if wants_accuracy {
        args.trials * accuracy_steps_per_trial
    } else {
        0
    };
    let total_steps =
        (models.len() * spectrum.len() * (ttft_steps_per_len + accuracy_steps_per_len)) as u64;
    let pb = ProgressBar::new(total_steps);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:30.cyan/dim} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("━╸─"),
    );

    // --- Run all models × sizes ---
    let mut all_model_results: Vec<(String, Vec<RunResult>)> = Vec::new();
    let mut all_accuracy_results: Vec<(String, Vec<AccuracyResult>)> = Vec::new();

    for (mi, model) in models.iter().enumerate() {
        // Unload previous model to free GPU memory before loading the next one
        if mi > 0 {
            unload_models().await;
        }

        let mut results = Vec::new();
        let mut accuracy_results = Vec::new();
        for (si, (doc_length, label)) in spectrum.iter().enumerate() {
            let step_prefix = match (models.len() > 1, spectrum.len() > 1) {
                (true, true) => format!("{} [{}/{}] ", model, si + 1, spectrum.len()),
                (true, false) => format!("{} ", model),
                (false, true) => format!("[{}/{}] ", si + 1, spectrum.len()),
                (false, false) => String::new(),
            };
            let ctx = RunCtx {
                model,
                num_docs: args.num_docs,
                doc_length: *doc_length,
                label,
                pb: &pb,
                step_prefix: &step_prefix,
            };

            if wants_ttft {
                let r = run_one(&ctx, args.reuse_iters, args.trials).await?;
                results.push(r);
            }

            if wants_accuracy {
                let ar = run_accuracy(
                    &ctx,
                    args.grading_model.as_deref(),
                    args.length as i32,
                    args.trials,
                )
                .await?;
                accuracy_results.push(ar);
            }
        }
        if wants_ttft {
            all_model_results.push((model.clone(), results));
        }
        if wants_accuracy {
            all_accuracy_results.push((model.clone(), accuracy_results));
        }
    }

    pb.finish_and_clear();
    eprintln!();

    // --- Report ---
    if !all_model_results.is_empty() {
        // Find the first TTFT output mode for the table (Json is handled separately)
        let ttft_modes: Vec<&OutputMode> = args
            .output
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    OutputMode::Speedup
                        | OutputMode::Iqr
                        | OutputMode::Hitrate
                        | OutputMode::Latency
                )
            })
            .collect();
        if args.output.contains(&OutputMode::Json) {
            print_results_json(&all_model_results);
        }
        for mode in ttft_modes {
            print_results_table(&all_model_results, mode);
        }
    }

    // --- Accuracy report ---
    if !all_accuracy_results.is_empty() {
        print_accuracy_table(&all_accuracy_results);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Results table (single unified output for all modes)
// ---------------------------------------------------------------------------

fn format_cell(r: &RunResult, output: &OutputMode) -> String {
    match output {
        OutputMode::Speedup => {
            let nocache_p50 = percentile(&r.nocache_ttfts, 50);
            let reuse_p50 = percentile(&r.reuse_ttfts, 50);
            let speedup = if reuse_p50 > 0.0 {
                nocache_p50 / reuse_p50
            } else {
                0.0
            };
            format!("{speedup:.2}x")
        }
        OutputMode::Iqr => {
            let p25 = percentile(&r.reuse_ttfts, 25);
            let p75 = percentile(&r.reuse_ttfts, 75);
            format!("{p25:.0},{p75:.0}")
        }
        OutputMode::Hitrate => {
            let (rate, _) = compute_hit_rate(r.reuse_hits, r.reuse_misses);
            format!("{rate:.0}%")
        }
        OutputMode::Latency => {
            let nocache_p50 = percentile(&r.nocache_ttfts, 50);
            let reuse_p50 = percentile(&r.reuse_ttfts, 50);
            format!("{nocache_p50:.0},{reuse_p50:.0}")
        }
        OutputMode::Json | OutputMode::Accuracy => unreachable!(),
    }
}

fn print_results_table(all_model_results: &[(String, Vec<RunResult>)], output: &OutputMode) {
    // Column labels from the first model's results
    let size_labels: Vec<String> = all_model_results[0]
        .1
        .iter()
        .map(|r| {
            if r.label.ends_with('w') {
                r.label.clone()
            } else {
                format!("{} {}w", r.label, r.doc_length)
            }
        })
        .collect();

    // Pre-format all cells to compute column widths
    let rows: Vec<(&str, Vec<String>)> = all_model_results
        .iter()
        .map(|(model, results)| {
            let cells: Vec<String> = results.iter().map(|r| format_cell(r, output)).collect();
            (model.as_str(), cells)
        })
        .collect();

    let model_w = rows.iter().map(|(m, _)| m.len()).max().unwrap_or(5).max(5);
    let col_ws: Vec<usize> = (0..size_labels.len())
        .map(|i| {
            let header_w = size_labels[i].len();
            let data_w = rows
                .iter()
                .map(|(_, cells)| cells[i].len())
                .max()
                .unwrap_or(0);
            header_w.max(data_w).max(6)
        })
        .collect();

    // Header
    let mut header = format!(" {:<w$}", "Model", w = model_w);
    let mut sep = format!(" {:<w$}", "─".repeat(model_w), w = model_w);
    for (i, label) in size_labels.iter().enumerate() {
        header.push_str(&format!("  {:>w$}", label, w = col_ws[i]));
        sep.push_str(&format!("  {:>w$}", "─".repeat(col_ws[i]), w = col_ws[i]));
    }
    eprintln!("{header}");
    eprintln!("{sep}");

    // Data rows
    for (model, cells) in &rows {
        let mut row = format!(" {:<w$}", model, w = model_w);
        for (i, cell) in cells.iter().enumerate() {
            row.push_str(&format!("  {:>w$}", cell, w = col_ws[i]));
        }
        eprintln!("{row}");
    }

    eprintln!();
}

fn print_results_json(all_model_results: &[(String, Vec<RunResult>)]) {
    let results: Vec<serde_json::Value> = all_model_results
        .iter()
        .flat_map(|(model, runs)| {
            runs.iter().map(move |r| {
                let nocache_p50 = percentile(&r.nocache_ttfts, 50);
                let reuse_p50 = percentile(&r.reuse_ttfts, 50);
                let speedup = if reuse_p50 > 0.0 {
                    nocache_p50 / reuse_p50
                } else {
                    0.0
                };
                let (hit_rate, _) = compute_hit_rate(r.reuse_hits, r.reuse_misses);
                serde_json::json!({
                    "model": model,
                    "size": r.label,
                    "doc_length": r.doc_length,
                    "speedup": (speedup * 100.0).round() / 100.0,
                    "prefix_p50_ms": (nocache_p50 * 100.0).round() / 100.0,
                    "pic_p50_ms": (reuse_p50 * 100.0).round() / 100.0,
                    "pic_p25_ms": (percentile(&r.reuse_ttfts, 25) * 100.0).round() / 100.0,
                    "pic_p75_ms": (percentile(&r.reuse_ttfts, 75) * 100.0).round() / 100.0,
                    "hit_rate": (hit_rate * 100.0).round() / 100.0,
                    "hits": r.reuse_hits,
                    "misses": r.reuse_misses,
                    "prefix_ttfts_ms": r.nocache_ttfts,
                    "pic_ttfts_ms": r.reuse_ttfts,
                })
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}

// ---------------------------------------------------------------------------
// Accuracy table (same layout as TTFT tables: rows=models, columns=doc sizes)
// ---------------------------------------------------------------------------

fn format_accuracy_cell(r: &AccuracyResult) -> String {
    let f1 = r.avg_token_f1();
    match r.avg_llm_score() {
        Some(llm) => format!("{f1:.0}%,{llm:.0}"),
        None => format!("{f1:.0}%"),
    }
}

fn print_accuracy_table(all: &[(String, Vec<AccuracyResult>)]) {
    let size_labels: Vec<String> = all[0]
        .1
        .iter()
        .map(|r| {
            if r.label.ends_with('w') {
                r.label.clone()
            } else {
                format!("{} {}w", r.label, r.doc_length)
            }
        })
        .collect();

    let rows: Vec<(&str, Vec<String>)> = all
        .iter()
        .map(|(model, results)| {
            let cells: Vec<String> = results.iter().map(format_accuracy_cell).collect();
            (model.as_str(), cells)
        })
        .collect();

    let model_w = rows.iter().map(|(m, _)| m.len()).max().unwrap_or(5).max(5);
    let col_ws: Vec<usize> = (0..size_labels.len())
        .map(|i| {
            let header_w = size_labels[i].len();
            let data_w = rows
                .iter()
                .map(|(_, cells)| cells[i].len())
                .max()
                .unwrap_or(0);
            header_w.max(data_w).max(6)
        })
        .collect();

    // Header
    let mut header = format!(" {:<w$}", "Model", w = model_w);
    let mut sep = format!(" {:<w$}", "─".repeat(model_w), w = model_w);
    for (i, label) in size_labels.iter().enumerate() {
        header.push_str(&format!("  {:>w$}", label, w = col_ws[i]));
        sep.push_str(&format!("  {:>w$}", "─".repeat(col_ws[i]), w = col_ws[i]));
    }
    eprintln!("{header}");
    eprintln!("{sep}");

    // Data rows
    for (model, cells) in &rows {
        let mut row = format!(" {:<w$}", model, w = model_w);
        for (i, cell) in cells.iter().enumerate() {
            row.push_str(&format!("  {:>w$}", cell, w = col_ws[i]));
        }
        eprintln!("{row}");
    }

    eprintln!();
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn compute_hit_rate(hits: u64, misses: u64) -> (f64, u64) {
    let total = hits + misses;
    let rate = if total > 0 {
        hits as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    (rate, total)
}

fn percentile(values: &[f64], pct: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (sorted.len() * pct / 100).min(sorted.len() - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- token_f1 ----

    #[test]
    fn token_f1_identical_texts() {
        assert_eq!(token_f1("hello world", "hello world"), 100.0);
    }

    #[test]
    fn token_f1_completely_different() {
        assert_eq!(token_f1("hello world", "foo bar"), 0.0);
    }

    #[test]
    fn token_f1_partial_overlap() {
        // "the cat sat" vs "the cat ran": shared = {the:1, cat:1} = 2
        // precision = 2/3, recall = 2/3, f1 = 2/3 * 100 ≈ 66.67
        let f1 = token_f1("the cat sat", "the cat ran");
        assert!((f1 - 66.67).abs() < 0.1, "got {f1}");
    }

    #[test]
    fn token_f1_both_empty() {
        assert_eq!(token_f1("", ""), 100.0);
    }

    #[test]
    fn token_f1_one_empty() {
        assert_eq!(token_f1("hello", ""), 0.0);
        assert_eq!(token_f1("", "hello"), 0.0);
    }

    // ---- normalize_tokens ----

    #[test]
    fn normalize_tokens_basic() {
        assert_eq!(normalize_tokens("Hello World!"), vec!["hello", "world"]);
    }

    #[test]
    fn normalize_tokens_punctuation() {
        assert_eq!(normalize_tokens("foo-bar, baz."), vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn normalize_tokens_empty() {
        let empty: Vec<String> = vec![];
        assert_eq!(normalize_tokens(""), empty);
    }

    // ---- parse_score ----

    #[test]
    fn parse_score_plain_number() {
        assert_eq!(parse_score("85"), 85.0);
    }

    #[test]
    fn parse_score_with_text() {
        assert_eq!(parse_score("Score: 92/100"), 92.0);
    }

    #[test]
    fn parse_score_garbage() {
        assert_eq!(parse_score("no numbers"), 0.0);
    }

    #[test]
    fn parse_score_clamped() {
        assert_eq!(parse_score("150"), 100.0);
    }

    // ---- resolve_spectrum ----

    #[test]
    fn resolve_spectrum_valid_sizes() {
        let sizes: Vec<String> = vec!["xs".into(), "m".into(), "xxl".into()];
        let result = resolve_spectrum(&sizes).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], (10, "xs".to_string()));
        assert_eq!(result[1], (200, "m".to_string()));
        assert_eq!(result[2], (2000, "xxl".to_string()));
    }

    #[test]
    fn resolve_spectrum_invalid_sizes() {
        let sizes: Vec<String> = vec!["bogus".into()];
        assert!(resolve_spectrum(&sizes).is_none());
    }

    #[test]
    fn resolve_spectrum_empty() {
        let sizes: Vec<String> = vec![];
        assert!(resolve_spectrum(&sizes).is_none());
    }

    #[test]
    fn resolve_spectrum_mixed() {
        let sizes: Vec<String> = vec!["xs".into(), "bogus".into(), "m".into()];
        let result = resolve_spectrum(&sizes).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (10, "xs".to_string()));
        assert_eq!(result[1], (200, "m".to_string()));
    }

    // ---- compute_hit_rate ----

    #[test]
    fn compute_hit_rate_zero_total() {
        assert_eq!(compute_hit_rate(0, 0), (0.0, 0));
    }

    #[test]
    fn compute_hit_rate_all_hits() {
        assert_eq!(compute_hit_rate(10, 0), (100.0, 10));
    }

    #[test]
    fn compute_hit_rate_half_and_half() {
        assert_eq!(compute_hit_rate(5, 5), (50.0, 10));
    }

    // ---- percentile ----

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 50), 0.0);
    }

    #[test]
    fn percentile_single() {
        assert_eq!(percentile(&[42.0], 50), 42.0);
    }

    #[test]
    fn percentile_known_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 50), 3.0);
        assert_eq!(percentile(&data, 25), 2.0);
    }

    // ---- integration test: benchmark produces speedup > 1 ----

    #[cfg(all(feature = "bench", feature = "local"))]
    #[tokio::test]
    #[ignore] // requires local model and GPU
    async fn pic_benchmark_shows_speedup() {
        let model = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "llama3.2:1b".to_string());
        let pb = ProgressBar::hidden();
        let ctx = RunCtx {
            model: &model,
            num_docs: 2,
            doc_length: 50,
            label: "test",
            pb: &pb,
            step_prefix: "",
        };
        let result = run_one(&ctx, 3, 1).await.expect("benchmark should succeed");
        let nocache_p50 = percentile(&result.nocache_ttfts, 50);
        let reuse_p50 = percentile(&result.reuse_ttfts, 50);
        let speedup = nocache_p50 / reuse_p50;
        assert!(
            speedup > 1.0,
            "Expected PIC speedup > 1.0, got {speedup:.2}x"
        );
    }
}

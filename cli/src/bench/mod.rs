mod haystack;
mod niah;
mod pic;
mod ragcsv;
mod ruler;

use clap::Subcommand;
use spnl::SpnlError;

#[derive(Subcommand, Debug, serde::Serialize)]
pub enum BenchCommands {
    /// Multi-document haystack benchmark (precision/recall)
    Haystack(haystack::HaystackArgs),
    /// Needle-in-a-haystack benchmark with token-based context
    Niah(niah::NiahArgs),
    /// RULER benchmark (NIAH + variable tracking)
    Ruler(ruler::RulerArgs),
    /// RAG CSV evaluation (accuracy grading from a CSV dataset)
    Ragcsv(ragcsv::RagcsvArgs),
    /// PIC cross-request cache reuse benchmark (latency)
    Pic(pic::PicArgs),
}

pub async fn run(command: BenchCommands) -> Result<(), SpnlError> {
    match command {
        BenchCommands::Haystack(args) => {
            Ok(tokio::task::spawn_blocking(move || haystack::run(args)).await??)
        }
        BenchCommands::Niah(args) => {
            Ok(tokio::task::spawn_blocking(move || niah::run(args)).await??)
        }
        BenchCommands::Ruler(args) => {
            Ok(tokio::task::spawn_blocking(move || ruler::run(args)).await??)
        }
        BenchCommands::Ragcsv(args) => ragcsv::run(args).await,
        BenchCommands::Pic(args) => pic::run(args).await,
    }
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;

pub fn create_benchmark_progress(_total: u64, message: impl Into<String>) -> Arc<ProgressBar> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("[{elapsed_precise}] {spinner:.cyan} [{pos}] {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.set_message(message.into());
    Arc::new(pb)
}

pub fn finish_benchmark_progress(pb: &ProgressBar, message: impl Into<String>) {
    pb.finish_with_message(message.into());
}

pub fn compute_quantiles(values: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
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
        sorted[(len * 99 / 100).min(len - 1)],
        sorted[len - 1],
    )
}

/// Extended quantiles including average (used by ragcsv)
pub fn compute_quantiles_with_avg(values: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(values);
    let avg = if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    (min, p25, p50, p75, p90, p99, max, avg)
}

// ---------------------------------------------------------------------------
// Shared essay fetching (used by niah and ruler)
// ---------------------------------------------------------------------------

use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct GitHubFile {
    name: String,
    download_url: String,
}

const PG_ESSAYS_API_URL: &str = "https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/contents/needlehaystack/PaulGrahamEssays";

const FALLBACK_ESSAYS: &str = r#"The way to get startup ideas is not to try to think of startup ideas. It's to look for problems, preferably problems you have yourself. The very best startup ideas tend to have three things in common: they're something the founders themselves want, that they themselves can build, and that few others realize are worth doing. Microsoft, Apple, Yahoo, Google, and Facebook all began this way.

One of the biggest things holding people back from doing great work is the fear of making something lame. And this fear is not an irrational one. Many things that are new are bad. But the way to get good ideas is to get lots of ideas. The way to get lots of ideas is to lower your standards. If you don't lower your standards, you won't get any ideas at all.

The most important quality in a startup founder is determination. Not intelligence—determination. This is a little depressing. It would be nice if intelligence were the most important quality, since that's what we're usually judged by. But determination is more important, because intelligence without determination is like a car without an engine."#;

pub fn get_cache_dir(bench_name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir()
        .ok_or("Could not determine cache directory")?
        .join("spnl")
        .join(bench_name);
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

pub fn fetch_pg_essays(bench_name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let cache_file = get_cache_dir(bench_name)?.join("paul_graham_essays_combined.txt");

    if cache_file.exists() {
        return Ok(fs::read_to_string(&cache_file)?);
    }

    eprintln!("Downloading Paul Graham essays from GitHub...");
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(PG_ESSAYS_API_URL)
        .header("User-Agent", "spnl-benchmark")
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
            .header("User-Agent", "spnl-benchmark")
            .send()?
            .text()?;
        combined_content.push_str(&essay_content);
        combined_content.push('\n');
    }
    eprintln!("\nDownload complete!");

    fs::File::create(&cache_file)?.write_all(combined_content.as_bytes())?;
    Ok(combined_content)
}

// ---------------------------------------------------------------------------
// Shared tokenizer utilities (used by niah and ruler)
// ---------------------------------------------------------------------------

use tokenizers::Tokenizer;

pub fn get_context_length_in_tokens(context: &str, tokenizer: &Tokenizer) -> usize {
    tokenizer
        .encode(context, false)
        .map(|encoding| encoding.get_ids().len())
        .unwrap_or(0)
}

pub fn encode_and_trim(
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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- compute_quantiles ----

    #[test]
    fn quantiles_single_value() {
        let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&[42.0]);
        assert_eq!(min, 42.0);
        assert_eq!(p25, 42.0);
        assert_eq!(p50, 42.0);
        assert_eq!(p75, 42.0);
        assert_eq!(p90, 42.0);
        assert_eq!(p99, 42.0);
        assert_eq!(max, 42.0);
    }

    #[test]
    fn quantiles_multiple_values() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&values);
        assert_eq!(min, 1.0);
        assert_eq!(p25, 26.0);
        assert_eq!(p50, 51.0);
        assert_eq!(p75, 76.0);
        assert_eq!(p90, 91.0);
        assert_eq!(p99, 100.0);
        assert_eq!(max, 100.0);
    }

    #[test]
    fn quantiles_unsorted_input_produces_sorted_output() {
        let values = vec![5.0, 1.0, 3.0, 4.0, 2.0];
        let (min, _, _, _, _, _, max) = compute_quantiles(&values);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn quantiles_empty_returns_zeros() {
        let (min, p25, p50, p75, p90, p99, max) = compute_quantiles(&[]);
        assert_eq!(
            (min, p25, p50, p75, p90, p99, max),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        );
    }

    // ---- compute_quantiles_with_avg ----

    #[test]
    fn quantiles_with_avg_computes_average() {
        let values = vec![10.0, 20.0, 30.0];
        let (_, _, _, _, _, _, _, avg) = compute_quantiles_with_avg(&values);
        assert!((avg - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn quantiles_with_avg_empty_returns_zeros() {
        let (min, p25, p50, p75, p90, p99, max, avg) = compute_quantiles_with_avg(&[]);
        assert_eq!(
            (min, p25, p50, p75, p90, p99, max, avg),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        );
    }

    // ---- clap arg-parsing smoke tests ----

    use crate::args::FullArgs;
    use clap::Parser;

    #[test]
    fn parse_bench_ruler_defaults() {
        FullArgs::try_parse_from(["spnl", "bench", "ruler"]).unwrap();
    }

    #[test]
    fn parse_bench_niah_defaults() {
        FullArgs::try_parse_from(["spnl", "bench", "niah"]).unwrap();
    }

    #[test]
    fn parse_bench_haystack_defaults() {
        FullArgs::try_parse_from(["spnl", "bench", "haystack"]).unwrap();
    }

    #[test]
    fn parse_bench_ruler_csv_context_lengths() {
        let args = FullArgs::try_parse_from([
            "spnl",
            "bench",
            "ruler",
            "--context-lengths",
            "1000,2000,4000",
        ])
        .unwrap();
        match args.command {
            crate::args::Commands::Bench {
                command: BenchCommands::Ruler(r),
            } => assert_eq!(r.context_lengths, vec![1000, 2000, 4000]),
            other => panic!("unexpected command: {:?}", other),
        }
    }

    #[test]
    fn parse_bench_niah_csv_depth_percentages() {
        let args =
            FullArgs::try_parse_from(["spnl", "bench", "niah", "--depth-percentages", "0,50,100"])
                .unwrap();
        match args.command {
            crate::args::Commands::Bench {
                command: BenchCommands::Niah(n),
            } => assert_eq!(n.depth_percentages, vec![0, 50, 100]),
            other => panic!("unexpected command: {:?}", other),
        }
    }
}

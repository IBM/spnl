//! Reusable progress bar utilities for benchmarks
//!
//! This module provides a simple wrapper around indicatif to show
//! progress during long-running Criterion benchmarks.

use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;

/// Creates a progress bar for benchmark iterations
///
/// # Arguments
/// * `total` - Total number of iterations (typically the sample_size)
/// * `message` - Description of what's being benchmarked
///
/// # Returns
/// An Arc-wrapped ProgressBar that can be cloned and shared across async tasks
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
/// Update progress bar message with running statistics
///
/// # Arguments
/// * `pb` - The progress bar to update
/// * `base_msg` - Base message (e.g., "basic docs=2")
/// * `avg_precision` - Average precision so far
/// * `avg_recall` - Average recall so far
/// * `high_precision_count` - Count of runs with precision >= 0.75
/// * `high_recall_count` - Count of runs with recall >= 0.75
/// * `total_count` - Total number of samples collected so far
pub fn update_progress_with_stats(
    pb: &ProgressBar,
    base_msg: &str,
    avg_precision: f64,
    avg_recall: f64,
    high_precision_count: usize,
    high_recall_count: usize,
    total_count: usize,
) {
    pb.set_message(format!(
        "{} \x1b[1m|\x1b[0m n={} \x1b[1m|\x1b[0m P={:.1}% n≥75%={} \x1b[1m|\x1b[0m R={:.1}% n≥75%={}",
        base_msg,
        total_count,
        avg_precision * 100.0,
        high_precision_count,
        avg_recall * 100.0,
        high_recall_count
    ));
}

/// Finish a progress bar with a completion message
pub fn finish_benchmark_progress(pb: &ProgressBar, message: impl Into<String>) {
    pb.finish_with_message(message.into());
}

// Made with Bob

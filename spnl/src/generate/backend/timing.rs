//! Shared timing metrics printing for all backends

use std::time::Duration;
use tabled::{Table, Tabled, settings::Style};

/// Timing data for a single task
#[derive(Debug, Clone)]
pub struct TaskTiming {
    pub ttft: Option<Duration>,
    pub total_duration: Duration,
    pub token_count: u64,
}

/// Print timing metrics for one or more tasks
pub fn print_timing_metrics(tasks: &[TaskTiming]) {
    use std::io::IsTerminal;

    let is_tty = std::io::stderr().is_terminal();

    if is_tty {
        // Use tabled for nice formatting when output is a TTY and we have multiple tasks
        #[derive(Tabled)]
        struct TimingRow {
            #[tabled(rename = "Task")]
            task: usize,
            #[tabled(rename = "TTFT (ms)")]
            ttft: String,
            #[tabled(rename = "ITL (ms/token)")]
            itl: String,
            #[tabled(rename = "Total (ms)")]
            total: String,
            #[tabled(rename = "Tokens")]
            tokens: u64,
        }

        let mut rows = Vec::new();
        for (i, task) in tasks.iter().enumerate() {
            let ttft_str = if let Some(ttft_duration) = task.ttft {
                format!("{:.2}", ttft_duration.as_secs_f64() * 1000.0)
            } else {
                "N/A".to_string()
            };

            let itl_str = if task.token_count > 1 {
                if let Some(ttft_duration) = task.ttft {
                    let time_after_first =
                        task.total_duration.as_secs_f64() - ttft_duration.as_secs_f64();
                    let itl = time_after_first / (task.token_count - 1) as f64;
                    format!("{:.2}", itl * 1000.0)
                } else {
                    "N/A".to_string()
                }
            } else {
                "N/A".to_string()
            };

            rows.push(TimingRow {
                task: i + 1,
                ttft: ttft_str,
                itl: itl_str,
                total: format!("{:.2}", task.total_duration.as_secs_f64() * 1000.0),
                tokens: task.token_count,
            });
        }

        let table = Table::new(rows).with(Style::sharp()).to_string();
        eprintln!("{}", table);
    } else {
        // Plain ASCII output for non-TTY or single task
        print_plain(tasks);
    }
}

fn print_plain(tasks: &[TaskTiming]) {
    for (i, task) in tasks.iter().enumerate() {
        if tasks.len() > 1 {
            eprintln!("Task {}:", i + 1);
        }

        if let Some(ttft_duration) = task.ttft {
            eprintln!("TTFT: {:.2}ms", ttft_duration.as_secs_f64() * 1000.0);

            // Calculate ITL (Inter-Token Latency) - time after first token divided by remaining tokens
            if task.token_count > 1 {
                let time_after_first = task.total_duration - ttft_duration;
                let itl = time_after_first.as_secs_f64() / (task.token_count - 1) as f64;
                eprintln!("ITL: {:.2}ms/token", itl * 1000.0);
            }
        }

        eprintln!(
            "Total time: {:.2}ms",
            task.total_duration.as_secs_f64() * 1000.0
        );
        eprintln!("Tokens: {}", task.token_count);

        if tasks.len() > 1 && i < tasks.len() - 1 {
            eprintln!();
        }
    }
}

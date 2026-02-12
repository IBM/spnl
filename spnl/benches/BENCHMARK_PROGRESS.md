# Benchmark Progress Bar Support

This document explains the progress bar and silent mode features for SPNL benchmarks.

## Overview

The benchmark infrastructure now supports:
1. **Progress bars** - Visual feedback during long-running benchmarks
2. **Silent mode** - Completely quiet execution for clean benchmark output

## Progress Bar Usage

### Basic Setup

1. Import the progress module:
```rust
mod bench_progress;
```

2. Create a progress bar before your benchmark:
```rust
let base_msg = format!("basic docs={}", num_docs);
let pb = bench_progress::create_benchmark_progress(
    100,  // total iterations (sample_size)
    base_msg.clone()
);
let pb_clone = Arc::clone(&pb);
let base_msg = Arc::new(base_msg);
let base_msg_clone = Arc::clone(&base_msg);
```

3. Update progress with running statistics in your benchmark iteration:
```rust
// Calculate running averages
let avg_precision = precisions.iter().sum::<f64>() / precisions.len() as f64;
let avg_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;

// Update progress bar with stats
bench_progress::update_progress_with_stats(&pb, &base_msg, avg_precision, avg_recall);
pb.inc(1);  // Increment by 1 after each iteration
```

4. Finish the progress bar after benchmark completes:
```rust
bench_progress::finish_benchmark_progress(&pb, "✓ basic docs=2");
```

### Progress Bar Display

The progress bar shows:
- Elapsed time: `[00:45]`
- Animated spinner: `⠋`
- Iteration count: `[155]`
- Benchmark name and running stats: `basic docs=2 | P:0.950 R:0.875`

Example output:
```
[00:45] ⠋ [155] basic docs=2 | P:0.950 R:0.875
```

**Note:** The iteration count includes Criterion's warmup and measurement iterations, so it will exceed the sample_size (100). This is normal and expected.

### Example

See `spnl/benches/haystack.rs` for a complete example with running precision/recall statistics.

## Silent Mode

### Purpose

The `silent` flag in `ExecuteOptions` ensures that backend implementations produce **no output**:
- No stdout text
- No progress bars
- No timing metrics

This is essential for benchmarks where you only want Criterion's output.

### Usage

```rust
let options = ExecuteOptions {
    silent: true,
    ..Default::default()
};
let result = execute(&query, &options).await?;
```

### Implementation Details

The `silent` flag is respected by all backend implementations:
- `openai.rs` - OpenAI/Gemini/Ollama backends
- `spnl.rs` - SPNL API backend
- `mistralrs/mod.rs` - mistral.rs local inference backend

When `silent: true`:
1. `quiet` flag is set to true (suppresses stdout)
2. Progress bars are not created (`pbs = None`)
3. Timing metrics are not printed

## Architecture

```
ExecuteOptions { silent: true }
    ↓
Backend (openai/spnl/mistralrs)
    ↓
quiet = true + pbs = None + no timing output
    ↓
Only benchmark progress bar visible
```

## Benefits

1. **Clean output** - Only Criterion and benchmark-specific output
2. **Performance** - No overhead from progress bars or stdout writes
3. **Reusability** - Progress bar logic in `bench_progress.rs` can be used by all benchmarks
4. **Flexibility** - Can still use progress bars at the benchmark level while silencing backend output

## Future Enhancements

- Add progress bar support to other benchmarks (`inner_outer.rs`, `mt_rag.rs`)
- Consider adding progress bar styles/themes
- Add option to show/hide elapsed time in progress bars
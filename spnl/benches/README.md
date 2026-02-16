# SPNL Benchmarks

This directory contains Criterion-based benchmarks for SPNL.

## Available Benchmarks

### [Haystack Benchmark](README-HAYSTACK.md)

Tests SPNL's ability to extract information from multiple documents using basic and map-reduce approaches.

**Quick Start:**
```bash
cargo bench --bench haystack
```

**Documentation:** See [README-HAYSTACK.md](README-HAYSTACK.md) for:
- Command-line filtering options
- Environment variable configuration
- Example configurations
- Progress bars and output

### [Needle In A Haystack (NIAH) Benchmark](README-NIAH.md)

Evaluates an LLM's ability to retrieve specific information from within large contexts at various depths. Faithful port of the [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) methodology.

**Quick Start:**
```bash
cargo bench --bench niah --features tok
```

**Documentation:** See [README-NIAH.md](README-NIAH.md) for:
- Command-line filtering options
- Environment variable configuration
- Model and tokenizer configuration
- Example configurations
- Debug mode and troubleshooting

### [RULER Benchmark](README-RULER.md)

Evaluates long-context language models across multiple task categories to determine their effective context length. Faithful port of the [RULER](https://github.com/NVIDIA/RULER) benchmark methodology.

**Quick Start:**
```bash
cargo bench --bench ruler --features tok
```

**Documentation:** See [README-RULER.md](README-RULER.md) for:
- Task categories (NIAH, Variable Tracking, and more)
- Environment variable configuration
- Model and tokenizer configuration
- Task-specific complexity settings
- Comprehensive output statistics
- Future work roadmap

## General Criterion Features

All benchmarks support Criterion's built-in features:

### Filtering Benchmarks

Run specific benchmarks by name pattern:
```bash
cargo bench --bench <benchmark_name> -- "<pattern>"
```

### Baseline Comparison

Save and compare against baselines:
```bash
# Save a baseline
cargo bench --bench <benchmark_name> -- --save-baseline my-baseline

# Compare against baseline
cargo bench --bench <benchmark_name> -- --baseline my-baseline
```

### Output Formats

Generate reports in different formats:
```bash
# HTML report (default, in target/criterion/)
cargo bench --bench <benchmark_name>

# JSON output
cargo bench --bench <benchmark_name> -- --output-format json
```

## Common Environment Variables

Most benchmarks support these common configuration options:

- `BENCH_SAMPLE_SIZE` - Number of samples to collect
- `BENCH_MEASUREMENT_TIME` - Maximum measurement time in seconds
- `BENCH_DEBUG` - Enable debug output (usually for first sample only)

See individual benchmark documentation for benchmark-specific variables.

## Progress Bars

All benchmarks use custom progress bars showing real-time statistics during execution. See `BENCHMARK_PROGRESS.md` for implementation details.

## Requirements

- Rust toolchain with Cargo
- Ollama running locally (for LLM-based benchmarks)
- Appropriate models installed in Ollama
- For NIAH: `tok` feature enabled for tokenizer support

## Tips for Faster Runs

1. **Use command-line filtering** to test specific configurations
2. **Reduce sample size** with `BENCH_SAMPLE_SIZE`
3. **Reduce measurement time** with `BENCH_MEASUREMENT_TIME`
4. **Test fewer configurations** using benchmark-specific environment variables
5. **Combine filters** for maximum control

Example quick test:
```bash
BENCH_SAMPLE_SIZE=3 BENCH_MEASUREMENT_TIME=10 cargo bench --bench haystack -- "basic/2"
```

## Benchmark Development

See `BENCHMARK_PROGRESS.md` for details on implementing progress bars in new benchmarks.
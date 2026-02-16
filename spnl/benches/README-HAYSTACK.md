# Haystack Benchmark

The haystack benchmark tests SPNL's ability to extract information from multiple documents.

## Running the Benchmark

Run all benchmarks:
```bash
cargo bench --bench haystack
```

## Command-line Filtering

Use Criterion's built-in filtering to run specific benchmarks:

Run only basic benchmarks:
```bash
cargo bench --bench haystack basic
```

Run only map-reduce benchmarks:
```bash
cargo bench --bench haystack map_reduce
```

Run basic benchmark with 2 documents only:
```bash
cargo bench --bench haystack basic/2
```

Run map-reduce with chunk size 4 only:
```bash
cargo bench --bench haystack map_reduce/4
```

## Environment Variables

Use environment variables to control benchmark behavior:

### `BENCH_SAMPLE_SIZE` (default: `100`)

Number of samples to collect for statistics:
```bash
# Quick test with fewer samples
BENCH_SAMPLE_SIZE=20 cargo bench --bench haystack

# More thorough test
BENCH_SAMPLE_SIZE=200 cargo bench --bench haystack
```

### `BENCH_MEASUREMENT_TIME` (default: `320` seconds)

Maximum time to spend measuring each benchmark:
```bash
# Quick test (may not complete all samples)
BENCH_MEASUREMENT_TIME=60 cargo bench --bench haystack

# Longer test for slower systems
BENCH_MEASUREMENT_TIME=600 cargo bench --bench haystack
```

**Note:** If you see "Unable to complete N samples in Xs", either:
- Increase `BENCH_MEASUREMENT_TIME`
- Decrease `BENCH_SAMPLE_SIZE`
- Use fewer test configurations (e.g., `BENCH_NUM_DOCS=2`)

### `BENCH_NUM_DOCS` (default: `2,4,8`)

Comma-separated list of document counts for basic benchmarks:
```bash
# Test only with 2 documents
BENCH_NUM_DOCS=2 cargo bench --bench haystack basic

# Test with 2, 8, and 16 documents
BENCH_NUM_DOCS=2,8,16 cargo bench --bench haystack basic
```

### `BENCH_CHUNK_SIZES` (default: `2,4`)

Comma-separated list of chunk sizes for map-reduce benchmarks:
```bash
# Test only chunk size 2
BENCH_CHUNK_SIZES=2 cargo bench --bench haystack map_reduce

# Test chunk sizes 2, 4, and 8
BENCH_CHUNK_SIZES=2,4,8 cargo bench --bench haystack map_reduce
```

### `BENCH_MAP_REDUCE_NUM_DOCS` (default: `8`)

Number of documents to use for map-reduce benchmarks:
```bash
# Test map-reduce with 16 documents
BENCH_MAP_REDUCE_NUM_DOCS=16 cargo bench --bench haystack map_reduce

# Test with fewer documents for faster runs
BENCH_MAP_REDUCE_NUM_DOCS=4 cargo bench --bench haystack map_reduce
```

### `BENCH_DOC_LENGTH` (default: `100`)

Number of lipsum words per document:
```bash
# Test with shorter documents (faster)
BENCH_DOC_LENGTH=50 cargo bench --bench haystack

# Test with longer documents
BENCH_DOC_LENGTH=200 cargo bench --bench haystack
```

### `BENCH_BASIC` and `BENCH_MAP_REDUCE` (default: `true`)

Enable/disable entire benchmark categories:
```bash
# Skip basic benchmarks entirely
BENCH_BASIC=false cargo bench --bench haystack

# Run only map-reduce
BENCH_BASIC=false BENCH_MAP_REDUCE=true cargo bench --bench haystack
```

## Example Configurations

Quick test with single configuration:
```bash
cargo bench --bench haystack basic/2
```

Test multiple document counts:
```bash
BENCH_NUM_DOCS=2,4,8,16 cargo bench --bench haystack basic
```

Test specific map-reduce configuration:
```bash
cargo bench --bench haystack map_reduce/4
```

Quick test with minimal configuration:
```bash
BENCH_SAMPLE_SIZE=20 BENCH_MEASUREMENT_TIME=60 BENCH_NUM_DOCS=2 cargo bench --bench haystack basic
```

Thorough test with more samples:
```bash
BENCH_SAMPLE_SIZE=200 BENCH_MEASUREMENT_TIME=600 cargo bench --bench haystack
```

## Progress Bars

The benchmark displays real-time progress with running statistics:
```
[00:45] ⠋ [155] basic docs=2 | P:0.950 R:0.875
```

- **Elapsed time**: `[00:45]`
- **Spinner**: `⠋` (animated)
- **Iteration count**: `[155]` (includes Criterion warmup + measurement)
- **Running stats**: `P:0.950 R:0.875` (precision and recall averages)

See `BENCHMARK_PROGRESS.md` for more details on the progress bar implementation.

## Output

After each benchmark completes, you'll see:
1. Criterion's timing statistics
2. Quantile statistics for precision and recall (p25, p50, p75, p90, p99)

## Requirements

- Ollama running locally with `granite3.3:2b` model
- Or configure a different model in the benchmark code
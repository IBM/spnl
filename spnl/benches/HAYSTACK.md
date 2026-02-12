# Haystack Benchmark

Tests SPNL's ability to extract information from multiple documents.

## Quick Start

Run all benchmarks:
```bash
cargo bench --bench haystack
```

Run specific benchmark:
```bash
cargo bench --bench haystack basic/2
```

Quick test (20 samples, 60 seconds):
```bash
BENCH_SAMPLE_SIZE=20 BENCH_MEASUREMENT_TIME=60 BENCH_NUM_DOCS=2 cargo bench --bench haystack basic
```

## Benchmark Types

### Basic
Tests information extraction from N documents without chunking.

### Map-Reduce
Tests information extraction using map-reduce pattern with document chunking.

## Filtering Benchmarks

Use Criterion's built-in filtering:

```bash
# Run only basic benchmarks
cargo bench --bench haystack basic

# Run only map-reduce benchmarks
cargo bench --bench haystack map_reduce

# Run basic with specific parameters
cargo bench --bench haystack "basic/docs=2/len=100"

# Run all benchmarks with length 50
cargo bench --bench haystack "len=50"

# Run map-reduce with specific chunk size
cargo bench --bench haystack "map_reduce/chunk=4"
```

## Configuration

### Benchmark Execution

#### `BENCH_SAMPLE_SIZE` (default: `100`)
Number of samples to collect:
```bash
BENCH_SAMPLE_SIZE=20 cargo bench --bench haystack
```

#### `BENCH_MEASUREMENT_TIME` (default: `320` seconds)
Maximum time per benchmark:
```bash
BENCH_MEASUREMENT_TIME=600 cargo bench --bench haystack
```

**Note:** If you see "Unable to complete N samples in Xs", either increase measurement time or decrease sample size.

### Test Parameters

#### `BENCH_NUM_DOCS` (default: `2,4,8`)
Document counts for basic benchmarks (comma-separated):
```bash
# Single value
BENCH_NUM_DOCS=2 cargo bench --bench haystack basic

# Multiple values
BENCH_NUM_DOCS=2,8,16 cargo bench --bench haystack basic

# Filter to specific value
BENCH_NUM_DOCS=2,4,8 cargo bench --bench haystack "docs=8"
```

#### `BENCH_CHUNK_SIZES` (default: `2,4`)
Chunk sizes for map-reduce:
```bash
BENCH_CHUNK_SIZES=2,4,8 cargo bench --bench haystack map_reduce
```

#### `BENCH_MAP_REDUCE_NUM_DOCS` (default: `8`)
Document count for map-reduce:
```bash
BENCH_MAP_REDUCE_NUM_DOCS=16 cargo bench --bench haystack map_reduce
```

#### `BENCH_DOC_LENGTH` (default: `100`)
Lipsum words per document (comma-separated):
```bash
# Single value - shorter documents (faster)
BENCH_DOC_LENGTH=50 cargo bench --bench haystack

# Multiple values
BENCH_DOC_LENGTH=50,100,200 cargo bench --bench haystack

# Test only length 1000
BENCH_DOC_LENGTH=1000 cargo bench --bench haystack

# Test multiple lengths but filter to one
BENCH_DOC_LENGTH=100,1000 cargo bench --bench haystack "len=1000"
```

### Enable/Disable Categories

#### `BENCH_BASIC` (default: `true`)
```bash
BENCH_BASIC=false cargo bench --bench haystack
```

#### `BENCH_MAP_REDUCE` (default: `true`)
```bash
BENCH_MAP_REDUCE=false cargo bench --bench haystack
```

## Example Configurations

### Quick Development Test
```bash
BENCH_SAMPLE_SIZE=20 \
BENCH_MEASUREMENT_TIME=60 \
BENCH_NUM_DOCS=2 \
BENCH_DOC_LENGTH=50 \
cargo bench --bench haystack basic
```

### Thorough Production Test
```bash
BENCH_SAMPLE_SIZE=200 \
BENCH_MEASUREMENT_TIME=600 \
BENCH_NUM_DOCS=2,4,8,16 \
cargo bench --bench haystack
```

### Test Specific Configuration
```bash
cargo bench --bench haystack basic/8
```

### Test Map-Reduce Scaling
```bash
BENCH_CHUNK_SIZES=2,4,8 \
BENCH_MAP_REDUCE_NUM_DOCS=16 \
cargo bench --bench haystack map_reduce
```

## Progress Display

Real-time progress with statistics:
```
[00:45] ⠋ [155] basic docs=2 | P:0.950 R:0.875
```

- **Elapsed time**: `[00:45]`
- **Spinner**: `⠋` (animated)
- **Iteration count**: `[155]` (includes warmup + measurement)
- **Running stats**: `P:0.950 R:0.875` (precision/recall averages)

## Output

After each benchmark:
1. Criterion timing statistics
2. Quantile statistics for precision and recall:
   - min, p25, p50 (median), p75, p90, p99, max

## Requirements

- Ollama running locally with `granite3.3:2b` model
- Or modify the model in the benchmark code
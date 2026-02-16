# Needle In A Haystack (NIAH) Benchmark

This benchmark is a faithful Rust port of the [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) test methodology adapted to use SPNL span queries.

## Overview

The test evaluates an LLM's ability to retrieve specific information ("needle") from within a large context ("haystack") by:

1. Loading Paul Graham essays as context
2. Encoding to tokens and trimming to exact context length
3. Inserting needle at sentence boundaries at various depths (0-100%)
4. Asking the model to retrieve the needle
5. Measuring retrieval accuracy

## Faithful to Original Methodology

This implementation matches the original Python code:
- Uses token-based context lengths (not word-based)
- Encodes context to tokens, trims to exact token count
- Inserts needle at sentence boundaries (backs up to find period token)
- Accounts for 200-token buffer for system/question/response
- Works in token space, then decodes back to text

## Running the Benchmark

### Basic Usage

Run with default settings (requires granite3.3:8b in Ollama):
```bash
cargo bench --bench niah --features tok
```

### Using a Different Model

Use any Ollama model:
```bash
BENCH_MODEL="ollama/llama3.2:3b" \
BENCH_TOKENIZER_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
cargo bench --bench niah --features tok
```

### Quick Test Run

For faster testing during development:
```bash
BENCH_SAMPLE_SIZE=3 \
BENCH_MEASUREMENT_TIME=10 \
BENCH_CONTEXT_LENGTHS="1000,2000" \
BENCH_DEPTH_PERCENTAGES="0,50,100" \
cargo bench --bench niah --features tok
```

This reduces from:
- **Default**: 4 context lengths × 5 depths × 10 samples = 200 tests
- **Quick**: 2 context lengths × 3 depths × 3 samples = 18 tests

### Debug Mode

Enable debug output for the first sample to see detailed information:
```bash
BENCH_DEBUG=1 cargo bench --bench niah --features tok
```

## Command-line Filtering

Use Criterion's built-in filtering to run specific benchmark configurations. The benchmark IDs follow the pattern: `retrieval/len={context_length}/depth={depth_percent}`

### Filter by Context Length

Run only benchmarks with 1000 token contexts:
```bash
cargo bench --bench niah --features tok -- "len=1000"
```

Run only 1000 and 2000 token contexts:
```bash
cargo bench --bench niah --features tok -- "len=(1000|2000)"
```

### Filter by Depth Percentage

Run only edge depths (beginning and end):
```bash
cargo bench --bench niah --features tok -- "depth=(0|100)"
```

Run only middle depth:
```bash
cargo bench --bench niah --features tok -- "depth=50"
```

### Filter by Specific Configuration

Run a single specific configuration:
```bash
cargo bench --bench niah --features tok -- "len=2000/depth=50"
```

Run multiple specific configurations:
```bash
cargo bench --bench niah --features tok -- "len=1000/depth=(0|50|100)"
```

### Combining Filters with Environment Variables

Combine command-line filters with environment variables for maximum control:
```bash
# Quick test of only small contexts with fewer samples
BENCH_SAMPLE_SIZE=3 cargo bench --bench niah --features tok -- "len=1000"

# Test edge cases with debug output
BENCH_DEBUG=1 cargo bench --bench niah --features tok -- "depth=(0|100)"
```

## Environment Variables

### `BENCH_SAMPLE_SIZE` (default: `10`)

Number of samples per configuration:
```bash
# Quick test with fewer samples
BENCH_SAMPLE_SIZE=3 cargo bench --bench niah --features tok

# More thorough test
BENCH_SAMPLE_SIZE=20 cargo bench --bench niah --features tok
```

### `BENCH_MEASUREMENT_TIME` (default: `60` seconds)

Maximum time to spend measuring each benchmark:
```bash
# Quick test
BENCH_MEASUREMENT_TIME=10 cargo bench --bench niah --features tok

# Longer test for slower systems
BENCH_MEASUREMENT_TIME=120 cargo bench --bench niah --features tok
```

### `BENCH_CONTEXT_LENGTHS` (default: `"1000,2000,4000,8000"`)

Comma-separated context lengths in TOKENS:
```bash
# Test only small contexts
BENCH_CONTEXT_LENGTHS="1000,2000" cargo bench --bench niah --features tok

# Test larger contexts
BENCH_CONTEXT_LENGTHS="4000,8000,16000" cargo bench --bench niah --features tok
```

### `BENCH_DEPTH_PERCENTAGES` (default: `"0,25,50,75,100"`)

Comma-separated depth percentages (0-100):
```bash
# Test only edge cases
BENCH_DEPTH_PERCENTAGES="0,100" cargo bench --bench niah --features tok

# Test more granular depths
BENCH_DEPTH_PERCENTAGES="0,10,20,30,40,50,60,70,80,90,100" cargo bench --bench niah --features tok
```

### `BENCH_MODEL` (default: `"ollama/granite3.3:8b"`)

Model to use for inference (Ollama format):
```bash
BENCH_MODEL="ollama/llama3.2:3b" cargo bench --bench niah --features tok
```

### `BENCH_TOKENIZER_MODEL` (default: `"ibm-granite/granite-3.3-8b-instruct"`)

HuggingFace model ID for tokenizer (must match the inference model's tokenizer):
```bash
BENCH_TOKENIZER_MODEL="meta-llama/Llama-3.2-3B-Instruct" cargo bench --bench niah --features tok
```

### `BENCH_FINAL_CONTEXT_LENGTH_BUFFER` (default: `200`)

Buffer for system message, question, and response (in tokens):
```bash
BENCH_FINAL_CONTEXT_LENGTH_BUFFER=300 cargo bench --bench niah --features tok
```

### `BENCH_DEBUG` (default: `false`)

Enable debug output for the first sample:
```bash
BENCH_DEBUG=1 cargo bench --bench niah --features tok
```

## Example Configurations

### Minimal Quick Test
```bash
BENCH_SAMPLE_SIZE=3 \
BENCH_MEASUREMENT_TIME=10 \
BENCH_CONTEXT_LENGTHS="1000" \
BENCH_DEPTH_PERCENTAGES="50" \
cargo bench --bench niah --features tok
```

### Edge Case Testing
```bash
BENCH_SAMPLE_SIZE=5 \
BENCH_DEPTH_PERCENTAGES="0,100" \
cargo bench --bench niah --features tok
```

### Comprehensive Test
```bash
BENCH_SAMPLE_SIZE=20 \
BENCH_MEASUREMENT_TIME=120 \
BENCH_CONTEXT_LENGTHS="1000,2000,4000,8000,16000" \
BENCH_DEPTH_PERCENTAGES="0,10,25,50,75,90,100" \
cargo bench --bench niah --features tok
```

### Debug Single Configuration
```bash
BENCH_DEBUG=1 \
BENCH_SAMPLE_SIZE=1 \
cargo bench --bench niah --features tok -- "len=2000/depth=50"
```

## Progress Bars

The benchmark displays real-time progress with running statistics:
```
[00:45] ⠋ len=2000 depth=50% | n=7 | Acc=85.7% | Perfect=6/7
```

- **Elapsed time**: `[00:45]`
- **Spinner**: `⠋` (animated)
- **Configuration**: `len=2000 depth=50%`
- **Sample count**: `n=7`
- **Running accuracy**: `Acc=85.7%`
- **Perfect retrievals**: `Perfect=6/7` (responses that got the exact answer)

## Output Statistics

After each configuration completes, you'll see detailed statistics:

```
=== Accuracy Stats: len=2000 depth=50% (n=10) ===
  avg:  85.0%
  min:  0.0%
  p25:  100.0%
  p50:  100.0%
  p75:  100.0%
  p90:  100.0%
  p99:  100.0%
  max:  100.0%
  perfect: 8/10
```

- **avg**: Mean accuracy across all samples
- **min/max**: Minimum and maximum accuracy
- **p25/p50/p75/p90/p99**: Percentile statistics
- **perfect**: Number of samples with 100% accuracy

## Data Caching

The benchmark automatically downloads and caches Paul Graham essays on first run:
- Cache location: `~/.cache/spnl/niah/paul_graham_essays_combined.txt`
- Size: ~728KB (49 essay files)
- One-time download, reused for all subsequent runs

## Requirements

- Ollama running locally with the specified model
- HuggingFace tokenizers library (included via `tok` feature)
- Internet connection for first run (to download essays)

## Troubleshooting

### "Failed to fetch Paul Graham essays"
The benchmark will fall back to embedded essays if download fails. For full testing, ensure internet connectivity on first run.

### "Unable to complete N samples in Xs"
Either:
- Increase `BENCH_MEASUREMENT_TIME`
- Decrease `BENCH_SAMPLE_SIZE`
- Use command-line filtering to test fewer configurations

### Model not found
Ensure the model is available in Ollama:
```bash
ollama list
ollama pull granite3.3:8b  # or your chosen model
```

### Tokenizer mismatch
Ensure `BENCH_TOKENIZER_MODEL` matches the tokenizer used by your inference model. Check the model card on HuggingFace for the correct tokenizer ID.

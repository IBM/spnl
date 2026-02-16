# RULER Benchmark for SPNL

This is a faithful Rust port of the [RULER benchmark](https://github.com/NVIDIA/RULER) (What's the Real Context Size of Your Long-Context Language Models?) adapted to use SPNL span queries.

## Overview

RULER evaluates long-context language models across multiple task categories to determine their effective context length. This implementation currently supports:

1. **✅ Retrieval (NIAH)**: Needle-in-a-haystack with configurable complexity
2. **✅ Multi-hop Tracing (Variable Tracking)**: Track variable assignments through chains
3. **🚧 Aggregation (Common Words Extraction)**: Extract most common words (not yet implemented)
4. **🚧 Aggregation (Frequent Words Extraction)**: Extract frequent words using Zipf distribution (not yet implemented)
5. **🚧 Question Answering**: Answer questions based on context (not yet implemented)

## Faithful to Original Methodology

This implementation matches the original Python code:

- ✅ Uses token-based context lengths (not word-based)
- ✅ Binary search to find optimal haystack size for target context length
- ✅ Same validation logic: `string_match_all` and `string_match_part`
- ✅ Same task templates and answer prefixes from `constants.py`
- ✅ Same complexity configurations (num_chains, num_hops, freq_cw, etc.)
- ✅ Same data generation approach (Paul Graham essays, random needles, variable chains)

## Key Differences from Original

1. **Language**: Rust instead of Python
2. **Framework**: Uses SPNL span queries instead of direct API calls
3. **Scope**: Currently implements 2 of 5 task categories (NIAH and Variable Tracking)
4. **Dependencies**: Uses HuggingFace tokenizers instead of tiktoken/sentencepiece

## Configuration via Environment Variables

### General Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_SAMPLE_SIZE` | Number of samples per configuration (minimum 10) | `10` |
| `BENCH_MEASUREMENT_TIME` | Measurement time in seconds | `60` |
| `BENCH_CONTEXT_LENGTHS` | Comma-separated context lengths in TOKENS | `"4000,8000"` |
| `BENCH_MODEL` | Model to use for inference | `"ollama/granite3.3:2b"` |
| `BENCH_TOKENIZER_MODEL` | HuggingFace model for tokenizer | `"ibm-granite/granite-3.3-2b-instruct"` |
| `BENCH_FINAL_CONTEXT_LENGTH_BUFFER` | Buffer for system/question/response | `200` |
| `BENCH_DEBUG` | Enable debug output for first sample | `false` |
| `BENCH_TASKS` | Comma-separated tasks to run: `niah`, `variable_tracking` | `"niah"` |

### Task-Specific Settings

#### NIAH (Needle-in-a-Haystack)

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_NIAH_NUM_NEEDLE_K` | Number of needles (keys) to insert | `1` |
| `BENCH_NIAH_NUM_NEEDLE_V` | Number of values per needle | `1` |
| `BENCH_NIAH_NUM_NEEDLE_Q` | Number of needles to query | `1` |
| `BENCH_NIAH_DEPTH_PERCENTAGES` | Comma-separated depth percentages | `"50"` |

#### Variable Tracking

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_VT_NUM_CHAINS` | Number of variable chains | `1` |
| `BENCH_VT_NUM_HOPS` | Number of hops per chain | `4` |

## Usage Examples

### Basic Usage

```bash
# Run with default settings (requires granite3.3:2b in Ollama)
cargo bench --bench ruler --features tok
```

### Run Specific Tasks

```bash
# Run only NIAH
BENCH_TASKS="niah" cargo bench --bench ruler --features tok

# Run both NIAH and Variable Tracking
BENCH_TASKS="niah,variable_tracking" cargo bench --bench ruler --features tok
```

### Debug Mode

```bash
# Enable debug output for the first sample
BENCH_DEBUG=1 cargo bench --bench ruler --features tok
```

### Custom NIAH Configuration

```bash
# Test with multiple needles
BENCH_NIAH_NUM_NEEDLE_K=3 BENCH_NIAH_NUM_NEEDLE_V=2 \
  cargo bench --bench ruler --features tok

# Test at different depths
BENCH_NIAH_DEPTH_PERCENTAGES="0,25,50,75,100" \
  cargo bench --bench ruler --features tok
```

### Custom Variable Tracking Configuration

```bash
# More complex variable tracking
BENCH_VT_NUM_CHAINS=2 BENCH_VT_NUM_HOPS=8 \
  cargo bench --bench ruler --features tok
```

### Custom Context Lengths

```bash
# Test at longer context lengths
BENCH_CONTEXT_LENGTHS="8000,16000,32000" \
  cargo bench --bench ruler --features tok
```

### Use Different Model

```bash
# Use a different Ollama model
BENCH_MODEL="ollama/llama3.2:3b" \
BENCH_TOKENIZER_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
  cargo bench --bench ruler --features tok
```

## Output Format

The benchmark produces detailed statistics for each configuration, including comprehensive quantile analysis:

```
=== NIAH Stats: len=4000 depth=50% (n=5) ===
  avg:  80.0%
  min:  60.0%
  p25:  80.0%
  p50:  80.0%
  p75:  100.0%
  p90:  100.0%
  p99:  100.0%
  max:  100.0%
  perfect: 4/5

=== Variable Tracking Stats: len=4000 (n=5) ===
  avg:  60.0%
  min:  40.0%
  p25:  60.0%
  p50:  60.0%
  p75:  80.0%
  p90:  80.0%
  p99:  80.0%
  max:  80.0%
  perfect: 3/5
```

The statistics include:
- **avg**: Mean accuracy across all samples
- **min/max**: Minimum and maximum accuracy observed
- **p25/p50/p75/p90/p99**: Percentile values showing distribution of results
- **perfect**: Count of samples with 100% accuracy

## Implementation Details

### NIAH Task

The NIAH task implementation:
1. Generates random 7-digit numbers as needles
2. Inserts needles into Paul Graham essays at specified depths
3. Uses binary search to find optimal haystack size for target token count
4. Evaluates using `string_match_all` metric (all needles must be found)

### Variable Tracking Task

The Variable Tracking task implementation:
1. Generates random 5-letter uppercase variable names
2. Creates chains of variable assignments (e.g., `VAR A = 12345`, `VAR B = VAR A`)
3. Inserts chains into noise sentences at random positions
4. Evaluates using `string_match_all` metric (all variables in chain must be found)

### Evaluation Metrics

- **`string_match_all`**: Returns the fraction of reference strings found in the prediction (case-insensitive)
- **`string_match_part`**: Returns 1.0 if ANY reference string is found, 0.0 otherwise (case-insensitive)

## Comparison with Original RULER

### Similarities

- ✅ Token-based context length measurement
- ✅ Binary search for optimal haystack size
- ✅ Same needle format: "One of the special magic numbers for {key} is: {value}."
- ✅ Same variable tracking format: "VAR {name} = {value}" or "VAR {name} = VAR {other}"
- ✅ Same evaluation metrics
- ✅ Uses Paul Graham essays as haystack
- ✅ Configurable complexity parameters

### Differences

- ⚠️ Only 2 of 5 task categories implemented (NIAH and Variable Tracking)
- ⚠️ Uses HuggingFace tokenizers instead of tiktoken
- ⚠️ Simplified random word generation (no wonderwords library)
- ⚠️ No few-shot examples in Variable Tracking (original uses ICL)
- ⚠️ Simplified sentence tokenization (split on '.' instead of NLTK)

## Future Work

To complete the RULER benchmark port, the following tasks remain:

1. **Common Words Extraction**: Implement word frequency counting task
2. **Frequent Words Extraction**: Implement Zipf distribution-based word extraction
3. **Question Answering**: Implement QA task using SQuAD/HotpotQA datasets
4. **Enhanced Metrics**: Add more detailed statistics and visualizations
5. **Multi-model Support**: Add support for more model backends

## References

- Original RULER Paper: [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
- Original RULER Repository: [https://github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER)
- NIAH Original: [https://github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

## Citation

If you use this benchmark, please cite both the original RULER paper and SPNL:

```bibtex
@article{hsieh2024ruler,
  title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Cheng-Ping Hsieh and Simeng Sun and Samuel Kriman and Shantanu Acharya and Dima Rekesh and Fei Jia and Yang Zhang and Boris Ginsburg},
  year={2024},
  journal={arXiv preprint arXiv:2404.06654},
}
```

---

Made with Bob
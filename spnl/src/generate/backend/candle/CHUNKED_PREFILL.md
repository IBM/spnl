# Chunked Prefill for Candle Backend

## Overview

The Candle backend now supports configurable chunked prefill processing. This allows you to test different chunk sizes to potentially improve TTFT (Time To First Token) performance on Metal (macOS GPU).

## Background

By default, the prefill phase processes all prompt tokens in a single forward pass, which maximizes parallelism. However, on Metal, different chunk sizes may perform better depending on:
- Model size
- Prompt length
- GPU memory bandwidth characteristics
- Metal kernel tile sizes

## Configuration

### Environment Variable

Set `CANDLE_PREFILL_CHUNK_SIZE` to control chunking behavior:

```bash
# No chunking (default) - process all tokens at once
export CANDLE_PREFILL_CHUNK_SIZE=0
# or unset CANDLE_PREFILL_CHUNK_SIZE

# Process in 128-token chunks
export CANDLE_PREFILL_CHUNK_SIZE=128

# Process in 256-token chunks
export CANDLE_PREFILL_CHUNK_SIZE=256

# Process in 512-token chunks
export CANDLE_PREFILL_CHUNK_SIZE=512

# Process in 1024-token chunks
export CANDLE_PREFILL_CHUNK_SIZE=1024
```

## Usage Examples

### Basic Usage

```bash
# Test with 256-token chunks
export CANDLE_PREFILL_CHUNK_SIZE=256
cargo run --release --features candle -- generate \
  --model path/to/model \
  --prompt "Your prompt here" \
  --max-tokens 100
```

### Benchmarking Script

Use the provided test script to systematically benchmark different chunk sizes:

```bash
cd spnl
./test-chunked-prefill.sh path/to/model "Your test prompt"
```

This will test chunk sizes: 0 (no chunking), 128, 256, 512, and 1024 tokens.

## How It Works

### Single-Pass Prefill (Default: chunk_size = 0)

```rust
// Process all prompt tokens at once
let input = Tensor::new(&tokens[..], config.device)?.unsqueeze(0)?;
let _logits = model.forward_pass(&input, 0)?;
```

**Pros:**
- Maximizes parallelism
- Single forward pass
- Optimal for most cases

**Cons:**
- May exceed optimal Metal tile size for very long prompts
- Higher peak memory usage

### Chunked Prefill (chunk_size > 0)

```rust
// Process tokens in chunks
let mut pos = 0;
for chunk_start in (0..prompt_len).step_by(chunk_size) {
    let chunk_end = (chunk_start + chunk_size).min(prompt_len);
    let chunk = &tokens[chunk_start..chunk_end];
    let input = Tensor::new(chunk, config.device)?.unsqueeze(0)?;
    let _logits = model.forward_pass(&input, pos)?;
    pos += chunk.len();
}
```

**Pros:**
- May better fit Metal's optimal tile size
- Lower peak memory usage
- Better memory access patterns for some workloads

**Cons:**
- Multiple forward passes add overhead
- Loses some parallelism benefit
- May be slower for short prompts

## Performance Considerations

### When Chunking Might Help

- **Very long prompts** (>1000 tokens): May exceed Metal's optimal batch size
- **Large models**: Memory bandwidth constraints may benefit from smaller chunks
- **Specific Metal GPU characteristics**: Different Apple Silicon chips may have different optimal sizes

### When Chunking Likely Won't Help

- **Short prompts** (<500 tokens): Overhead of multiple passes outweighs benefits
- **Small models**: Already fit well in Metal's processing pipeline
- **Well-optimized kernels**: If Candle's Metal kernels are already optimal

## Previous Test Results

**Test Configuration:**
- Model: Qwen 0.6B
- Hardware: Metal (macOS GPU)
- Prompt: 500-1000 tokens
- Chunk size tested: 256 tokens

**Result:** No improvement (added overhead from multiple forward passes)

**Why Re-implement?**
- Different chunk sizes may work better
- Different models/prompt lengths may benefit
- Easy to test without code changes
- Metal performance characteristics vary by workload

## Recommended Testing Approach

1. **Establish Baseline:**
   ```bash
   unset CANDLE_PREFILL_CHUNK_SIZE
   # Run your workload and measure TTFT
   ```

2. **Test Different Chunk Sizes:**
   ```bash
   for size in 128 256 512 1024; do
     export CANDLE_PREFILL_CHUNK_SIZE=$size
     # Run your workload and measure TTFT
   done
   ```

3. **Compare Results:**
   - Look for consistent improvements across multiple runs
   - Consider both TTFT and total generation time
   - Test with different prompt lengths

4. **Use Optimal Setting:**
   ```bash
   # If 256 was best:
   export CANDLE_PREFILL_CHUNK_SIZE=256
   # Add to your shell profile for persistence
   ```

## Implementation Details

**Location:** `spnl/src/generate/backend/candle/model.rs` lines 65-85

**Key Points:**
- Chunking only applies to prefill phase (not token generation)
- KV cache is properly maintained across chunks
- Position tracking ensures correct RoPE application
- Default behavior (no chunking) is unchanged

## Troubleshooting

### Slower Performance with Chunking

This is expected in many cases. The overhead of multiple forward passes often outweighs the benefits. Try:
- Larger chunk sizes (512, 1024)
- Disable chunking (set to 0)
- Profile with Xcode Instruments to see where time is spent

### No Difference in Performance

If you see no difference between chunk sizes:
- The bottleneck may be elsewhere (e.g., model loading, tokenization)
- Metal kernels may already be optimal for your workload
- Try profiling to identify the actual bottleneck

### Errors or Crashes

If you encounter errors:
- Ensure chunk size is reasonable (>0 if set, typically 128-2048)
- Check that your model supports the position-independent KV cache
- Verify you're using the latest version of the code

## Future Work

- Automatic chunk size selection based on model/prompt characteristics
- Dynamic chunking that adapts to available memory
- Integration with Metal Performance Shaders for better optimization
- Upstream contributions to Candle for better Metal kernel performance

## References

- **OPTIMIZATIONS.md**: Full performance analysis and optimization plan
- **model.rs**: Implementation of chunked prefill
- **test-chunked-prefill.sh**: Benchmarking script
- **Candle Metal Backend**: https://github.com/huggingface/candle

---

*Last Updated: 2026-02-08*
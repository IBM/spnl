# mistral.rs Backend for SPNL

This backend provides local inference using the [mistral.rs](https://github.com/EricLBuehler/mistral.rs) library.

## Features

- **Multiple Architectures**: Llama, Mistral, Mixtral, Qwen, Phi, Gemma, and more
- **Quantization Support**: GGUF, GPTQ formats with embedded tokenizers
- **Flash Attention**: Faster inference on supported hardware
- **Model Caching**: Automatic caching of loaded models
- **Streaming Output**: Real-time token streaming to stdout (like ChatGPT)
- **Smart Caching**: Checks local cache before downloading from HuggingFace

## Usage

Use the `mistralrs/` prefix for model names:

```spnl
(generate
  (model "mistralrs/meta-llama/Llama-3.2-1B-Instruct")
  (input "What is the capital of France?"))
```

## Supported Models

Any model supported by mistral.rs can be used. Examples:

### Standard Models
- `mistralrs/meta-llama/Llama-3.2-1B-Instruct`
- `mistralrs/mistralai/Mistral-7B-Instruct-v0.3`
- `mistralrs/Qwen/Qwen2.5-7B-Instruct`
- `mistralrs/microsoft/Phi-3-mini-4k-instruct`

### GGUF Models (Quantized)
- `mistralrs/Qwen/Qwen3-1.7B-GGUF`
- `mistralrs/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF`
- Any HuggingFace model with "GGUF" in the name

## Environment Variables

Configure the backend with these environment variables:

### Performance & Optimization
- `MISTRALRS_ISQ`: In-situ quantization type for non-GGUF models (`Q4K`, `Q5K`, `Q8_0`, etc.) - default: none (full precision)
- `MISTRALRS_PAGED_ATTN`: Enable PagedAttention (`true`/`false`) - default: `true` if supported
- `MISTRALRS_PAGED_ATTN_BLOCK_SIZE`: PagedAttention block size - default: `32`
- `MISTRALRS_PREFIX_CACHE_N`: Number of sequences to cache (set to `0` or `false` to disable) - default: `16`

### General Configuration
- `MISTRALRS_NO_STREAM`: Disable streaming output (`1` to disable) - default: streaming enabled
- `MISTRALRS_DEVICE`: Device to use (`auto`, `cpu`, `cuda`, `metal`) - default: `auto`
- `HF_HOME`: HuggingFace cache directory - default: `~/.cache/huggingface`
- `HF_TOKEN`: HuggingFace token for private models

## Examples

### Simple completion:

```spnl
(generate
  (model "mistralrs/meta-llama/Llama-3.2-1B-Instruct")
  (max-tokens 100)
  (temperature 0.7)
  (input "Explain quantum computing in simple terms."))
```

### Chat conversation:

```spnl
(repeat
  (n 3)
  (g
    (model "mistralrs/mistralai/Mistral-7B-Instruct-v0.3")
    (input
      (seq
        (system "You are a helpful assistant.")
        (user "What are the benefits of Rust?")))))
```

### Batch processing:

```spnl
(map
  (model "mistralrs/Qwen/Qwen2.5-7B-Instruct")
  (inputs
    "Translate to French: Hello"
    "Translate to Spanish: Hello"
    "Translate to German: Hello"))
```

## Implementation Status

### Phase 1: MVP ✅
- [x] Basic model loading
- [x] Text generation (completion and chat)
- [x] Model caching
- [x] Environment variable configuration
- [x] Multiple architecture support

### Phase 2: Core Features ✅
- [x] GGUF support with embedded tokenizers
- [x] Smart cache checking (local-first)
- [x] Streaming output with real-time tokens
- [ ] Progress bars for downloads (deferred)
- [ ] Parallel inference optimization

### Phase 3: Advanced Features (Planned)
- [ ] Vision model support
- [ ] Tool/function calling
- [ ] Speculative decoding
- [ ] Performance benchmarks

## Comparison with Candle Backend

| Feature | Candle | mistral.rs |
|---------|--------|------------|
| Model Architectures | 7 | 15+ |
| Quantization | Limited GGUF | GGUF, GPTQ, GGML |
| Flash Attention | No | Yes |
| Streaming Output | ✅ | ✅ |
| Code Complexity | ~3,500 lines | ~640 lines |
| Maintenance | High | Low |

## Streaming Output

By default, the backend streams tokens to stdout in real-time (green colored output). This provides immediate feedback during generation, similar to ChatGPT.

To disable streaming (useful for scripting):
```bash
MISTRALRS_NO_STREAM=1 spnl -c '(generate (model "mistralrs/...") (input "..."))'
```

## Troubleshooting

### Model not loading
- Check that the model name is correct
- Verify HuggingFace token if using private models: `export HF_TOKEN=your_token`
- Ensure sufficient disk space for model downloads
- For GGUF models, ensure the model name contains "GGUF"

### Out of memory
- Try a smaller model (e.g., TinyLlama-1.1B instead of Llama-7B)
- Use quantized GGUF models (Q4_K_M, Q8_0 formats)
- Set `MISTRALRS_DEVICE=cpu` to use CPU instead of GPU

### Slow inference
- **Enable quantization for non-GGUF models**: `MISTRALRS_ISQ=Q4K` (2-4x faster decode)
- Use GPU if available (Metal on macOS, CUDA on Linux/Windows)
- Use GGUF models which are pre-quantized and optimized
- Disable PagedAttention for single requests: `MISTRALRS_PAGED_ATTN=false`

**Performance tip**: For models like Phi-3.5, use quantization to match Ollama's speed:
```bash
MISTRALRS_ISQ=Q4K spnl generate --backend mistralrs --model "microsoft/Phi-3.5-mini-instruct" "test"
```

### HTTP 404 errors
- For GGUF models, the backend automatically handles missing tokenizer.json files
- Ensure the model exists on HuggingFace
- Check your internet connection

## Resources

- [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs)
- [mistral.rs Documentation](https://ericlbuehler.github.io/mistral.rs/)
- [Supported Models](https://github.com/EricLBuehler/mistral.rs#supported-models)
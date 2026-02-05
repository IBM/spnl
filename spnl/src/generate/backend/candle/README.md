# Candle Backend

This directory contains the Candle-based inference backend for SPNL, providing local LLM inference capabilities using the [Candle](https://github.com/huggingface/candle) machine learning framework.

## Overview

The Candle backend enables running language models locally on your machine with optimized performance for Metal (macOS GPU) and CPU. It automatically downloads models from HuggingFace Hub and provides streaming text generation with progress tracking.

## Architecture

### Core Components

- **`mod.rs`** - Main entry point providing `generate_completion()` and `generate_chat()` functions
- **`model.rs`** - Defines the `CandleModel` trait that all model implementations must follow
- **`loader.rs`** - Model loading orchestration with automatic architecture detection
- **`config.rs`** - Generic configuration parsing for different model architectures
- **`download.rs`** - HuggingFace Hub integration with parallel downloads and caching
- **`progress.rs`** - Smart progress bars that only appear for slow downloads (>500ms)

### Supported Model Architectures

Each architecture has its own module with configuration mapping and model wrapper:

- **`llama.rs`** - Llama/Llama2/Llama3 models
- **`qwen2.rs`** - Qwen2 models with sliding window attention
- **`qwen3.rs`** - Qwen3 models with enhanced attention mechanisms
- **`qwen3_moe.rs`** - Qwen3 Mixture-of-Experts models (e.g., A3B)

## How It Works

### 1. Model Loading

When you request a model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`):

1. **Download Phase** (`download.rs`):
   - Checks local HuggingFace cache first
   - Downloads missing files: `tokenizer.json`, `config.json`, and model weights
   - Handles both single-file and sharded models
   - Uses parallel downloads (default: 8 concurrent) for sharded models
   - Shows progress bars only for downloads taking >500ms

2. **Architecture Detection** (`loader.rs`, `config.rs`):
   - Parses `config.json` to identify model architecture
   - Maps generic config to architecture-specific config
   - Detects MoE models by presence of `num_experts` field

3. **Model Initialization**:
   - Loads weights using memory-mapped safetensors for efficiency
   - Selects optimal device (Metal GPU on macOS, otherwise CPU)
   - Uses F16 precision on Metal for faster inference, F32 on CPU

### 2. Text Generation

The generation process (`mod.rs`):

1. **Tokenization**: Converts input text to token IDs using the model's tokenizer
2. **Configuration**: Sets up generation parameters (temperature, max_tokens, etc.)
3. **Streaming**: Generates tokens one at a time with optional callback for real-time output
4. **Progress Tracking**: Updates progress bars for multi-prompt generation
5. **Decoding**: Converts generated token IDs back to text

### 3. The CandleModel Trait

All model implementations must implement:

```rust
pub trait CandleModel: Send {
    fn generate(
        &mut self,
        tokens: &[u32],
        config: GenerateConfig,
        token_callback: Option<&mut TokenCallback>,
    ) -> anyhow::Result<String>;

    fn eos_token_id(&self, tokenizer: &Tokenizer) -> u32;
}
```

This abstraction allows adding new architectures without changing the core generation logic.

## Features

### Intelligent Caching
- Leverages HuggingFace Hub's local cache
- Avoids re-downloading already cached files
- Checks cache before showing progress bars

### Parallel Downloads
- Sharded models download multiple files concurrently
- Configurable via `MAX_CONCURRENT_DOWNLOADS` environment variable (default: 8)
- Overall progress bar tracks shard download completion

### Parallel Inference
- Batch generation uses a work queue with N parallel workers
- Each worker loads its own model instance for true concurrent inference
- Configurable via `CANDLE_NUM_PARALLEL` environment variable (default: 4)
- Workers pull tasks from a shared queue, enabling efficient resource utilization
- Particularly beneficial for processing multiple prompts or generating multiple completions

### Smart Progress Display
- Progress bars only appear for operations taking >500ms
- Prevents UI clutter for fast cached operations
- Separate bars for each file being downloaded

### Device Optimization
- Automatically uses Metal GPU on macOS when available
- Falls back to CPU on other platforms
- Defaults to F16 precision on Metal for 2x faster inference and 2x less memory
- Automatically retries with F32 if F16 fails with weight-related errors
- Can force F32 via `CANDLE_FORCE_F32=1` to skip F16 attempt

### Streaming Output
- Token-by-token generation with callbacks
- Real-time text display in non-quiet mode
- Colored output for better readability

## Usage

The Candle backend is automatically selected when you specify a local model path or use certain model identifiers. It's integrated into SPNL's generation pipeline and doesn't require direct interaction.

### Environment Variables

- `HF_TOKEN` - HuggingFace API token for accessing gated models
- `MAX_CONCURRENT_DOWNLOADS` - Number of concurrent downloads for sharded models (default: 8)
- `CANDLE_NUM_PARALLEL` - Number of parallel inference workers for batch generation (default: 4)
- `CANDLE_FORCE_F32` - Force F32 precision instead of trying F16 first (use if you know a model has F16 issues)

## Adding New Model Architectures

To add support for a new architecture:

1. Create a new module (e.g., `my_model.rs`)
2. Define a config struct that maps from `GenericConfig`
3. Implement a model wrapper struct
4. Implement the `CandleModel` trait
5. Add architecture detection logic in `loader.rs`
6. Export the new types in `mod.rs`

Example structure:
```rust
pub struct MyModelGenericConfig { /* fields */ }
impl From<MyModelGenericConfig> for MyModelConfig { /* conversion */ }

pub struct MyModelWrapper {
    model: MyModel,
    config: MyModelConfig,
}

impl CandleModel for MyModelWrapper {
    fn generate(/* ... */) -> anyhow::Result<String> { /* implementation */ }
    fn eos_token_id(&self, tokenizer: &Tokenizer) -> u32 { /* implementation */ }
}
```

## Performance Considerations

- **Metal GPU**: Significantly faster on Apple Silicon Macs
- **Memory-mapped weights**: Efficient loading without full memory copy
- **F16 precision**: Default on Metal for 2x memory reduction and faster inference
- **Automatic fallback**: If F16 fails with weight errors, automatically retries with F32
- **F32 precision**: Used on CPU or when F16 fails, or via `CANDLE_FORCE_F32=1`
- **Parallel downloads**: Faster initial model acquisition
- **Cached models**: Near-instant loading after first download

## Limitations

- Prepare mode not supported (returns error)
- Limited to architectures with Candle implementations
- GPU support currently limited to Metal (macOS)

## Dependencies

- `candle-core` - Core tensor operations
- `candle-nn` - Neural network layers
- `candle-transformers` - Pre-built transformer models
- `hf-hub` - HuggingFace Hub API client
- `tokenizers` - Fast tokenization
- `indicatif` - Progress bars

---

*Made with Bob*
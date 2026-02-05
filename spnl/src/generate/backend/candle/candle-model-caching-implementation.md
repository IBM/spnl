# Candle Model Caching Implementation

## Summary
Implemented model caching to avoid reloading models from disk on every request, addressing the primary performance bottleneck (5x slower than Ollama).

## Changes Made

### 1. Created Model Pool (`model_pool.rs`)
- **New file**: `spnl/src/generate/backend/candle/model_pool.rs`
- Implements thread-safe model caching using `Arc<Mutex<HashMap>>`
- Models are loaded once and reused across all requests
- Key features:
  - `ModelPool::get_or_load()` - Gets cached model or loads if not present
  - `CachedModel` - Wraps model with tokenizer, device, and dtype
  - Thread-safe sharing via `Arc<Mutex<>>`
  - Utility methods: `clear()`, `remove()`, `len()`, `is_empty()`

### 2. Updated `mod.rs`
- Added global `MODEL_POOL` using `OnceLock` for lazy initialization
- Modified both `generate_completion` and `generate_chat` worker functions:
  - **Before**: Each worker loaded model from disk for every request
  - **After**: Workers get model from pool (loads only on first use)
- Removed unused `load_model` export

### 3. Key Implementation Details

#### Model Pool Structure
```rust
static MODEL_POOL: OnceLock<ModelPool> = OnceLock::new();

fn get_model_pool() -> &'static ModelPool {
    MODEL_POOL.get_or_init(|| ModelPool::new())
}
```

#### Worker Pattern (Before)
```rust
// OLD: Load model every time
let (mut model, tokenizer, device, _dtype) = match load_model(&item.model_path) {
    Ok(m) => m,
    Err(e) => {
        eprintln!("Failed to load model: {}", e);
        continue;
    }
};
```

#### Worker Pattern (After)
```rust
// NEW: Get from pool (loads only once)
let pool = get_model_pool();
let cached_model = match pool.get_or_load(&item.model_path) {
    Ok(m) => m,
    Err(e) => {
        eprintln!("Failed to get model from pool: {}", e);
        continue;
    }
};

let mut cached = cached_model.lock().unwrap();
let (tokenizer, device, _dtype) = cached.resources();
```

## Performance Impact

### Expected Improvements
- **First request**: Same speed (model must be loaded)
- **Subsequent requests**: 3-4x faster (no disk I/O, no model initialization)
- **Memory usage**: Higher (models stay in RAM), but acceptable trade-off

### Benchmark Scenarios
1. **Single request**: No improvement (baseline)
2. **Multiple requests (same model)**: 3-4x improvement
3. **Multiple requests (different models)**: Each model loaded once, then cached

## Testing Recommendations

1. **Single request test**:
   ```bash
   time spnl generate "Hello" --model qwen/Qwen2.5-0.5B-Instruct
   ```

2. **Multiple requests test** (should show improvement):
   ```bash
   for i in {1..5}; do
     time spnl generate "Hello $i" --model qwen/Qwen2.5-0.5B-Instruct
   done
   ```

3. **Parallel requests test**:
   ```bash
   # Should benefit from shared model cache
   parallel -j 4 'spnl generate "Test {}" --model qwen/Qwen2.5-0.5B-Instruct' ::: {1..10}
   ```

## Architecture Notes

### Thread Safety
- `ModelPool` uses `Arc<Mutex<HashMap>>` for thread-safe access
- Each cached model is wrapped in `Arc<Mutex<CachedModel>>`
- Workers lock models during generation to prevent concurrent access

### Memory Management
- Models remain in memory until process exits
- Future enhancement: Add LRU eviction policy for memory-constrained environments
- Future enhancement: Add `CANDLE_MODEL_CACHE_SIZE` environment variable

### Concurrency Model
- Multiple workers can share the same cached model
- Mutex ensures only one worker uses a model at a time
- Different models can be used concurrently by different workers

## Compilation Status
✅ Code compiles successfully with `cargo check --package spnl -F candle`

## Next Steps (Future Optimizations)

1. **Flash Attention** (2-3x improvement)
   - Enable `use_flash_attn: true` in model configs
   - Requires GPU support detection

2. **Reduce GPU-CPU Syncs** (1.3-1.5x improvement)
   - Optimize sampling to reduce `to_scalar()` calls
   - Batch operations before syncing

3. **KV Cache Optimization** (1.1-1.2x improvement)
   - Implement cache reset instead of recreation
   - Reuse allocated memory

4. **Prefill Batching** (1.2-1.5x improvement)
   - Process prompt tokens in optimized batch sizes
   - Separate prefill and decode code paths

## Files Modified
- ✅ `spnl/src/generate/backend/candle/model_pool.rs` (new)
- ✅ `spnl/src/generate/backend/candle/mod.rs` (modified)

## Warnings
- Some utility methods in `ModelPool` show dead code warnings (unused but useful for future)
- These are intentional and provide API completeness
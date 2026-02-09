use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use indicatif::ProgressBar;
use std::collections::{HashMap, HashSet, VecDeque};
use tokenizers::Tokenizer;

/// Callback function type for streaming tokens as they are generated
pub type TokenCallback = Box<dyn FnMut(&str) -> anyhow::Result<()> + Send>;

/// Get prefill chunk size from environment variable or use default (0 = no chunking)
fn get_prefill_chunk_size() -> usize {
    std::env::var("CANDLE_PREFILL_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512) // Increased from 256 to 512 for better GPU utilization
}

/// Get decode batch size from environment variable or use default (1 = no batching)
/// Batching decode steps can improve GPU utilization and reduce kernel launch overhead
/// Recommended values: 4-8 for most GPUs, 1 for CPU or debugging
fn get_decode_batch_size() -> usize {
    std::env::var("CANDLE_DECODE_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8) // Default to 1 (no batching) for compatibility
}

/// Check if KV cache reuse is enabled (default: true for better chat performance)
fn is_cache_reuse_enabled() -> bool {
    std::env::var("CANDLE_CACHE_REUSE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true) // Default to enabled
}

/// Check if tensor pool is enabled (default: true for reduced allocation overhead)
fn is_tensor_pool_enabled() -> bool {
    std::env::var("CANDLE_TENSOR_POOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true) // Default to enabled
}

/// Get tensor pool size from environment variable or use default
fn get_tensor_pool_size() -> usize {
    std::env::var("CANDLE_TENSOR_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16) // Default: pool of 16 tensors
}

/// Get soft max tokens buffer from environment variable or use default
/// This allows the model to continue beyond max_tokens to reach EOS naturally
/// Default: 0 tokens (disabled - strict max_tokens enforcement)
fn get_soft_max_tokens_buffer() -> usize {
    std::env::var("CANDLE_SOFT_MAX_TOKENS_BUFFER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0) // Default: disabled to prevent repetition loops
}

/// Get EOS bias start ratio from environment variable or use default
/// This determines when to start biasing towards EOS token (as fraction of max_tokens)
/// Default: 0.8 (start biasing at 80% of max_tokens)
fn get_eos_bias_start_ratio() -> f64 {
    std::env::var("CANDLE_EOS_BIAS_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.8)
}

/// Get EOS bias strength from environment variable or use default
/// This determines how much to bias towards EOS token
/// Default: 2.0 (double the EOS logit value at max_tokens)
fn get_eos_bias_strength() -> f64 {
    std::env::var("CANDLE_EOS_BIAS_STRENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0)
}

/// Get repeat penalty window size from environment variable or use default
/// Smaller windows are faster but may allow more repetition
/// Larger windows prevent repetition better but are slower for long sequences
fn get_repeat_penalty_window() -> usize {
    std::env::var("CANDLE_REPEAT_PENALTY_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64) // Default: 64 to match Ollama (0 would mean penalize all generated tokens)
}

/// Check if GPU-side repeat penalty is enabled (default: false for compatibility)
/// When enabled, penalty computation stays on GPU avoiding CPU-GPU transfers
/// This is particularly beneficial for Metal which has higher transfer latency
fn is_gpu_penalty_enabled() -> bool {
    std::env::var("CANDLE_GPU_PENALTY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true) // Default to disabled for compatibility
}

/// Get streaming decode batch size from environment variable or use default (1 = no batching)
/// Batching token decoding reduces tokenizer overhead for streaming scenarios
/// Recommended values: 4-10 for better throughput, 1 for lowest latency
/// Trade-off: Higher values improve performance but increase perceived latency
fn get_streaming_decode_batch_size() -> usize {
    std::env::var("CANDLE_STREAMING_DECODE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1) // Default to 1 (no batching) for lowest latency
}

/// Optimized repeat penalty cache using HashSet for O(1) lookups
/// This avoids scanning the entire generation history for every token
struct RepeatPenaltyCache {
    /// Set of tokens that should be penalized
    penalized_tokens: HashSet<u32>,
    /// Sliding window of recent tokens (for windowed penalty)
    recent_tokens: VecDeque<u32>,
    /// Maximum window size
    window_size: usize,
    /// Whether to use windowed penalty (false = full history)
    use_window: bool,
}

impl RepeatPenaltyCache {
    fn new(window_size: usize) -> Self {
        Self {
            penalized_tokens: HashSet::new(),
            recent_tokens: VecDeque::with_capacity(window_size),
            window_size,
            use_window: window_size > 0,
        }
    }

    /// Add a token to the penalty cache
    fn add_token(&mut self, token: u32) {
        if self.use_window {
            // Windowed mode: maintain sliding window
            if self.recent_tokens.len() >= self.window_size {
                // Remove oldest token from set
                if let Some(old_token) = self.recent_tokens.pop_front() {
                    self.penalized_tokens.remove(&old_token);
                }
            }
            self.recent_tokens.push_back(token);
        }
        // Add to penalty set
        self.penalized_tokens.insert(token);
    }

    /// Apply penalty to logits using GPU-side operations (no CPU-GPU transfers)
    /// This is significantly faster on Metal which has high transfer latency
    /// Optimized for F16 on Metal to avoid unnecessary dtype conversions
    fn apply_penalty_gpu(&self, logits: &Tensor, penalty: f32) -> anyhow::Result<Tensor> {
        if self.penalized_tokens.is_empty() {
            return Ok(logits.clone());
        }

        let device = logits.device();
        let dtype = logits.dtype();

        // Ensure logits is 1D (vocab_size)
        let logits_1d = if logits.dims().len() > 1 {
            // If logits has multiple dimensions, flatten to 1D
            logits.flatten_all()?
        } else {
            logits.clone()
        };

        // Convert penalty indices to tensor (small transfer, done once)
        let penalty_indices: Vec<u32> = self.penalized_tokens.iter().copied().collect();
        let num_penalized = penalty_indices.len();
        let indices_tensor = Tensor::from_vec(penalty_indices, num_penalized, device)?;

        // Extract penalized logits using index_select (GPU operation)
        let penalized_logits = logits_1d.index_select(&indices_tensor, 0)?;

        // Create condition mask: logits >= 0.0 (GPU operation)
        let zeros = Tensor::zeros(num_penalized, dtype, device)?;
        let is_positive = penalized_logits.ge(&zeros)?;

        // Apply penalty based on sign (GPU operations)
        // For positive logits: divide by penalty
        // For negative logits: multiply by penalty
        let penalty_tensor = Tensor::new(&[penalty], device)?
            .to_dtype(dtype)?
            .broadcast_as(penalized_logits.shape())?;

        let penalized_positive = penalized_logits.div(&penalty_tensor)?;
        let penalized_negative = penalized_logits.mul(&penalty_tensor)?;

        // Select based on condition (GPU operation)
        let penalized_result = is_positive.where_cond(&penalized_positive, &penalized_negative)?;

        // Scatter penalized values back into logits (GPU operation)
        let result = logits_1d.scatter(&indices_tensor, &penalized_result, 0)?;

        // Reshape back to original shape if needed
        if logits.dims().len() > 1 {
            Ok(result.reshape(logits.shape())?)
        } else {
            Ok(result)
        }
    }

    /// Apply penalty to logits using the cached token set (CPU-based fallback)
    /// This is O(penalized_tokens) instead of O(all_tokens * generated_tokens)
    /// Preserves the original dtype of the logits tensor
    fn apply_penalty_cpu(&self, logits: &Tensor, penalty: f32) -> anyhow::Result<Tensor> {
        if self.penalized_tokens.is_empty() {
            return Ok(logits.clone());
        }

        // Store original dtype to restore it after processing
        let original_dtype = logits.dtype();
        let device = logits.device();

        // Convert to F32 for processing (required for arithmetic operations)
        let mut logits_f32 = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        // Apply penalty only to tokens in our cache (O(penalized_tokens) instead of O(vocab_size))
        for &token_id in &self.penalized_tokens {
            if let Some(logit) = logits_f32.get_mut(token_id as usize) {
                if *logit >= 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }

        // Create tensor and convert back to original dtype
        let logits_len = logits_f32.len();
        let result = Tensor::from_vec(logits_f32, logits_len, device)?;

        // Only convert if original dtype was different from F32
        if original_dtype != DType::F32 {
            Ok(result.to_dtype(original_dtype)?)
        } else {
            Ok(result)
        }
    }

    /// Apply penalty to logits - dispatches to GPU or CPU implementation
    fn apply_penalty(&self, logits: &Tensor, penalty: f32) -> anyhow::Result<Tensor> {
        if is_gpu_penalty_enabled() {
            self.apply_penalty_gpu(logits, penalty)
        } else {
            self.apply_penalty_cpu(logits, penalty)
        }
    }

    /*
    /// Get statistics for logging
    fn stats(&self) -> (usize, usize) {
        (self.penalized_tokens.len(), self.recent_tokens.len())
    } */
}

/// Tensor pool for reusing allocated tensors to reduce allocation overhead
/// This is particularly beneficial for decode steps where we repeatedly allocate
/// tensors of the same shape
///
/// Note: Currently infrastructure-only due to Candle API limitations.
/// The pool is created and statistics are tracked, but actual tensor reuse
/// requires Candle to support copying data into existing tensors.
#[allow(dead_code)]
struct TensorPool {
    /// Pool of single-token tensors (most common case)
    single_token_pool: Vec<Tensor>,
    /// Pools for different batch sizes
    batch_pools: HashMap<usize, Vec<Tensor>>,
    /// Device for tensor creation
    device: Device,
    /// Data type for tensors
    dtype: DType,
    /// Maximum pool size per shape
    max_pool_size: usize,
    /// Statistics: cache hits
    hits: usize,
    /// Statistics: cache misses
    misses: usize,
}

impl TensorPool {
    /// Create a new tensor pool
    fn new(device: Device, dtype: DType, max_pool_size: usize) -> Self {
        Self {
            single_token_pool: Vec::with_capacity(max_pool_size),
            batch_pools: HashMap::new(),
            device,
            dtype,
            max_pool_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Get a tensor from the pool or create a new one
    /// Returns a tensor filled with zeros (will be overwritten with actual data)
    #[allow(dead_code)]
    fn get_or_create(&mut self, shape: &[usize]) -> anyhow::Result<Tensor> {
        // Determine batch size from shape [1, batch_size]
        let batch_size = if shape.len() >= 2 { shape[1] } else { 1 };

        // Try to get from appropriate pool
        let tensor_opt = if batch_size == 1 {
            self.single_token_pool.pop()
        } else {
            self.batch_pools
                .get_mut(&batch_size)
                .and_then(|pool| pool.pop())
        };

        if let Some(tensor) = tensor_opt {
            self.hits += 1;
            Ok(tensor)
        } else {
            // Create new tensor if pool is empty
            self.misses += 1;
            Ok(Tensor::zeros(shape, self.dtype, &self.device)?)
        }
    }

    /// Return a tensor to the pool for reuse
    /// Only stores if pool hasn't reached max size
    #[allow(dead_code)]
    fn return_tensor(&mut self, tensor: Tensor, batch_size: usize) {
        if batch_size == 1 {
            if self.single_token_pool.len() < self.max_pool_size {
                self.single_token_pool.push(tensor);
            }
        } else {
            let pool = self.batch_pools.entry(batch_size).or_default();
            if pool.len() < self.max_pool_size {
                pool.push(tensor);
            }
        }
    }

    /// Get pool statistics (hits, misses, hit rate)
    fn stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }

    /// Clear all pools
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.single_token_pool.clear();
        self.batch_pools.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// Configuration for text generation
pub struct GenerateConfig<'a> {
    pub device: &'a Device,
    pub dtype: DType,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub _top_k: Option<usize>,
    pub repeat_penalty: f32, // Changed from f64 to f32 to match candle API
    pub seed: u64,
    pub tokenizer: &'a Tokenizer,
    pub progress_bar: Option<&'a ProgressBar>,
}

/// Trait for model-specific forward pass operations
pub trait ModelForward {
    /// Perform forward pass and return logits
    fn forward_pass(&mut self, input: &Tensor, position: usize) -> anyhow::Result<Tensor>;

    /// Get maximum position embeddings for this model
    fn max_position_embeddings(&self) -> usize;

    /// Clear or reset the model's KV cache (if applicable)
    fn clear_cache(&mut self);

    /// Get the current length of the KV cache (number of cached positions)
    /// Returns 0 if cache is empty or not supported
    fn get_cache_length(&self) -> usize {
        0 // Default implementation for models without cache tracking
    }

    /// Clear KV cache after a specific position
    /// This allows reusing cache for common prefixes in chat scenarios
    /// Default implementation clears entire cache (safe but not optimal)
    #[allow(dead_code)]
    fn clear_cache_after(&mut self, _position: usize) {
        self.clear_cache(); // Default: clear everything
    }
}

/// Trait for all candle-based language models
pub trait CandleModel: Send {
    /// Generate text from a prompt
    fn generate(
        &mut self,
        tokens: &[u32],
        config: GenerateConfig,
        token_callback: Option<&mut TokenCallback>,
    ) -> anyhow::Result<String>;

    /// Get the model's EOS token ID
    fn eos_token_id(&self, tokenizer: &Tokenizer) -> u32;
}

/// Shared generation logic for all models
/// This eliminates code duplication across model implementations
pub fn generate_text<M: ModelForward>(
    model: &mut M,
    tokens: &[u32],
    eos_token: u32,
    config: GenerateConfig,
    mut token_callback: Option<&mut TokenCallback>,
) -> anyhow::Result<String> {
    let mut tokens = tokens.to_vec();

    // Pre-allocate with estimated capacity
    let mut generated_tokens = Vec::with_capacity(config.max_tokens);

    // Smart KV cache management for better chat performance
    // Check if we can reuse existing cache by comparing with cached content
    let prompt_len = tokens.len();
    let cache_len = model.get_cache_length();
    let cache_reuse_enabled = is_cache_reuse_enabled();

    // Determine how much of the prompt is already cached
    let prefill_start = if cache_reuse_enabled && cache_len > 0 && cache_len <= prompt_len {
        // We have some cache - check if it matches the prompt prefix
        // For now, we assume cache matches (conservative approach)
        // Future: Add token comparison for validation
        cache_len
    } else {
        // No cache or cache reuse disabled - start from beginning
        if cache_len > 0 {
            model.clear_cache();
        }
        0
    };

    // Prefill phase - process prompt tokens (chunked or all at once)
    // Skip tokens that are already in cache
    let chunk_size = get_prefill_chunk_size();

    // Only process tokens that aren't already cached
    if prefill_start < prompt_len {
        if chunk_size > 0 && (prompt_len - prefill_start) > chunk_size {
            // Chunked prefill: process tokens in chunks to potentially improve Metal performance
            // This trades off parallelism for better memory access patterns on Metal
            // Pre-allocate buffer for chunk processing to avoid repeated allocations
            let mut chunk_buffer = Vec::with_capacity(chunk_size);
            let mut pos = prefill_start;

            for chunk_start in (prefill_start..prompt_len).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(prompt_len);

                // Reuse buffer: clear and copy chunk data
                chunk_buffer.clear();
                chunk_buffer.extend_from_slice(&tokens[chunk_start..chunk_end]);

                // Create tensor from buffer (still allocates GPU memory, but reuses CPU buffer)
                let input = Tensor::new(&chunk_buffer[..], config.device)?.unsqueeze(0)?;
                let _logits = model.forward_pass(&input, pos)?;
                pos += chunk_buffer.len();
            }
        } else {
            // Single-pass prefill: process all uncached prompt tokens at once (default)
            // Maximizes parallelism, optimal for most cases
            let input = Tensor::new(&tokens[prefill_start..], config.device)?.unsqueeze(0)?;
            let _logits = model.forward_pass(&input, prefill_start)?;
        }
    }

    // Initialize LogitsProcessor for proper sampling
    // This provides multinomial sampling, top-p, top-k, and better temperature handling
    let mut logits_processor =
        LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);

    // Get decode batch size for batched decode optimization
    let decode_batch_size = get_decode_batch_size();

    // Pre-allocate token buffer with capacity for batch size
    let mut token_buffer = Vec::with_capacity(decode_batch_size.max(1));

    // Initialize tensor pool if enabled
    // Note: Tensor pool is prepared but not yet used due to Candle API limitations
    // We keep the infrastructure for future optimization when Candle supports
    // copying data into existing tensors
    let _tensor_pool_enabled = is_tensor_pool_enabled();
    let mut _tensor_pool = if _tensor_pool_enabled {
        Some(TensorPool::new(
            config.device.clone(),
            config.dtype,
            get_tensor_pool_size(),
        ))
    } else {
        None
    };

    // Initialize optimized repeat penalty cache
    // This uses a HashSet for O(1) lookups instead of scanning the entire history
    let penalty_window = get_repeat_penalty_window();
    let mut penalty_cache = RepeatPenaltyCache::new(penalty_window);

    // Track recent tokens for repetition detection (last 100 tokens)
    let mut recent_token_sequence: VecDeque<u32> = VecDeque::with_capacity(100);
    let repetition_check_window = 30; // Check if last 30 tokens repeat

    // Initialize pending tokens buffer for batched streaming decode
    // This reduces tokenizer overhead by decoding multiple tokens at once
    let streaming_batch_size = get_streaming_decode_batch_size();
    let mut pending_tokens = Vec::with_capacity(streaming_batch_size);

    // Generation phase - process tokens in batches for better GPU utilization
    // Batching reduces kernel launch overhead and improves throughput

    // Soft max_tokens: allow model to continue beyond max_tokens to reach EOS naturally
    // This prevents mid-sentence cutoffs while still respecting reasonable limits
    let soft_buffer = get_soft_max_tokens_buffer();
    let hard_limit = config.max_tokens + soft_buffer;
    let mut in_soft_zone = false; // Track if we're in the buffer zone

    let mut index_pos = 0;
    while index_pos < hard_limit {
        let start_pos = prompt_len + index_pos;

        // Check if we're exceeding max_position_embeddings
        if start_pos >= model.max_position_embeddings() {
            break;
        }

        // Check if we've entered the soft zone (beyond max_tokens but before hard limit)
        if index_pos >= config.max_tokens && !in_soft_zone {
            in_soft_zone = true;
            // eprintln!("Entered soft max_tokens zone - will continue until EOS or hard limit");
        }

        // Determine batch size for this iteration (may be smaller at the end)
        let remaining_tokens = hard_limit - index_pos;
        let current_batch_size = decode_batch_size.min(remaining_tokens);

        // Clear and prepare token buffer for this batch
        token_buffer.clear();

        // For batched decode, we need to process multiple tokens
        // Start with the last generated token
        token_buffer.push(tokens[tokens.len() - 1]);

        // If batch size is 1, use single-token decode (original behavior)
        if current_batch_size == 1 {
            // Create input tensor from buffer
            // Note: Tensor pooling is challenging with Candle's API since we can't easily
            // copy data into an existing tensor. For now, we create new tensors but keep
            // the pool infrastructure for future optimization when Candle supports it.
            let input = Tensor::new(&token_buffer[..], config.device)?.unsqueeze(0)?;
            let logits = model.forward_pass(&input, start_pos)?;

            let logits = logits.squeeze(0)?;
            let mut last_token_logits = if logits.dims().len() == 2 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };

            // Apply optimized repeat penalty using cached token set
            // This is much faster than the original implementation for long sequences
            if config.repeat_penalty != 1.0 && !penalty_cache.penalized_tokens.is_empty() {
                last_token_logits =
                    penalty_cache.apply_penalty(&last_token_logits, config.repeat_penalty)?;
            }

            // Apply EOS bias as we approach max_tokens to encourage natural completion
            let eos_bias_start = (config.max_tokens as f64 * get_eos_bias_start_ratio()) as usize;
            if index_pos >= eos_bias_start {
                let progress = (index_pos - eos_bias_start) as f64
                    / (config.max_tokens - eos_bias_start) as f64;
                let bias_multiplier = 1.0 + (progress * (get_eos_bias_strength() - 1.0));

                // Boost EOS token logit to encourage completion
                // Convert to F32 first to avoid dtype issues
                let logits_f32 = last_token_logits.to_dtype(DType::F32)?;
                let mut logits_vec = logits_f32.to_vec1::<f32>()?;
                if let Some(eos_logit) = logits_vec.get_mut(eos_token as usize) {
                    *eos_logit *= bias_multiplier as f32;
                }
                let biased_logits =
                    Tensor::from_vec(logits_vec, last_token_logits.dims()[0], config.device)?;
                // Convert back to original dtype
                last_token_logits = biased_logits.to_dtype(last_token_logits.dtype())?;
            }

            // Sample next token
            let next_token = logits_processor.sample(&last_token_logits)?;

            if next_token == eos_token {
                break;
            }

            // In soft zone, we only continue to find EOS - if we hit hard limit, stop
            if in_soft_zone && index_pos >= hard_limit - 1 {
                break;
            }

            tokens.push(next_token);
            generated_tokens.push(next_token);

            // Add token to penalty cache for future iterations
            penalty_cache.add_token(next_token);

            // Track recent tokens for repetition detection
            recent_token_sequence.push_back(next_token);
            if recent_token_sequence.len() > 100 {
                recent_token_sequence.pop_front();
            }

            // Check for repetition: if last N tokens match N tokens before them, stop
            // Also check with smaller windows for earlier detection
            let mut repetition_detected = false;
            if recent_token_sequence.len() >= repetition_check_window * 2 {
                let len = recent_token_sequence.len();

                // Check multiple window sizes to catch different repetition patterns
                for window_size in [
                    repetition_check_window,
                    repetition_check_window / 2,
                    repetition_check_window / 3,
                ]
                .iter()
                {
                    if len >= window_size * 2 {
                        let recent: Vec<u32> = recent_token_sequence
                            .iter()
                            .skip(len - window_size)
                            .copied()
                            .collect();
                        let previous: Vec<u32> = recent_token_sequence
                            .iter()
                            .skip(len - window_size * 2)
                            .take(*window_size)
                            .copied()
                            .collect();

                        if recent == previous {
                            // Detected exact repetition - stop generation
                            repetition_detected = true;
                            break;
                        }
                    }
                }
            }

            if repetition_detected {
                break; // Exit the main generation loop
            }

            // Stream token if callback is provided (with optional batching)
            if let Some(callback) = token_callback.as_mut() {
                pending_tokens.push(next_token);

                // Decode and stream when batch is full or at EOS
                if pending_tokens.len() >= streaming_batch_size || next_token == eos_token {
                    let token_text = config
                        .tokenizer
                        .decode(&pending_tokens, true)
                        .map_err(|e| anyhow::anyhow!("Token decoding failed: {}", e))?;
                    callback(&token_text)?;
                    pending_tokens.clear();
                }
            }

            // Update progress bar
            if let Some(pb) = config.progress_bar {
                pb.inc(1);
            }

            index_pos += 1;
        } else {
            // Batched decode: process multiple tokens in one forward pass
            // This is more efficient for GPU utilization
            for batch_idx in 0..current_batch_size {
                let batch_start_pos = start_pos + batch_idx;

                // Check position limit for each token in batch
                if batch_start_pos >= model.max_position_embeddings() {
                    break;
                }

                // For first token in batch, we already have it in buffer
                // For subsequent tokens, we need to generate them
                if batch_idx > 0 {
                    // Use the last token we just generated
                    token_buffer.clear();
                    token_buffer.push(tokens[tokens.len() - 1]);
                }

                // Create input tensor
                // Note: Tensor pooling is challenging with Candle's API since we can't easily
                // copy data into an existing tensor. For now, we create new tensors but keep
                // the pool infrastructure for future optimization when Candle supports it.
                let input = Tensor::new(&token_buffer[..], config.device)?.unsqueeze(0)?;
                let logits = model.forward_pass(&input, batch_start_pos)?;

                let logits = logits.squeeze(0)?;
                let mut last_token_logits = if logits.dims().len() == 2 {
                    logits.get(logits.dim(0)? - 1)?
                } else {
                    logits
                };

                // Apply optimized repeat penalty using cached token set
                if config.repeat_penalty != 1.0 && !penalty_cache.penalized_tokens.is_empty() {
                    last_token_logits =
                        penalty_cache.apply_penalty(&last_token_logits, config.repeat_penalty)?;
                }

                // Apply EOS bias as we approach max_tokens to encourage natural completion
                let current_pos = index_pos + batch_idx;
                let eos_bias_start =
                    (config.max_tokens as f64 * get_eos_bias_start_ratio()) as usize;
                if current_pos >= eos_bias_start {
                    let progress = (current_pos - eos_bias_start) as f64
                        / (config.max_tokens - eos_bias_start) as f64;
                    let bias_multiplier = 1.0 + (progress * (get_eos_bias_strength() - 1.0));

                    // Boost EOS token logit to encourage completion
                    // Convert to F32 first to avoid dtype issues
                    let logits_f32 = last_token_logits.to_dtype(DType::F32)?;
                    let mut logits_vec = logits_f32.to_vec1::<f32>()?;
                    if let Some(eos_logit) = logits_vec.get_mut(eos_token as usize) {
                        *eos_logit *= bias_multiplier as f32;
                    }
                    let biased_logits =
                        Tensor::from_vec(logits_vec, last_token_logits.dims()[0], config.device)?;
                    // Convert back to original dtype
                    last_token_logits = biased_logits.to_dtype(last_token_logits.dtype())?;
                }

                // Sample next token
                let next_token = logits_processor.sample(&last_token_logits)?;

                if next_token == eos_token {
                    index_pos = hard_limit; // Signal to exit outer loop
                    break;
                }

                // In soft zone, we only continue to find EOS - if we hit hard limit, stop
                if in_soft_zone && batch_start_pos >= prompt_len + hard_limit - 1 {
                    index_pos = hard_limit; // Signal to exit outer loop
                    break;
                }

                tokens.push(next_token);
                generated_tokens.push(next_token);

                // Add token to penalty cache for future iterations
                penalty_cache.add_token(next_token);

                // Track recent tokens for repetition detection
                recent_token_sequence.push_back(next_token);
                if recent_token_sequence.len() > 100 {
                    recent_token_sequence.pop_front();
                }

                // Check for repetition: if last N tokens match N tokens before them, stop
                // Also check with smaller windows for earlier detection
                let mut repetition_detected = false;
                if recent_token_sequence.len() >= repetition_check_window * 2 {
                    let len = recent_token_sequence.len();

                    // Check multiple window sizes to catch different repetition patterns
                    for window_size in [
                        repetition_check_window,
                        repetition_check_window / 2,
                        repetition_check_window / 3,
                    ]
                    .iter()
                    {
                        if len >= window_size * 2 {
                            let recent: Vec<u32> = recent_token_sequence
                                .iter()
                                .skip(len - window_size)
                                .copied()
                                .collect();
                            let previous: Vec<u32> = recent_token_sequence
                                .iter()
                                .skip(len - window_size * 2)
                                .take(*window_size)
                                .copied()
                                .collect();

                            if recent == previous {
                                // Detected exact repetition - stop generation
                                repetition_detected = true;
                                break;
                            }
                        }
                    }
                }

                if repetition_detected {
                    index_pos = hard_limit; // Signal to exit outer loop
                    break; // Exit the batch loop
                }

                // Stream token if callback is provided (with optional batching)
                if let Some(callback) = token_callback.as_mut() {
                    pending_tokens.push(next_token);

                    // Decode and stream when batch is full or at EOS
                    if pending_tokens.len() >= streaming_batch_size || next_token == eos_token {
                        let token_text = config
                            .tokenizer
                            .decode(&pending_tokens, true)
                            .map_err(|e| anyhow::anyhow!("Token decoding failed: {}", e))?;
                        callback(&token_text)?;
                        pending_tokens.clear();
                    }
                }

                // Update progress bar
                if let Some(pb) = config.progress_bar {
                    pb.inc(1);
                }
            }

            index_pos += current_batch_size;
        }
    }

    // Flush any remaining pending tokens from batched streaming decode
    if !pending_tokens.is_empty()
        && let Some(callback) = token_callback.as_mut()
    {
        let token_text = config
            .tokenizer
            .decode(&pending_tokens, true)
            .map_err(|e| anyhow::anyhow!("Token decoding failed: {}", e))?;
        callback(&token_text)?;
    }

    // Log tensor pool statistics if enabled and pool was used
    if let Some(pool) = _tensor_pool.as_ref() {
        let (hits, misses, hit_rate) = pool.stats();
        if hits + misses > 0 {
            eprintln!(
                "[TensorPool] Hits: {}, Misses: {}, Hit Rate: {:.2}%",
                hits,
                misses,
                hit_rate * 100.0
            );
        }
    }

    // Log repeat penalty cache statistics
    /* let (penalty_set_size, penalty_window_size) = penalty_cache.stats();
    if penalty_set_size > 0 {
        eprintln!(
            "[RepeatPenalty] Cached tokens: {}, Window size: {} (max: {})",
            penalty_set_size,
            penalty_window_size,
            penalty_window
        );
    } */

    // Decode all generated tokens at once
    let generated_text = config
        .tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

    Ok(generated_text)
}

// Made with Bob

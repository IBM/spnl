use candle_core::{DType, Device};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

use super::{CandleModel, loader::load_model};

/// A cached model with its associated resources
pub struct CachedModel {
    model: Box<dyn CandleModel>,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

/// Type alias for a shared, thread-safe model instance
type SharedModel = Arc<Mutex<CachedModel>>;

/// Type alias for the model pool storage
type ModelStorage = Arc<Mutex<HashMap<String, Vec<SharedModel>>>>;

/// Thread-safe model pool that maintains multiple instances per model path
/// This allows parallel processing by giving each worker its own model instance
pub struct ModelPool {
    // Maps model_path -> Vec of available model instances
    models: ModelStorage,
}

impl ModelPool {
    /// Create a new empty model pool
    pub fn new() -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get a model from the pool, loading a new instance if all are busy
    /// This allows multiple workers to process requests in parallel
    pub fn get_or_load(&self, model_path: &str) -> anyhow::Result<SharedModel> {
        let mut models = self.models.lock().unwrap();

        // Check if we have any instances for this model path
        if let Some(instances) = models.get_mut(model_path) {
            // Try to find an available (unlocked) instance
            for instance in instances.iter() {
                if let Ok(_guard) = instance.try_lock() {
                    // Found an available instance
                    return Ok(Arc::clone(instance));
                }
            }

            // All instances are busy, load a new one for parallel processing
            let (model, tokenizer, device, dtype) = load_model(model_path)?;
            let new_instance = Arc::new(Mutex::new(CachedModel {
                model,
                tokenizer,
                device,
                dtype,
            }));

            instances.push(Arc::clone(&new_instance));
            return Ok(new_instance);
        }

        // First time loading this model path
        let (model, tokenizer, device, dtype) = load_model(model_path)?;
        let instance = Arc::new(Mutex::new(CachedModel {
            model,
            tokenizer,
            device,
            dtype,
        }));

        models.insert(model_path.to_string(), vec![Arc::clone(&instance)]);
        Ok(instance)
    }

    /*
    /// Clear all cached models to free memory
    pub fn clear(&self) {
        let mut models = self.models.lock().unwrap();
        models.clear();
    }

    /// Remove a specific model from the cache
    pub fn remove(&self, model_path: &str) -> bool {
        let mut models = self.models.lock().unwrap();
        models.remove(model_path).is_some()
    }

    /// Get the number of cached models
    pub fn len(&self) -> usize {
        let models = self.models.lock().unwrap();
        models.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    } */
}

impl Default for ModelPool {
    fn default() -> Self {
        Self::new()
    }
}

impl CachedModel {
    /// Generate text using the cached model
    pub fn generate(
        &mut self,
        tokens: &[u32],
        config: super::model::GenerateConfig,
        token_callback: Option<&mut super::model::TokenCallback>,
    ) -> anyhow::Result<String> {
        self.model.generate(tokens, config, token_callback)
    }

    /// Get cloned references to the model's resources
    /// Returns clones to avoid borrow checker issues
    pub fn resources(&self) -> (Tokenizer, Device, DType) {
        (self.tokenizer.clone(), self.device.clone(), self.dtype)
    }
}

// Made with Bob

use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Arc;

use super::download_progress::DownloadProgress;

/// Download model files from HuggingFace Hub
/// Returns (tokenizer_path, config_path, weight_files)
pub fn download_model_files(model_id: &str) -> anyhow::Result<(PathBuf, PathBuf, Vec<PathBuf>)> {
    // Configure API with token from environment if available
    let api = if let Ok(token) = std::env::var("HF_TOKEN") {
        ApiBuilder::new().with_token(Some(token)).build()?
    } else {
        ApiBuilder::new().build()?
    };
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Create a MultiProgress for all downloads
    let mp = MultiProgress::new();
    let repo = Arc::new(repo);

    // Create cache instance for checking if files are already downloaded
    let cache = Cache::default();
    let cache_repo = cache.repo(Repo::new(model_id.to_string(), RepoType::Model));
    let model_id = model_id.to_string();

    // Load tokenizer - check cache first, download with progress if needed
    let tokenizer_path = match cache_repo.get("tokenizer.json") {
        Some(path) => path,
        None => {
            let progress = DownloadProgress::new(&mp, "tokenizer.json");
            repo.download_with_progress("tokenizer.json", progress)?
        }
    };

    // Load model config - check cache first, download with progress if needed
    let config_path = match cache_repo.get("config.json") {
        Some(path) => path,
        None => {
            let progress = DownloadProgress::new(&mp, "config.json");
            repo.download_with_progress("config.json", progress)?
        }
    };

    // Load weights - handle both single file and sharded models
    let filenames = match cache_repo.get("model.safetensors") {
        Some(single_file) => {
            // Single file model already cached
            vec![single_file]
        }
        None => {
            // Try to download single file model with progress
            let progress = DownloadProgress::new(&mp, "model.safetensors");
            match repo.download_with_progress("model.safetensors", progress) {
                Ok(single_file) => vec![single_file],
                Err(_) => {
                    // Sharded model - download all shards
                    download_sharded_model(&repo, &cache, &mp, &model_id)?
                }
            }
        }
    };

    // Clear the MultiProgress to ensure all progress bars are removed
    mp.clear().ok();

    Ok((tokenizer_path, config_path, filenames))
}

/// Download sharded model files
fn download_sharded_model(
    repo: &Arc<hf_hub::api::sync::ApiRepo>,
    cache: &Cache,
    mp: &MultiProgress,
    model_id: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    // Get the cache repo for this model
    let cache_repo = cache.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Check cache first for index file
    let json_file = match cache_repo.get("model.safetensors.index.json") {
        Some(path) => path,
        None => {
            let progress = DownloadProgress::new(mp, "model.safetensors.index.json");
            repo.download_with_progress("model.safetensors.index.json", progress)?
        }
    };
    let json_content = std::fs::read_to_string(&json_file)?;
    let json: serde_json::Value = serde_json::from_str(&json_content)?;

    let weight_map = json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            anyhow::anyhow!("Invalid model.safetensors.index.json: missing weight_map")
        })?;

    // Collect unique safetensors files
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }

    let files: Vec<_> = safetensors_files.into_iter().collect();

    // Check cache to see how many shard files are already downloaded
    let mut already_cached = 0usize;
    for filename in &files {
        if cache_repo.get(filename).is_some() {
            already_cached += 1;
        }
    }

    // Only create overall progress bar if we need to download files
    let overall_bar = if already_cached < files.len() {
        let bar = mp.add(ProgressBar::new(files.len() as u64));
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.green/blue}] {pos}/{len} shards")
                .unwrap()
                .progress_chars("=>-"),
        );
        bar.set_position(already_cached as u64);
        bar.set_message("Loading model");
        Some(bar)
    } else {
        // All files cached, no progress bar needed
        None
    };

    // Download shard files in parallel with configurable concurrent downloads
    let max_concurrent = std::env::var("MAX_CONCURRENT_DOWNLOADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);

    let mp = Arc::new(mp.clone());
    let overall_bar = Arc::new(overall_bar);
    let cache_repo = Arc::new(cache_repo);

    use std::sync::{Mutex, mpsc};
    let (tx, rx) = mpsc::channel::<String>();
    let rx = Arc::new(Mutex::new(rx));

    // Spawn worker threads
    let mut handles = Vec::new();
    for _ in 0..max_concurrent {
        let rx = Arc::clone(&rx);
        let repo = Arc::clone(repo);
        let mp = Arc::clone(&mp);
        let overall_bar = Arc::clone(&overall_bar);
        let cache_repo = Arc::clone(&cache_repo);

        let handle = std::thread::spawn(move || {
            let mut results = Vec::new();
            loop {
                let filename = match rx.lock().unwrap().recv() {
                    Ok(f) => f,
                    Err(_) => break, // Channel closed
                };

                // Only download with progress if not cached
                let result = match cache_repo.get(&filename) {
                    Some(_) => {
                        // File is cached, use fast get without progress
                        // Don't increment progress bar since it was already counted in already_cached
                        repo.get(&filename)
                    }
                    None => {
                        // File not cached, download with progress
                        let progress = DownloadProgress::new(&mp, &filename);
                        let download_result = repo.download_with_progress(&filename, progress);
                        // Only increment for newly downloaded files
                        if let Some(bar) = overall_bar.as_ref() {
                            bar.inc(1);
                        }
                        download_result
                    }
                };

                results.push(result);
            }
            results
        });
        handles.push(handle);
    }

    // Send all files to the work queue
    for filename in files {
        tx.send(filename).unwrap();
    }
    drop(tx); // Close the channel to signal workers to finish

    // Collect results from all workers
    let mut paths = Vec::new();
    for handle in handles {
        let results = handle
            .join()
            .map_err(|_| anyhow::anyhow!("Thread panicked"))?;
        for result in results {
            paths.push(result?);
        }
    }

    if let Some(bar) = overall_bar.as_ref() {
        bar.finish_and_clear();
    }
    Ok(paths)
}

// Made with Bob

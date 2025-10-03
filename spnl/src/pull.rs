use fs4::fs_std::FileExt;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{Generate, Query};

/// Pull models (in parallel, if needed) used by the given query
pub async fn pull_if_needed(query: &Query) -> anyhow::Result<()> {
    futures::future::try_join_all(
        extract_models(query)
            .iter()
            .map(String::as_str)
            .map(pull_model_if_needed),
    )
    .await?;

    Ok(())
}

/// Pull the given model, if needed
async fn pull_model_if_needed(model: &str) -> anyhow::Result<()> {
    match model {
        m if model.starts_with("ollama/") => ollama_pull_if_needed(&m[7..]).await,
        m if model.starts_with("ollama_chat/") => ollama_pull_if_needed(&m[12..]).await,
        _ => Ok(()),
    }
}

#[derive(serde::Deserialize)]
struct OllamaModel {
    model: String,
}

#[derive(serde::Deserialize)]
struct OllamaTags {
    models: Vec<OllamaModel>,
}

// struct to hold request params
#[derive(serde::Serialize)]
struct PullRequest {
    model: String,
    insecure: Option<bool>,
    stream: Option<bool>,
}

// struct to hold response params
#[derive(Debug, serde::Deserialize)]
struct PullResponse {
    status: String,
    total: Option<u64>,
    completed: Option<u64>,
}

async fn ollama_exists(model: &str) -> anyhow::Result<bool> {
    let tags: OllamaTags = reqwest::get("http://localhost:11434/api/tags")
        .await?
        .json()
        .await?;
    Ok(tags.models.into_iter().any(|m| m.model == model))
}

// The Ollama implementation of a single model pull
async fn ollama_pull_if_needed(model: &str) -> anyhow::Result<()> {
    // don't ? the cmd! so that we can "finally" unlock the file
    if !ollama_exists(model).await? {
        let path = ::std::env::temp_dir().join(format!("ollama-pull-{model}"));
        let f = ::std::fs::File::create(&path)?;
        /*f.lock_exclusive()?;
        if !ollama_exists(model).await?*/
        {
            // create new MultiProgress bar
            let m = MultiProgress::new();
            let style = ProgressStyle::with_template(
                "{msg} {wide_bar:.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]",
            )?;

            // creating client and request body
            let http_client = reqwest::Client::new();
            let request_body = PullRequest {
                model: model.to_string(),
                insecure: Some(false),
                stream: Some(true),
            };

            // receiving response and error handling
            let response = http_client
                .post("http://localhost:11434/api/pull")
                .json(&request_body)
                .send()
                .await?;
            if !response.status().is_success() {
                eprintln!("API request failed with status: {}", response.status(),);
                return Err(anyhow::anyhow!("Ollama API request failed"));
            }

            // creating streaming structure
            let byte_stream = response
                .bytes_stream()
                .map(|r| r.map_err(|e| std::io::Error::other(e)));
            let stream_reader = tokio_util::io::StreamReader::new(byte_stream);
            let buf_reader = BufReader::new(stream_reader);
            let mut lines = buf_reader.lines();

            let pb = m.add(ProgressBar::new(0));
            pb.set_style(style);

            while let Some(line) = lines.next_line().await? {
                // stores in pull response struct
                let update: PullResponse = match serde_json::from_str(&line) {
                    Ok(u) => u,
                    Err(e) => {
                        return Err(anyhow::anyhow!("Invalid JSON in PullResponse: {e}"));
                    }
                };

                // checks for error or end of stream
                if update.status.to_lowercase() == "error" {
                    pb.finish_and_clear();
                    FileExt::unlock(&f)?;
                    return Err(anyhow::anyhow!("Ollama streaming error: "));
                } else if update.status.to_lowercase() == "success" {
                    pb.finish_with_message(format!("Model {model} pulled successfully"));
                    break;
                }

                // sets progress bar length
                pb.set_message(update.status.to_lowercase());
                if let (Some(total), Some(done)) = (update.total, update.completed) {
                    if total == done {
                        pb.set_length(total);
                        pb.set_position(done);
                    }
                    if pb.length().unwrap_or(0) == 0 {
                        pb.set_length(total);
                    }
                    pb.set_position(done);
                }
            }
            FileExt::unlock(&f)?;
        }
    }

    Ok(())
}

/// Extract models referenced by the query
pub fn extract_models(query: &Query) -> Vec<String> {
    let mut models = vec![];
    extract_models_iter(query, &mut models);

    // A single query may specify the same model more than once. Dedup!
    models.sort();
    models.dedup();

    models
}

/// Produce a vector of the models used by the given `query`
fn extract_models_iter(query: &Query, models: &mut Vec<String>) {
    match query {
        #[cfg(feature = "rag")]
        Query::Augment(crate::Augment {
            embedding_model, ..
        }) => models.push(embedding_model.clone()),
        Query::Generate(Generate { model, .. }) => models.push(model.clone()),
        Query::Plus(v) | Query::Cross(v) => {
            v.iter().for_each(|vv| extract_models_iter(vv, models));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    // testing a valid model pull
    #[tokio::test]
    async fn test_pull_local_ollama() {
        let result = ollama_pull_if_needed("qwen:0.5b").await;
        assert!(result.is_ok());
    }

    // testing invalid model pull
    #[tokio::test]
    async fn test_pull_invalid_model() {
        let result = ollama_pull_if_needed("notamodel").await;
        assert!(result.is_err());
    }
}

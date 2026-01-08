use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::ir::{Generate, GenerateMetadata, Query};

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
pub async fn pull_model_if_needed(model: &str) -> anyhow::Result<()> {
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

#[derive(serde::Serialize)]
struct DeleteRequest {
    model: String,
    name: String,
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
struct ValidPullResponse {
    status: String,
    digest: Option<String>,
    total: Option<u64>,
    completed: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct InvalidPullResponse {
    error: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum PullResponse {
    Ok(ValidPullResponse),
    Err(InvalidPullResponse),
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
    let mut err: Option<anyhow::Error> = None;
    for _ in 0..5 {
        if !ollama_exists(model).await? {
            // creating client and request body
            let http_client = reqwest::Client::new();
            let request_body = PullRequest {
                model: model.to_string(),
                insecure: Some(false),
                stream: Some(true),
            };

            // receiving response and error handling
            let response = http_client
                .post("http://localhost:11434/api/pull") // TODO use OLLAMA_API_BASE?
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
                .map(|r| r.map_err(std::io::Error::other));
            let stream_reader = tokio_util::io::StreamReader::new(byte_stream);
            let buf_reader = BufReader::new(stream_reader);
            let mut lines = buf_reader.lines();

            // creation of multiprogress container and style
            let m = MultiProgress::new();
            let style =
                ProgressStyle::with_template("{msg:<20} {percent:>3}% ▕{wide_bar}▏ {bytes:>7}")
                    .expect("Failed to create progress style template")
                    .progress_chars("█ ");
            let mut digests: HashMap<String, ProgressBar> = HashMap::new();
            let mut final_status_lines: Vec<String> = Vec::new();

            while let Some(line) = lines.next_line().await? {
                // stores in pull response struct
                let update = match serde_json::from_str(&line) {
                    Ok(PullResponse::Ok(u)) => {
                        err = None; // clear any prior error
                        Ok(u)
                    }
                    Ok(PullResponse::Err(e)) => {
                        if e.error == "pull model manifest: file does not exist" {
                            return Err(anyhow::anyhow!(e.error));
                        }
                        eprintln!("Possible transient error in ollama pull {}", e.error);
                        err = Some(anyhow::anyhow!(e.error));

                        let _ = http_client
                            .post("http://localhost:11434/api/delete") // TODO use OLLAMA_API_BASE?
                            .json(&DeleteRequest {
                                model: model.to_string(),
                                name: model.to_string(),
                            })
                            .send()
                            .await;

                        ::std::thread::sleep(::std::time::Duration::from_millis(2000));
                        break; // break out of while iteration over lines
                    }
                    Err(e) => {
                        eprintln!("Invalid response from ollama pull: {line}");
                        eprintln!("Parse error: {e}");
                        Err(anyhow::anyhow!("Ollama API request failed"))
                    }
                }?;

                let my_status = update.status.to_lowercase();

                if let Some(digest) = update.digest {
                    // handles multiple progress bars
                    let current_pb = digests.entry(digest.clone()).or_insert_with(|| {
                        let new_pb = m.add(ProgressBar::new(0));
                        new_pb.set_style(style.clone());
                        new_pb
                    });

                    current_pb.set_message(my_status.clone());

                    // sets progress bar length
                    if let (Some(total), Some(done)) = (update.total, update.completed) {
                        if current_pb.length().unwrap_or(0) == 0 {
                            current_pb.set_length(total);
                        }
                        current_pb.set_position(done);
                    }
                } else if digests.is_empty() {
                    // prints out status updates (before download)
                    m.println(&my_status).unwrap();
                } else {
                    // stores to print out status updates (after download)
                    final_status_lines.push(my_status.clone());
                }

                // checks for error or end of stream
                if my_status == "error" {
                    return Err(anyhow::anyhow!("Ollama streaming error: {}", line));
                } else if my_status == "success" {
                    break;
                }
            }

            if err.is_some() {
                continue; // continue for retry loop
            }

            // finishes drawing progress bars and outputs rest of status updates
            m.set_draw_target(indicatif::ProgressDrawTarget::hidden());
            for line in final_status_lines {
                eprintln!("{}", line);
            }

            break; // break out of retry loop
        }
    }

    match err {
        Some(err) => Err(err),
        None => Ok(()),
    }
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
        Query::Augment(crate::ir::Augment {
            embedding_model, ..
        }) => models.push(embedding_model.clone()),
        Query::Generate(Generate {
            metadata: GenerateMetadata { model, .. },
            ..
        }) => models.push(model.clone()),
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
        // this is the smallest model @starpit could find as of 20260108
        let result = ollama_pull_if_needed("smollm:135m").await;
        assert!(result.is_ok());
    }

    // testing invalid model pull
    #[tokio::test]
    async fn test_pull_invalid_model() {
        let result = ollama_pull_if_needed("notamodel").await;
        assert!(result.is_err());
    }
}

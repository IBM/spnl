use duct::cmd;
use fs4::fs_std::FileExt;

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

async fn ollama_exists(model: &str) -> anyhow::Result<bool> {
    let tags: OllamaTags = reqwest::get("http://localhost:11434/api/tags")
        .await?
        .json()
        .await?;
    Ok(tags.models.into_iter().any(|m| m.model == model))
}

/// The Ollama implementation of a single model pull
async fn ollama_pull_if_needed(model: &str) -> anyhow::Result<()> {
    // don't ? the cmd! so that we can "finally" unlock the file
    if !ollama_exists(model).await? {
        let path = ::std::env::temp_dir().join(format!("ollama-pull-{model}"));
        let f = ::std::fs::File::create(&path)?;
        /*f.lock_exclusive()?;
        if !ollama_exists(model).await?*/
        {
            let pull_res = cmd!("ollama", "pull", model)
                .stdout_to_stderr()
                .run()
                .map(|_| ());
            FileExt::unlock(&f)?;
            return Ok(pull_res?);
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

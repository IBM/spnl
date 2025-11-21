use tokio_stream::StreamExt;

use async_openai::types::CreateChatCompletionStreamResponse;
use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use crate::{
    SpnlResult,
    ir::{Bulk, GenerateMetadata, Map, Message::Assistant, Query, Repeat, to_string},
};

pub enum Spec {
    Map(Map),
    Repeat(Repeat),
}

impl Spec {
    fn n(&self) -> usize {
        match self {
            Spec::Map(m) => m.inputs.len(),
            Spec::Repeat(r) => r.n.into(),
        }
    }

    fn metadata(&self) -> GenerateMetadata {
        match self {
            Spec::Map(m) => m.metadata.clone(),
            Spec::Repeat(r) => r.generate.metadata.clone(),
        }
    }

    fn query(self) -> Query {
        match self {
            Spec::Map(m) => Query::Bulk(Bulk::Map(m)),
            Spec::Repeat(r) => Query::Bulk(Bulk::Repeat(r)),
        }
    }
}

#[derive(serde::Deserialize)]
struct Message {
    // role: String,
    content: String,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: Message,
}

#[derive(serde::Deserialize)]
struct Response {
    choices: Vec<Choice>,
}

const DATA_COLON: &[u8] = &[100, 97, 116, 97, 58, 32];

/// Call the /api/query/{prepare|execute} API, passing the given query `spec`
pub async fn generate(spec: Spec, m: Option<&MultiProgress>, prepare: bool) -> SpnlResult {
    let exec = if prepare { "prepare" } else { "execute" };
    let client = reqwest::Client::new();

    // eprintln!("Sending query {:?}", to_string(&query)?);
    let pbs = super::progress::bars(spec.n(), &spec.metadata(), &m)?;
    let mut response_strings = ::std::iter::repeat_n(String::new(), spec.n()).collect::<Vec<_>>();

    let non_streaming = if let Some(mt) = spec.metadata().max_tokens && mt == 1 { true } else { false };
    let response = client
        .post(format!("http://localhost:8000/v1/query/{exec}"))
        .query(&[("stream", if non_streaming { "false" } else {"true" })])
        .header("Content-Type", "text/plain")
        .body(to_string(&spec.query())?)
        .send()
        .await?;

    let mut stdout = stdout();
    let quiet = m.is_some();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    if non_streaming {
        response_strings = if prepare {
            vec!["prepared".to_string()]
        } else {
            response
                .json::<Response>()
                .await?
                .choices
                .into_iter()
                .map(|choice| choice.message.content)
                .collect()
        };
    } else {
    let mut stream = response.error_for_status()?.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.extend_from_slice(if chunk.starts_with(DATA_COLON) {
            &chunk[DATA_COLON.len()..]
        } else {
            &chunk
        });

        // Process complete JSON objects as they arrive
        if let Ok(res) = serde_json::from_slice::<CreateChatCompletionStreamResponse>(&buffer) {
            buffer.clear();
            for choice in res.choices.iter() {
                if let Some(ref content) = choice.delta.content {
                    let idx: usize = choice.index.try_into()?;
                    if !quiet {
                        stdout.write_all(b"\x1b[32m").await?; // green
                        stdout.write_all(content.as_bytes()).await?;
                        stdout.flush().await?;
                        stdout.write_all(b"\x1b[0m").await?; // reset color
                    } else if let Some(ref pbs) = pbs
                        && idx < pbs.len()
                    {
                        pbs[idx].inc(content.len() as u64);
                    }
                    if idx < response_strings.len() {
                        response_strings[idx] += content.as_str();
                    }
                }
            }
        }
    }
    }

    let response = response_strings
        .into_iter()
        .map(|s| Query::Message(Assistant(s)))
        .collect::<Vec<_>>();
    if response.len() == 1 {
        Ok(response.into_iter().next().unwrap())
    } else {
        Ok(Query::Par(response))
    }
}

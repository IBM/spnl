use tokio_stream::StreamExt;

use async_openai::types::CreateChatCompletionStreamResponse;
use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use crate::{
    SpnlResult,
    ir::{Bulk, Message::Assistant, Query, Repeat, to_string},
};

/// Call the /api/query/{prepare|execute} API, passing the given query `spec`
pub async fn generate(spec: Repeat, m: Option<&MultiProgress>, prepare: bool) -> SpnlResult {
    let exec = if prepare { "prepare" } else { "execute" };
    let client = reqwest::Client::new();

    // eprintln!("Sending query {:?}", to_string(&query)?);
    let pbs = super::progress::bars(spec.n.into(), &spec.generate.metadata, &m)?;
    let mut response_strings =
        ::std::iter::repeat_n(String::new(), spec.n.into()).collect::<Vec<_>>();

    let response = client
        .post(format!("http://localhost:8000/v1/query/{exec}"))
        .query(&[("stream", "true")])
        .header("Content-Type", "text/plain")
        .body(to_string(&Query::Bulk(Bulk::Repeat(spec)))?)
        .send()
        .await?;

    let mut stdout = stdout();
    let quiet = m.is_some();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    let mut stream = response.error_for_status()?.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.extend_from_slice(&chunk);

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

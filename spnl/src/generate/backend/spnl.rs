use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use crate::{
    SpnlResult,
    ir::{Bulk, Message::Assistant, Query, Repeat, to_string},
};

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

/// Call the /api/query/{prepare|execute} API, passing the given query `spec`
pub async fn generate(spec: Repeat, m: Option<&MultiProgress>, prepare: bool) -> SpnlResult {
    let exec = if prepare { "prepare" } else { "execute" };
    let client = reqwest::Client::new();

    // eprintln!("Sending query {:?}", to_string(&query)?);

    let response = client
        .post(format!("http://localhost:8000/v1/query/{exec}"))
        .header("Content-Type", "text/plain")
        .body(to_string(&Query::Bulk(Bulk::Repeat(spec)))?)
        .send()
        .await?;

    let response_strings = if prepare {
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

    let quiet = m.is_some();
    let mut stdout = stdout();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
        for response_string in response_strings.iter() {
            stdout.write_all(response_string.as_bytes()).await?;
        }
        stdout.write_all(b"\n").await?;
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

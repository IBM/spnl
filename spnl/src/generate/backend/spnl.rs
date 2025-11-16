use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use crate::{
    SpnlResult,
    ir::{Generate, Message::Assistant, Query, to_string},
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

pub async fn generate(spec: Generate, m: Option<&MultiProgress>, prepare: bool) -> SpnlResult {
    let exec = if prepare { "prepare" } else { "execute" };
    let client = reqwest::Client::new();

    // eprintln!("Sending query {:?}", to_string(&query)?);

    let response = client
        .post(format!("http://localhost:8000/v1/query/{exec}"))
        .header("Content-Type", "text/plain")
        .body(to_string(&Query::Generate(spec))?)
        .send()
        .await?;

    let response_string = if prepare {
        "prepared".to_string()
    } else {
        response.json::<Response>().await?.choices[0]
            .message
            .content
            .clone()
    };

    let quiet = m.is_some();
    let mut stdout = stdout();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
        stdout.write_all(response_string.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
    }

    Ok(Query::Message(Assistant(response_string)))
}

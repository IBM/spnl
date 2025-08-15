use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use crate::{Generate, Query, run::result::SpnlResult, to_string};

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

pub async fn generate(
    model: &str,
    input: &Query,
    max_tokens: &Option<i32>,
    temp: &Option<f32>,
    m: Option<&MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    let exec = if prepare { "prepare" } else { "execute" };
    let client = reqwest::Client::new();

    let query = Query::Generate(Generate {
        model: model.to_string(),
        input: Box::new(input.clone()),
        max_tokens: *max_tokens,
        temperature: *temp,
        accumulate: None,
    });
    // eprintln!("Sending query {:?}", to_string(&query)?);

    let response = client
        .post(format!("http://localhost:8000/v1/query/{exec}"))
        .header("Content-Type", "text/plain")
        .body(to_string(&query)?)
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

    if quiet {
        Ok(Query::User(response_string))
    } else {
        Ok(Query::Generate(Generate {
            model: format!("spnl/{model}"),
            input: Box::new(Query::User(response_string)),
            max_tokens: *max_tokens,
            temperature: *temp,
            accumulate: None,
        }))
    }
}

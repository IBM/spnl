use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent,
};

use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar};
use tokio::io::{AsyncWriteExt, stdout};

use async_openai::{Client, config::OpenAIConfig, types::CreateChatCompletionRequestArgs};

use crate::{Generate, Query, run::result::SpnlResult};

pub async fn generate(
    model: &str,
    input: &Query,
    max_tokens: &Option<i32>,
    temp: &Option<f32>,
    m: Option<&MultiProgress>,
) -> SpnlResult {
    let client = Client::with_config(OpenAIConfig::new().with_api_base("http://localhost:8000/v1"));

    let input_messages = messagify(input);

    let quiet = m.is_some();
    let mut stdout = stdout();
    /* if !quiet {
        if let Some(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(content),
            ..
        })) = input_messages.last()
        {
            stdout.write_all(b"\x1b[1mUser: \x1b[0m").await?;
            stdout.write_all(content.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
        }
    } */

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(input_messages)
        .temperature(temp.unwrap_or_default())
        .max_completion_tokens(if let Some(max_tokens) = max_tokens {
            *max_tokens as u32
        } else {
            10000
        })
        .build()?;

    let mut pb = m.map(|m| {
        m.add(if let Some(max_tokens) = max_tokens {
            ProgressBar::new(*max_tokens as u64)
        } else {
            ProgressBar::no_length()
        })
    });

    // println!("A {:?}", client.models().list().await?);

    let mut response_string = String::new();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    let mut stream = client.chat().create_stream(request).await?;
    while let Some(Ok(res)) = stream.next().await {
        for chat_choice in res.choices.iter() {
            if let Some(ref content) = chat_choice.delta.content {
                if !quiet {
                    stdout.write_all(b"\x1b[32m").await?; // green
                    stdout.write_all(content.as_bytes()).await?;
                    stdout.flush().await?;
                    stdout.write_all(b"\x1b[0m").await?; // reset color
                } else if let Some(pb) = pb.as_mut() {
                    pb.inc(content.len() as u64)
                }
                response_string += content.as_str();
            }
        }
    }
    if !quiet {
        stdout.write_all(b"\n").await?;
    }

    if m.is_some() {
        Ok(Query::User(response_string))
    } else {
        Ok(Query::Generate(Generate {
            model: format!("openai/{model}"),
            input: Box::new(Query::User(response_string)),
            max_tokens: *max_tokens,
            temperature: *temp,
            accumulate: None,
        }))
    }
}

pub fn messagify(input: &Query) -> Vec<ChatCompletionRequestMessage> {
    match input {
        Query::Cross(v) => v.iter().flat_map(messagify).collect(),
        Query::Plus(v) => v.iter().flat_map(messagify).collect(),
        Query::System(s) => vec![ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                name: None,
                content: ChatCompletionRequestSystemMessageContent::Text(s.clone()),
            },
        )],
        o => {
            let s = o.to_string();
            if s.is_empty() {
                vec![]
            } else {
                vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        name: None,
                        content: ChatCompletionRequestUserMessageContent::Text(o.to_string()),
                    },
                )]
            }
        }
    }
}

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequestArgs, ReasoningEffort,
    },
    types::completions::CreateCompletionRequestArgs,
};

use futures::StreamExt;
use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use std::borrow::Cow;

use crate::{
    SpnlResult,
    generate::GenerateOptions,
    ir::{Map, Message::*, Query, Repeat},
};

#[cfg(feature = "rag")]
use crate::augment::embed::EmbedData;

pub enum Provider {
    OpenAI,
    Gemini,
    Ollama,
}

fn api_base(provider: &Provider) -> (Cow<'static, str>, ReasoningEffort) {
    match provider {
        // Note: NO TRAILING SLASHES!
        Provider::OpenAI => (
            match ::std::env::var("OPENAI_API_BASE") {
                Ok(v) => Cow::Owned(v),
                Err(_) => Cow::Borrowed("https://api.openai.com/v1"),
            },
            ReasoningEffort::Low,
        ),
        Provider::Gemini => (
            match ::std::env::var("GEMINI_API_BASE") {
                Ok(v) => Cow::Owned(v),
                Err(_) => Cow::Borrowed("https://generativelanguage.googleapis.com/v1beta/openai"),
            },
            ReasoningEffort::Low,
        ),
        Provider::Ollama => (
            match ::std::env::var("OLLAMA_API_BASE") {
                Ok(b) => Cow::Owned(format!("{b}/v1")),
                Err(_) => Cow::Borrowed("http://localhost:11434/v1"),
            },
            ReasoningEffort::None,
        ),
    }
}

pub async fn generate_completion(
    provider: Provider,
    spec: Map,
    m: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    if let Some(true) = options.prepare {
        todo!()
    }

    let n_prompts = spec.inputs.len();
    let mut stdout = stdout();

    // Extract a max tokens
    let mt = spec
        .metadata
        .max_tokens
        .map(|mt| match mt {
            0 => 2048_u32, // vllm 400's if given 0
            _ => mt as u32,
        })
        .unwrap_or(2048);

    let quiet = m.is_some() || options.time || options.silent;

    let pbs = if options.silent {
        None
    } else {
        super::progress::bars(n_prompts, &spec.metadata, &m, None)?
    };

    let request = CreateCompletionRequestArgs::default()
        .model(spec.metadata.model)
        .prompt(spec.inputs)
        .temperature(spec.metadata.temperature.unwrap_or_default())
        .max_tokens(mt)
        .build()?;

    // println!("A {:?}", client.models().list().await?);

    let mut response_strings = ::std::iter::repeat_n(String::new(), n_prompts).collect::<Vec<_>>();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    // TODO: handle with chat_choice.delta.role, rather than hard-wire
    // Asistant (at the end of this function)
    let client = Client::with_config(OpenAIConfig::new().with_api_base(api_base(&provider).0));
    let mut stream = client.completions().create_stream(request).await?;

    // Timing tracking - track per task
    let start_time = if options.time {
        Some(::std::time::Instant::now())
    } else {
        None
    };
    let mut ttfts: Vec<Option<::std::time::Duration>> = vec![None; n_prompts];
    let mut token_counts: Vec<u64> = vec![0; n_prompts];

    loop {
        match stream.next().await {
            Some(Ok(res)) => {
                for choice in res.choices.iter() {
                    let idx: usize = choice.index.try_into()?;

                    // Track TTFT (time to first token) per task
                    if idx < ttfts.len()
                        && ttfts[idx].is_none()
                        && !choice.text.is_empty()
                        && let Some(start) = start_time
                    {
                        ttfts[idx] = Some(start.elapsed());
                    }

                    // Count tokens per task (approximate by characters for now)
                    if idx < token_counts.len() {
                        token_counts[idx] += choice.text.len() as u64;
                    }

                    if !quiet {
                        stdout.write_all(b"\x1b[32m").await?; // green
                        stdout.write_all(choice.text.as_bytes()).await?;
                        stdout.flush().await?;
                        stdout.write_all(b"\x1b[0m").await?; // reset color
                    } else if let Some(ref pbs) = pbs
                        && idx < pbs.len()
                    {
                        pbs[idx].inc(choice.text.len() as u64);
                    }
                    if idx < response_strings.len() {
                        response_strings[idx] += choice.text.as_str();
                    }
                }
            }
            Some(Err(err)) => return Err(err.into()),
            None => break,
        }
    }

    // Finish progress bars before printing timing metrics
    if let Some(ref pbs) = pbs {
        for pb in pbs {
            pb.finish_and_clear();
        }
    }

    if !quiet {
        stdout.write_all(b"\n").await?;
    }

    let response = response_strings
        .into_iter()
        .map(|s| Query::Message(Assistant(s)))
        .collect::<Vec<_>>();

    // Report timing metrics (unless in silent mode)
    if let Some(start) = start_time
        && !options.silent
    {
        let total_time = start.elapsed();
        let tasks: Vec<super::timing::TaskTiming> = ttfts
            .into_iter()
            .zip(token_counts.into_iter())
            .map(|(ttft, token_count)| super::timing::TaskTiming {
                ttft,
                total_duration: total_time,
                token_count,
            })
            .collect();
        super::timing::print_timing_metrics(&tasks);
    }

    if response.len() == 1 {
        Ok(response.into_iter().next().unwrap())
    } else {
        Ok(Query::Par(response))
    }
}

pub async fn generate_chat(
    provider: Provider,
    spec: Repeat,
    m: Option<&MultiProgress>,
    options: &GenerateOptions,
) -> SpnlResult {
    if let Some(true) = options.prepare {
        todo!()
    }

    let input_messages = messagify(&spec.generate.input);

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

    // Extract a max tokens
    let mt = spec
        .generate
        .metadata
        .max_tokens
        .map(|mt| match mt {
            0 => 2048_u32, // vllm 400's if given 0
            _ => mt as u32,
        })
        .unwrap_or(2048);

    let quiet = m.is_some() || options.time || options.silent;

    let pbs = if options.silent {
        None
    } else {
        super::progress::bars(spec.n.into(), &spec.generate.metadata, &m, None)?
    };

    let (apibase, reasoning_effort) = api_base(&provider);

    let mut request_builder_0 = CreateChatCompletionRequestArgs::default();
    let request_builder_1 = request_builder_0
        .model(spec.generate.metadata.model)
        .n(spec.n)
        .messages(input_messages)
        .temperature(spec.generate.metadata.temperature.unwrap_or_default())
        .reasoning_effort(reasoning_effort)
        .max_completion_tokens(mt);

    let request_builder = match &provider {
        Provider::Ollama => {
            // yes, this is deprecated, but... for ollama https://github.com/ollama/ollama/issues/7125
            request_builder_1.max_tokens(mt)
        }
        _ => request_builder_1,
    };

    let request = request_builder.build()?;

    // println!("A {:?}", client.models().list().await?);

    let mut response_strings =
        ::std::iter::repeat_n(String::new(), spec.n.into()).collect::<Vec<_>>();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    // Timing tracking - track per task
    let start_time = if options.time {
        Some(::std::time::Instant::now())
    } else {
        None
    };
    let mut ttfts: Vec<Option<::std::time::Duration>> = vec![None; spec.n.into()];
    let mut token_counts: Vec<u64> = vec![0; spec.n.into()];

    // TODO: handle with choice.delta.role, rather than hard-wire
    // Asistant (at the end of this function)
    let client = Client::with_config(OpenAIConfig::new().with_api_base(apibase));
    let mut stream = client.chat().create_stream(request).await?;
    loop {
        match stream.next().await {
            Some(Ok(res)) => {
                for choice in res.choices.iter() {
                    if let Some(ref content) = choice.delta.content {
                        let idx: usize = choice.index.try_into()?;

                        // Track TTFT (time to first token) per task
                        if idx < ttfts.len()
                            && ttfts[idx].is_none()
                            && !content.is_empty()
                            && let Some(start) = start_time
                        {
                            ttfts[idx] = Some(start.elapsed());
                        }

                        // Count tokens per task (approximate by characters for now)
                        if idx < token_counts.len() {
                            token_counts[idx] += content.len() as u64;
                        }

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
            Some(Err(err)) => return Err(err.into()),
            None => break,
        }
    }

    // Finish progress bars before printing timing metrics
    if let Some(ref pbs) = pbs {
        for pb in pbs {
            pb.finish_and_clear();
        }
    }

    if !quiet {
        stdout.write_all(b"\n").await?;
    }

    let response = response_strings
        .into_iter()
        .map(|s| Query::Message(Assistant(s)))
        .collect::<Vec<_>>();

    // Report timing metrics (unless in silent mode)
    if let Some(start) = start_time
        && !options.silent
    {
        let total_time = start.elapsed();
        let tasks: Vec<super::timing::TaskTiming> = ttfts
            .into_iter()
            .zip(token_counts.into_iter())
            .map(|(ttft, token_count)| super::timing::TaskTiming {
                ttft,
                total_duration: total_time,
                token_count,
            })
            .collect();
        super::timing::print_timing_metrics(&tasks);
    }

    if response.len() == 1 {
        Ok(response.into_iter().next().unwrap())
    } else {
        Ok(Query::Par(response))
    }
}

pub fn messagify(input: &Query) -> Vec<ChatCompletionRequestMessage> {
    match input {
        Query::Par(v) | Query::Seq(v) | Query::Plus(v) | Query::Cross(v) => {
            v.iter().flat_map(messagify).collect()
        }
        Query::Message(System(s)) => vec![ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                name: None,
                content: ChatCompletionRequestSystemMessageContent::Text(s.clone()),
            },
        )],
        Query::Message(Assistant(s)) => vec![ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                name: None,
                refusal: None,
                audio: None,
                tool_calls: None,
                #[allow(deprecated)]
                function_call: None,
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                    s.clone(),
                )),
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
                        content: ChatCompletionRequestUserMessageContent::Text(s),
                    },
                )]
            }
        }
    }
}

#[cfg(feature = "rag")]
pub async fn embed(
    provider: Provider,
    embedding_model: &str,
    data: &EmbedData,
) -> anyhow::Result<impl Iterator<Item = Vec<f32>> + use<>> {
    use async_openai::types::embeddings::CreateEmbeddingRequestArgs;

    let client = Client::with_config(OpenAIConfig::new().with_api_base(api_base(&provider).0));

    let request = match data {
        EmbedData::String(s) => CreateEmbeddingRequestArgs::default()
            .model(embedding_model)
            .input(s.as_str())
            .build()?,
        EmbedData::Vec(v) => CreateEmbeddingRequestArgs::default()
            .model(embedding_model)
            .input(v)
            .build()?,
        EmbedData::Query(u) => CreateEmbeddingRequestArgs::default()
            .model(embedding_model)
            .input(crate::augment::embed::contentify(u))
            .build()?,
    };

    Ok(client
        .embeddings()
        .create(request)
        .await?
        .data
        .into_iter()
        .map(|e| e.embedding))
}

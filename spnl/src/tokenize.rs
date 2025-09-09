use pyo3::prelude::*;

use crate::{
    Generate,
    Message::{self, *},
    Query,
    chat_template::{self, ChatTemplate},
};

struct Tokenizer {
    tok: tokenizers::tokenizer::Tokenizer,
    tmpl: ChatTemplate,
}

impl Tokenizer {
    fn assistant(&self, m: &str) -> String {
        chat_template::apply(self.tmpl, &[Message::Assistant(m.to_owned())], false)
    }
    fn system(&self, m: &str) -> String {
        chat_template::apply(self.tmpl, &[Message::System(m.to_owned())], false)
    }
    fn user(&self, m: &str) -> String {
        chat_template::apply(self.tmpl, &[Message::User(m.to_owned())], false)
    }

    fn usertok(&self, m: &str) -> tokenizers::tokenizer::Result<Vec<u32>> {
        Ok(self
            .tok
            .encode_fast(self.user(m), false)?
            .get_ids()
            .to_vec())
    }

    fn assistanttok(&self, m: &str) -> tokenizers::tokenizer::Result<Vec<u32>> {
        Ok(self
            .tok
            .encode_fast(self.assistant(m), false)?
            .get_ids()
            .to_vec())
    }

    fn systemtok(&self, m: &str) -> tokenizers::tokenizer::Result<Vec<u32>> {
        Ok(self
            .tok
            .encode_fast(self.system(m), false)?
            .get_ids()
            .to_vec())
    }
}

#[pyclass]
pub struct TokenizerState {
    cache: moka::sync::Cache<String, ::std::sync::Arc<Tokenizer>>,
}

impl TokenizerState {
    fn get_or_create(
        &mut self,
        model: &String,
    ) -> Result<::std::sync::Arc<Tokenizer>, ::std::sync::Arc<tokenizers::tokenizer::Error>> {
        self.cache.try_get_with(model.clone(), || {
            Ok(::std::sync::Arc::new(Tokenizer {
                tmpl: chat_template::detect(model)?,
                tok: tokenizers::tokenizer::Tokenizer::from_pretrained(model, None)?,
            }))
        })
    }
}

#[pyfunction]
pub fn init(max_capacity: u64) -> TokenizerState {
    TokenizerState {
        cache: moka::sync::Cache::new(max_capacity),
    }
}

#[pyclass]
#[derive(Debug)]
pub struct TokenizedQuery {
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    max_tokens: Option<i32>,
    #[pyo3(get)]
    temperature: Option<f32>,
    messages_: Vec<u32>,
}

#[pymethods]
impl TokenizedQuery {
    #[getter]
    fn messages(&self) -> Vec<u32> {
        self.messages_.clone()
    }
}

fn pad(pad_token: u32, block_size: usize, toklist: Vec<u32>) -> Vec<u32> {
    let n_pads = block_size - (toklist.len() % block_size);
    if n_pads == block_size {
        toklist.clone()
    } else {
        toklist[0..toklist.len() - 1]
            .iter()
            .copied()
            .chain(::std::iter::repeat_n(pad_token, n_pads))
            .chain(toklist[toklist.len() - 1..].iter().copied())
            .collect()
    }
}

fn pad_seq(
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
    seq: impl Iterator<Item = u32>,
) -> Vec<u32> {
    if let Some(cross_token) = cross_token
        && let Some(plus_token) = plus_token
    {
        let mut v = vec![];
        for t in seq {
            if t == cross_token || t == plus_token {
                let n_pads = block_size - v.len() % block_size;
                if n_pads < block_size {
                    v.extend(::std::iter::repeat_n(pad_token, n_pads));
                }
            }
            v.push(t);
        }
        v
    } else {
        seq.collect()
    }
}

fn encode_nonplus_part(
    part: &str,
    tok: &Tokenizer,
    pad_token: u32,
    block_size: usize,
) -> tokenizers::tokenizer::Result<Vec<u32>> {
    let encoded = tok.tok.encode_fast(part, false)?;
    let toks = encoded.get_ids();
    Ok(pad(pad_token, block_size, toks.to_vec()))
}

fn encode_plus_part(
    toks: &[u32],
    pad_token: u32,
    plus_token: Option<u32>,
    block_size: usize,
) -> tokenizers::tokenizer::Result<Vec<u32>> {
    if let Some(plus_token) = plus_token {
        Ok(pad(pad_token, block_size, [&[plus_token], toks].concat()))
    } else {
        Ok(toks.to_vec())
    }
}

fn extract_up_to_plus(tok: &Tokenizer, q: &Query) -> Vec<String> {
    match q {
        Query::Seq(v) | Query::Cross(v) => v
            .iter()
            .flat_map(|qq| extract_up_to_plus(tok, qq))
            .collect(),
        Query::Plus(_) => vec![],
        Query::Message(Assistant(m)) => vec![tok.assistant(m)],
        Query::Message(System(m)) => vec![tok.system(m)],
        Query::Message(User(m)) => vec![tok.user(m)],
        _ => vec![],
    }
}

fn extract_parts(tok: &Tokenizer, q: &Query, in_plus: bool) -> Vec<String> {
    match (q, in_plus) {
        (Query::Seq(v), _) | (Query::Cross(v), _) => v
            .iter()
            .flat_map(|qq| extract_parts(tok, qq, in_plus))
            .collect(),
        (Query::Plus(v), _) => v
            .iter()
            .map(|qq| extract_parts(tok, qq, true).join(""))
            .collect(),
        (Query::Message(Assistant(m)), true) => vec![tok.assistant(m)],
        (Query::Message(System(m)), true) => vec![tok.system(m)],
        (Query::Message(User(m)), true) => vec![tok.user(m)],
        _ => vec![],
    }
}

fn tokenize_part(
    input: &NonGenerateInput,
    tok: &Tokenizer,
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
) -> tokenizers::tokenizer::Result<Vec<u32>> {
    match input {
        NonGenerateInput::Seq(v) => v
            .iter()
            .map(|u| tokenize_part(u, tok, pad_token, cross_token, plus_token, block_size))
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(Ok).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect::<Result<_, _>>(),

        NonGenerateInput::Cross(v) => {
            let (left, right) = v.split_at(v.len() - 1);
            left.iter()
                .map(|u| tokenize_part(u, tok, pad_token, cross_token, plus_token, block_size))
                .flat_map(|result| match result {
                    Ok(vec) => vec.into_iter().map(Ok).collect(),
                    Err(er) => vec![Err(er)],
                })
                .chain(cross_token.map(Ok))
                .chain(
                    right
                        .iter()
                        .map(|u| {
                            tokenize_part(u, tok, pad_token, cross_token, plus_token, block_size)
                        })
                        .flat_map(|result| match result {
                            Ok(vec) => vec.into_iter().map(Ok).collect(),
                            Err(er) => vec![Err(er)],
                        }),
                )
                .collect::<Result<_, _>>()
        }

        NonGenerateInput::Plus(v) => v
            .iter()
            .map(|part| {
                encode_plus_part(
                    tokenize_part(part, tok, pad_token, cross_token, plus_token, block_size)?
                        .as_slice(),
                    pad_token,
                    plus_token,
                    block_size,
                )
            })
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(Ok).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect::<Result<_, _>>(),

        NonGenerateInput::Message(Assistant(m)) => tok.assistanttok(m),
        NonGenerateInput::Message(System(m)) => tok.systemtok(m),
        NonGenerateInput::Message(User(m)) => tok.usertok(m),
    }
}

fn handle_arc_err(e: ::std::sync::Arc<tokenizers::tokenizer::Error>) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("Error in tokenization {e}"))
}

fn handle_err(e: tokenizers::tokenizer::Error) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("Error in tokenization {e}"))
}

pub fn handle_serde_err(e: serde_json::Error) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("Error in deserialization {e}"))
}

//#[pyclass]
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NonGenerateInput {
    /// Reduce
    Plus(Vec<NonGenerateInput>),

    /// Map
    Cross(Vec<NonGenerateInput>),

    /// Execute serially
    Seq(Vec<NonGenerateInput>),

    /// Some sort of message
    #[serde(untagged)]
    Message(Message),
}

//#[pyclass]
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SingleGenerate {
    pub model: String,
    pub input: NonGenerateInput,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
}

//#[pyclass]
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SingleGenerateQuery {
    pub g: SingleGenerate,
}

impl From<NonGenerateInput> for Query {
    fn from(input: NonGenerateInput) -> Self {
        match input {
            NonGenerateInput::Message(m) => Query::Message(m),
            NonGenerateInput::Plus(v) => Query::Plus(v.into_iter().map(|m| m.into()).collect()),
            NonGenerateInput::Cross(v) => Query::Cross(v.into_iter().map(|m| m.into()).collect()),
            NonGenerateInput::Seq(v) => Query::Seq(v.into_iter().map(|m| m.into()).collect()),
        }
    }
}

impl From<SingleGenerateQuery> for Query {
    fn from(q: SingleGenerateQuery) -> Self {
        Self::Generate(Generate {
            model: q.g.model.clone(),
            input: Box::new(q.g.input.clone().into()),
            max_tokens: q.g.max_tokens,
            temperature: q.g.temperature,
        })
    }
}

#[pyfunction]
pub fn tokenize_query(
    state: &mut TokenizerState,
    q: &str,
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
) -> Result<TokenizedQuery, PyErr> {
    let query: SingleGenerateQuery = serde_json::from_str(q).map_err(handle_serde_err)?;
    let SingleGenerate {
        model,
        input,
        max_tokens,
        temperature,
    } = query.g;

    let s = ::std::time::Instant::now();
    let tok = state.get_or_create(&model).map_err(handle_arc_err)?;
    println!(
        "Spnl tokenize_query from pretrained {model}. Loaded in {:?}",
        s.elapsed()
    );
    let tokens = pad_seq(
        pad_token,
        cross_token,
        plus_token,
        block_size,
        tokenize_part(&input, &tok, pad_token, cross_token, plus_token, block_size)
            .map_err(handle_err)?
            .into_iter()
            .chain(
                tok.tok
                    .encode_fast(chat_template::apply(tok.tmpl, &[], true), false)
                    .map_err(handle_err)?
                    .get_ids()
                    .iter()
                    .copied(),
            ),
    );

    /* if let Ok(s) = tok.tok.decode(&tokens, false) {
        eprintln!("Reverse de-tokenized message (for debugging): {s}");
    } */

    Ok(TokenizedQuery {
        model: model.clone(),
        messages_: tokens,
        max_tokens,
        temperature,
    })
}

/// Extract the relocatable spans from the given query `q`. If
/// `collect_prefix_too`, then include also every span of input that
/// precedes the first relocatable span.
#[pyfunction]
pub fn tokenize_prepare(
    state: &mut TokenizerState,
    q: &str,
    collect_prefix_too: bool,
    pad_token: u32,
    plus_token: Option<u32>,
    block_size: usize,
) -> Result<Vec<Vec<u32>>, PyErr> {
    let squery: SingleGenerateQuery = serde_json::from_str(q).map_err(handle_serde_err)?;
    let query: Query = squery.into();
    match query {
        Query::Generate(Generate { model, input, .. }) => {
            let s = ::std::time::Instant::now();
            let tok = state.get_or_create(&model).map_err(handle_arc_err)?;
            println!(
                "Spnl tokenize_plus from pretrained {model}. Loaded in {:?}",
                s.elapsed()
            );

            let parts = extract_parts(&tok, &input, false).into_iter().map(|part| {
                encode_plus_part(
                    tok.tok.encode_fast(part, false)?.get_ids(),
                    pad_token,
                    plus_token,
                    block_size,
                )
            });

            if collect_prefix_too {
                parts
                    .chain(
                        extract_up_to_plus(&tok, &input)
                            .into_iter()
                            .map(|part| encode_nonplus_part(&part, &tok, pad_token, block_size)),
                    )
                    .collect::<Result<_, _>>()
                    .map_err(handle_err)
            } else {
                parts.collect::<Result<_, _>>().map_err(handle_err)
            }
        }
        _ => todo!(),
    }
}

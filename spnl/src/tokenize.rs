use pyo3::prelude::*;
use tokenizers::tokenizer::Tokenizer;

use crate::{Generate, Query};

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
            Ok(::std::sync::Arc::new(Tokenizer::from_pretrained(
                model, None,
            )?))
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
    toklist[0..toklist.len() - 1]
        .iter()
        .copied()
        .chain(::std::iter::repeat_n(
            pad_token,
            block_size - (toklist.len() % block_size),
        ))
        .chain(toklist[toklist.len() - 1..].iter().copied())
        .collect()
}

fn user(m: &String) -> String {
    format!("\n<|user|>\n{m}")
}
fn usertok(m: &String, tok: &Tokenizer) -> tokenizers::tokenizer::Result<Vec<u32>> {
    Ok(tok.encode_fast(user(m), false)?.get_ids().to_vec())
}

fn system(m: &String) -> String {
    format!("\n<|system|>\n{m}")
}
fn systemtok(m: &String, tok: &Tokenizer) -> tokenizers::tokenizer::Result<Vec<u32>> {
    Ok(tok.encode_fast(system(m), false)?.get_ids().to_vec())
}

fn encode_nonplus_part(
    part: &str,
    tok: &Tokenizer,
    pad_token: u32,
    block_size: usize,
) -> tokenizers::tokenizer::Result<Vec<u32>> {
    let encoded = tok.encode_fast(part, false)?;
    let toks = encoded.get_ids();
    Ok(pad(pad_token, block_size, toks.to_vec()))
}

fn encode_plus_part(
    part: &str,
    tok: &Tokenizer,
    pad_token: u32,
    plus_token: Option<u32>,
    block_size: usize,
) -> tokenizers::tokenizer::Result<Vec<u32>> {
    let encoded = tok.encode_fast(part, false)?;
    let toks = encoded.get_ids();
    if let Some(plus_token) = plus_token {
        Ok(pad(pad_token, block_size, [&[plus_token], toks].concat()))
    } else {
        Ok(toks.to_vec())
    }
}

fn extract_up_to_plus(q: &Query) -> Box<dyn Iterator<Item = String> + '_> {
    match q {
        Query::Cross(v) => Box::new(v.iter().flat_map(extract_up_to_plus)),
        Query::Plus(_) => Box::new(std::iter::empty()),
        Query::User(m) => Box::new(std::iter::once(user(m))),
        Query::System(m) => Box::new(std::iter::once(system(m))),
        _ => Box::new(std::iter::empty()),
    }
}

fn extract_parts(q: &Query, in_plus: bool) -> Box<dyn Iterator<Item = String> + '_> {
    match (q, in_plus) {
        (Query::Cross(v), _) => Box::new(v.iter().flat_map(move |qq| extract_parts(qq, in_plus))),
        (Query::Plus(v), _) => Box::new(
            v.iter()
                .map(|qq| extract_parts(qq, true).collect::<Vec<_>>().join("")),
        ),
        (Query::User(m), true) => Box::new(std::iter::once(user(m))),
        (Query::System(m), true) => Box::new(std::iter::once(system(m))),
        _ => Box::new(std::iter::empty()),
    }
}

fn tokenize_part(
    input: &Query,
    tok: &Tokenizer,
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
) -> tokenizers::tokenizer::Result<Vec<u32>> {
    match input {
        Query::Cross(v) => v
            .iter()
            .map(|u| tokenize_part(u, tok, pad_token, cross_token, plus_token, block_size))
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(Ok).collect(),
                Err(er) => vec![Err(er)],
            })
            .collect::<Result<_, _>>(),
        Query::Plus(_) => {
            let parts = extract_parts(input, false)
                .map(|part| encode_plus_part(&part, tok, pad_token, plus_token, block_size))
                .flat_map(|result| match result {
                    Ok(vec) => vec.into_iter().map(Ok).collect(),
                    Err(er) => vec![Err(er)],
                });
            if let Some(cross_token) = cross_token {
                parts
                    .chain([Ok(cross_token)]) // add cross token at the very end of the plus vector
                    .collect::<Result<_, _>>()
            } else {
                parts.collect::<Result<_, _>>()
            }
        }

        // TODO... we may over-pad here. We could collapse consecutive
        // System/User messages into one padded section
        Query::User(m) => Ok(pad(pad_token, block_size, usertok(m, tok)?)),
        Query::System(m) => Ok(pad(pad_token, block_size, systemtok(m, tok)?)),
        _ => {
            eprintln!("Warning: Unhandled span query component {input}");
            Ok(vec![])
        }
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
    User(String),
    System(String),
    Plus(Vec<NonGenerateInput>),
    Cross(Vec<NonGenerateInput>),
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
pub struct SimpleQuery {
    pub g: SingleGenerate,
}

impl From<NonGenerateInput> for Query {
    fn from(input: NonGenerateInput) -> Self {
        match input {
            NonGenerateInput::User(m) => Query::User(m.clone()),
            NonGenerateInput::System(m) => Query::System(m.clone()),
            NonGenerateInput::Plus(v) => Query::Plus(v.into_iter().map(|m| m.into()).collect()),
            NonGenerateInput::Cross(v) => Query::Cross(v.into_iter().map(|m| m.into()).collect()),
        }
    }
}

impl From<SimpleQuery> for Query {
    fn from(q: SimpleQuery) -> Self {
        Self::Generate(Generate {
            model: q.g.model.clone(),
            input: Box::new(q.g.input.clone().into()),
            max_tokens: q.g.max_tokens,
            temperature: q.g.temperature,
            accumulate: None,
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
    let squery: SimpleQuery = serde_json::from_str(q).map_err(handle_serde_err)?;
    let query: Query = squery.into();
    Ok(match query {
        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
            ..
        }) => {
            let s = ::std::time::Instant::now();
            let tok = state.get_or_create(&model).map_err(handle_arc_err)?;
            println!(
                "Spnl tokenize_query from pretrained {model}. Loaded in {:?}",
                s.elapsed()
            );
            let messages =
                tokenize_part(&input, &tok, pad_token, cross_token, plus_token, block_size)
                    .map_err(handle_err)?
                    .into_iter()
                    .chain(
                        tok.encode_fast("\n<|assistant|>\n", false)
                            .map_err(handle_err)?
                            .get_ids()
                            .iter()
                            .copied(),
                    )
                    .collect();

            TokenizedQuery {
                model: model.clone(),
                messages_: messages,
                max_tokens,
                temperature,
            }
        }
        _ => todo!(),
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
    let squery: SimpleQuery = serde_json::from_str(q).map_err(handle_serde_err)?;
    let query: Query = squery.into();
    match query {
        Query::Generate(Generate { model, input, .. }) => {
            let s = ::std::time::Instant::now();
            let tok = state.get_or_create(&model).map_err(handle_arc_err)?;
            println!(
                "Spnl tokenize_plus from pretrained {model}. Loaded in {:?}",
                s.elapsed()
            );

            let parts = extract_parts(&input, false)
                .map(|part| encode_plus_part(&part, &tok, pad_token, plus_token, block_size));

            if collect_prefix_too {
                parts
                    .chain(
                        extract_up_to_plus(&input)
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

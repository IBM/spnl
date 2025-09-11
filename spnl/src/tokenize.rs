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
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
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

    fn usertok(&self, m: &str, tokens: &mut Vec<u32>) -> tokenizers::tokenizer::Result<()> {
        self.extend(self.tok.encode_fast(self.user(m), false)?.get_ids(), tokens);
        Ok(())
    }

    fn assistanttok(&self, m: &str, tokens: &mut Vec<u32>) -> tokenizers::tokenizer::Result<()> {
        self.extend_crop(
            self.tok.encode_fast(self.assistant(m), false)?.get_ids(),
            tokens,
        );
        Ok(())
    }

    fn systemtok(&self, m: &str, tokens: &mut Vec<u32>) -> tokenizers::tokenizer::Result<()> {
        self.extend(
            self.tok.encode_fast(self.system(m), false)?.get_ids(),
            tokens,
        );
        Ok(())
    }

    /// Push plus token
    fn plus(&self, tokens: &mut Vec<u32>) {
        if let Some(plus_token) = self.plus_token {
            self.pad_push(plus_token, tokens);
        }
    }

    /// Push cross token
    fn cross(&self, tokens: &mut Vec<u32>) {
        if let Some(cross_token) = self.cross_token {
            self.pad_push(cross_token, tokens);
        }
    }

    /// Extend with tokens
    fn extend(&self, extra: &[u32], tokens: &mut Vec<u32>) {
        tokens.extend(extra);
    }

    /// Extend with tokens, cropping to a block boundary
    fn extend_crop(&self, extra: &[u32], tokens: &mut Vec<u32>) {
        // Round down to nearest block boundary. Note: for future
        // reference, if we need to round up to nearest block
        // boundary, replace `tokens.len()` with
        // `tokens.len()+self.block_size-1`.
        let end = extra.len() + tokens.len();
        let nearest_block_boundary = end / self.block_size * self.block_size;
        let amount_to_crop = end - nearest_block_boundary;
        let extra_end = extra.len() - amount_to_crop;

        self.extend(&extra[0..extra_end], tokens);
    }

    /// Pad to block boundary, then push
    fn pad_push(&self, token: u32, tokens: &mut Vec<u32>) {
        let n_pads = self.block_size - tokens.len() % self.block_size;
        if n_pads < self.block_size {
            tokens.extend(::std::iter::repeat_n(self.pad_token, n_pads));
        }
        tokens.push(token);
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
        pad_token: u32,
        cross_token: Option<u32>,
        plus_token: Option<u32>,
        block_size: usize,
    ) -> Result<::std::sync::Arc<Tokenizer>, ::std::sync::Arc<tokenizers::tokenizer::Error>> {
        self.cache.try_get_with(model.clone(), || {
            Ok(::std::sync::Arc::new(Tokenizer {
                tmpl: chat_template::detect(model)?,
                tok: tokenizers::tokenizer::Tokenizer::from_pretrained(model, None)?,
                pad_token,
                cross_token,
                plus_token,
                block_size,
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
        Query::Par(v) | Query::Seq(v) | Query::Cross(v) => v
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
        (Query::Par(v), _) | (Query::Seq(v), _) | (Query::Cross(v), _) => v
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
    tokens: &mut Vec<u32>,
) -> tokenizers::tokenizer::Result<()> {
    match input {
        NonGenerateInput::Seq(v) | NonGenerateInput::Par(v) => {
            v.iter().try_for_each(|u| tokenize_part(u, tok, tokens))
        }

        NonGenerateInput::Cross(v) => {
            // add cross token prior to last entry
            let (left, right) = v.split_at(v.len() - 1);

            left.iter()
                .try_for_each(|u| tokenize_part(u, tok, tokens))?;

            if !right.is_empty() {
                tok.cross(tokens);
                right.iter().try_for_each(|u| tokenize_part(u, tok, tokens))
            } else {
                Ok(())
            }
        }

        NonGenerateInput::Plus(v) => v.iter().try_for_each(|part| {
            tok.plus(tokens);
            tokenize_part(part, tok, tokens)
        }),

        NonGenerateInput::Message(Assistant(m)) => tok.assistanttok(m, tokens),
        NonGenerateInput::Message(System(m)) => tok.systemtok(m, tokens),
        NonGenerateInput::Message(User(m)) => tok.usertok(m, tokens),
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

    /// Execute in parallel
    Par(Vec<NonGenerateInput>),

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
            NonGenerateInput::Par(v) => Query::Par(v.into_iter().map(|m| m.into()).collect()),
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

/// Are we mid-stream in a Plus? This means there is a plus token with no following cross token
fn in_plus(tok: &Tokenizer, tokens: &[u32]) -> bool {
    if let Some(plus_token) = tok.plus_token
        && let Some(cross_token) = tok.cross_token
    {
        let iter = tokens.iter().rev();
        for token in iter {
            if *token == cross_token {
                return false;
            } else if *token == plus_token {
                return true;
            }
        }
    }

    false
}

/// Add the final assistant token, i.e. the token that "prompts" the
/// model to start talking.
fn add_final_assistant_token(
    tok: &Tokenizer,
    tokens: &mut Vec<u32>,
) -> tokenizers::tokenizer::Result<()> {
    if in_plus(tok, tokens) {
        // add a plus token before the final assistant token
        // if we are in a Plus
        tok.plus(tokens);
    }

    // Note: in `encode_fast(.., true)`, the `true` means add the
    // assistant token to the given (empty) list of tokens. This has
    // the net effect of providing just the model's assistant token.
    tokens.extend(
        tok.tok
            .encode_fast(chat_template::apply(tok.tmpl, &[], true), false)?
            .get_ids()
            .iter()
            .copied(),
    );

    Ok(())
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
    let tok = state
        .get_or_create(&model, pad_token, cross_token, plus_token, block_size)
        .map_err(handle_arc_err)?;
    println!(
        "Spnl tokenize_query from pretrained {model}. Loaded in {:?}",
        s.elapsed()
    );

    let mut tokens: Vec<u32> = vec![];
    tokenize_part(&input, &tok, &mut tokens)
        .and_then(|()| add_final_assistant_token(&tok, &mut tokens))
        .map_err(handle_err)?;

    // TODO: add a verbose parameter, and print this out if so?
    /* if let Ok(s) = tok.tok.decode(&tokens, false) {
        eprintln!("Tokens {tokens:?}");
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
            let tok = state
                .get_or_create(&model, pad_token, None, plus_token, block_size)
                .map_err(handle_arc_err)?;
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

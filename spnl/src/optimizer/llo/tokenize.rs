use pyo3::prelude::*;

use crate::{
    ir::{
        GenerateMetadata,
        Message::{self, *},
    },
    optimizer::llo::chat_template::{self, ChatTemplate},
};

use super::llir::*;

struct Tokenizer {
    tok: tokenizers::tokenizer::Tokenizer,
    tmpl: ChatTemplate,
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
    assistant_suffix_num_tokens: usize,
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
        let encoding = self.tok.encode_fast(self.assistant(m), false)?;
        let extra = encoding.get_ids();

        self.extend_crop(
            // TODO: for now, we always drop any assistant suffix. we
            // will need to figure out how to isolate these on their
            // own block
            &extra[0..extra.len() - self.assistant_suffix_num_tokens],
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

    /// Push plus token on block boundary
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
        let amount_to_crop = ::std::cmp::min(extra.len(), end - nearest_block_boundary);
        let extra_end = extra.len() - amount_to_crop;

        self.extend(&extra[0..extra_end], tokens);
    }

    /// Pad to block boundary, then push
    fn pad_push(&self, token: u32, tokens: &mut Vec<u32>) {
        self.pad(tokens);
        tokens.push(token);
    }

    /// Pad to block boundary
    fn pad(&self, tokens: &mut Vec<u32>) {
        let n_pads = self.block_size - tokens.len() % self.block_size;
        if n_pads > 0 && n_pads < self.block_size {
            tokens.extend(::std::iter::repeat_n(self.pad_token, n_pads));
        }
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
            let tok = tokenizers::tokenizer::Tokenizer::from_pretrained(model, None)?;
            let tmpl = chat_template::detect(model)?;

            let m = "hello";
            let binding = tok.encode_fast(
                chat_template::apply(tmpl, &[Message::Assistant(m.to_owned())], false),
                false,
            )?;
            let binding2 = tok.encode_fast(m, false)?;
            let with_chat_template = binding.get_ids();
            let without_chat_template = binding2.get_ids();

            // TODO this is imperfect...
            let start_of_message_idx = with_chat_template
                .iter()
                .position(|t| *t == without_chat_template[0]);
            let end_of_message_idx = start_of_message_idx
                .map(|start_of_message_idx| start_of_message_idx + without_chat_template.len());
            // [pppppmmmmmmmmmss]  <- ppppp are the prefix speical tokens added by chat template; ss suffix special tokens
            //       ^ start_of_message_idx
            //                ^ end_of_message_idx
            let assistant_suffix_num_tokens = if let Some(end_of_message_idx) = end_of_message_idx {
                with_chat_template.len() - end_of_message_idx
            } else {
                eprintln!(
                    "Warning: could not determine length of end of assistant special token sequence"
                );
                0
            };

            Ok(::std::sync::Arc::new(Tokenizer {
                tmpl,
                tok,
                pad_token,
                cross_token,
                plus_token,
                block_size,
                assistant_suffix_num_tokens,
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

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct TokenizedChatCompletionQuery {
    #[pyo3(get)]
    n: u8,
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    max_tokens: Option<i32>,
    #[pyo3(get)]
    temperature: Option<f32>,
    messages_: Vec<u32>,
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct CompletionRequest {
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    max_tokens: Option<i32>,
    #[pyo3(get)]
    temperature: Option<f32>,
    #[pyo3(get, set)]
    stream: Option<bool>,
    #[pyo3(get)]
    inputs: Vec<Vec<u32>>,
}

#[pyclass]
#[derive(Debug)]
pub enum TokenizedQuery {
    CompletionRequest(CompletionRequest),
    TokenizedChatCompletionQuery(TokenizedChatCompletionQuery),
}

#[pymethods]
impl TokenizedChatCompletionQuery {
    #[getter]
    fn messages(&self) -> Vec<u32> {
        self.messages_.clone()
    }
}

fn pad(pad_token: u32, block_size: usize, mut toklist: Vec<u32>) -> Vec<u32> {
    let n_pads = block_size - (toklist.len() % block_size);
    if n_pads > 0 && n_pads < block_size {
        let last = toklist.pop().unwrap();
        toklist.extend(::std::iter::repeat_n(pad_token, n_pads));
        toklist.push(last);
    }
    toklist
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

fn extract_up_to_plus(tok: &Tokenizer, q: &NonGenerateInput) -> Vec<String> {
    match q {
        NonGenerateInput::Par(v) | NonGenerateInput::Seq(v) | NonGenerateInput::Cross(v) => v
            .iter()
            .flat_map(|qq| extract_up_to_plus(tok, qq))
            .collect(),
        NonGenerateInput::Plus(_) => vec![],
        NonGenerateInput::Message(Assistant(m)) => vec![tok.assistant(m)],
        NonGenerateInput::Message(System(m)) => vec![tok.system(m)],
        NonGenerateInput::Message(User(m)) => vec![tok.user(m)],
    }
}

fn extract_parts(tok: &Tokenizer, q: &NonGenerateInput, in_plus: bool) -> Vec<String> {
    match (q, in_plus) {
        (NonGenerateInput::Par(v), _)
        | (NonGenerateInput::Seq(v), _)
        | (NonGenerateInput::Cross(v), _) => v
            .iter()
            .flat_map(|qq| extract_parts(tok, qq, in_plus))
            .collect(),
        (NonGenerateInput::Plus(v), _) => v
            .iter()
            .map(|qq| extract_parts(tok, qq, true).join(""))
            .collect(),
        (NonGenerateInput::Message(Assistant(m)), true) => vec![tok.assistant(m)],
        (NonGenerateInput::Message(System(m)), true) => vec![tok.system(m)],
        (NonGenerateInput::Message(User(m)), true) => vec![tok.user(m)],
        _ => vec![],
    }
}

fn tokenize_message(
    m: &Message,
    tok: &Tokenizer,
    tokens: &mut Vec<u32>,
) -> tokenizers::tokenizer::Result<()> {
    match m {
        Assistant(m) => tok.assistanttok(m, tokens),
        System(m) => tok.systemtok(m, tokens),
        User(m) => tok.usertok(m, tokens),
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

        NonGenerateInput::Message(m) => tokenize_message(m, tok, tokens),
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

fn tokenize(
    state: &mut TokenizerState,
    input: &NonGenerateInput,
    model: &String,
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
) -> Result<Vec<u32>, PyErr> {
    let tok = state
        .get_or_create(model, pad_token, cross_token, plus_token, block_size)
        .map_err(handle_arc_err)?;
    let mut tokens: Vec<u32> = vec![];
    tokenize_part(input, &tok, &mut tokens)
        .and_then(|()| add_final_assistant_token(&tok, &mut tokens))
        .map_err(handle_err)?;

    // TODO: add a verbose parameter, and print this out if so?
    /* if let Ok(s) = tok.tok.decode(&tokens, false) {
        eprintln!("Tokens {tokens:?}");
        eprintln!("Reverse de-tokenized message (for debugging): {s}");
    } */

    Ok(tokens)
}

fn tokenize_map(
    state: &mut TokenizerState,
    inputs: &[String],
    model: &String,
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
) -> Result<Vec<Vec<u32>>, PyErr> {
    let tok = state
        .get_or_create(model, pad_token, cross_token, plus_token, block_size)
        .map_err(handle_arc_err)?;

    inputs
        .iter()
        .map(|input| {
            let mut tokens: Vec<u32> = vec![];
            tok.plus(&mut tokens);
            tokenize_message(&Message::User(input.clone()), &tok, &mut tokens)
                .map_err(handle_err)?;
            tok.pad(&mut tokens);
            Ok(tokens)
        })
        .collect::<Result<Vec<_>, _>>()
}

fn tokenize_query_generate(
    state: &mut TokenizerState,
    Repeat {
        n,
        generate: SingleGenerate { input, metadata },
    }: Repeat,
    pad_token: u32,
    cross_token: Option<u32>,
    plus_token: Option<u32>,
    block_size: usize,
) -> Result<TokenizedQuery, PyErr> {
    Ok(TokenizedQuery::TokenizedChatCompletionQuery(
        TokenizedChatCompletionQuery {
            n,
            messages_: tokenize(
                state,
                &input,
                &metadata.model,
                pad_token,
                cross_token,
                plus_token,
                block_size,
            )?,
            model: metadata.model,
            max_tokens: metadata.max_tokens,
            temperature: metadata.temperature,
        },
    ))
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
    match serde_json::from_str(q).map_err(handle_serde_err)? {
        SingleGenerateQuery::SingleGenerate(generate) => tokenize_query_generate(
            state,
            Repeat { n: 1, generate },
            pad_token,
            cross_token,
            plus_token,
            block_size,
        ),
        SingleGenerateQuery::Bulk(Bulk::Repeat(repeat)) => tokenize_query_generate(
            state,
            repeat,
            pad_token,
            cross_token,
            plus_token,
            block_size,
        ),
        SingleGenerateQuery::Bulk(Bulk::Map(map)) => {
            Ok(TokenizedQuery::CompletionRequest(CompletionRequest {
                inputs: tokenize_map(
                    state,
                    &map.inputs,
                    &map.metadata.model,
                    pad_token,
                    cross_token,
                    plus_token,
                    block_size,
                )?,
                model: map.metadata.model,
                max_tokens: map.metadata.max_tokens,
                temperature: map.metadata.temperature,
                stream: None,
            }))
        }
    }
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
    match serde_json::from_str(q).map_err(handle_serde_err)? {
        SingleGenerateQuery::SingleGenerate(SingleGenerate {
            input,
            metadata: GenerateMetadata { model, .. },
        }) => {
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

        // SingleGenerateQuery::Bulk(Bulk::Repeat(Repeat { n, generate })) => {
        //     todo!()
        // }

        // SingleGenerateQuery::Bulk(Bulk::Map(Map { metadata, inputs })) => {
        //     todo!()
        // }
        _ => {
            todo!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    const PAD_TOKEN: u32 = 27;
    const BLOCK_SIZE: usize = 16;

    const MODEL: &str = "ibm-granite/granite-3.3-2b-instruct"; // TODO find smaller model with public tokenizers.json
    const START_OF_ROLE: u32 = 49152;
    const END_OF_ROLE: u32 = 49153;
    const END_OF_TEXT: u32 = 0;
    const USER: u32 = 496;
    const ASSISTANT: u32 = 17594;
    const HELLO: u32 = 7656;
    const LONGER: u32 = 8928;

    fn tok() -> Result<::std::sync::Arc<Tokenizer>, ::std::sync::Arc<tokenizers::tokenizer::Error>>
    {
        init(2).get_or_create(&MODEL.into(), PAD_TOKEN, None, None, BLOCK_SIZE)
    }

    #[test]
    fn create_tokenizer() -> Result<(), ::std::sync::Arc<tokenizers::tokenizer::Error>> {
        tok().map(|_| ())
    }

    #[test]
    fn user() -> Result<(), ::std::sync::Arc<tokenizers::tokenizer::Error>> {
        assert_eq!(
            tok().map(|tok| tok.user("hello"))?,
            "<|start_of_role|>user<|end_of_role|>hello<|end_of_text|>"
        );
        Ok(())
    }

    #[test]
    fn usertok() -> Result<(), ::std::sync::Arc<tokenizers::tokenizer::Error>> {
        let mut tokens = vec![];
        tok()?.usertok("hello", &mut tokens)?;
        assert_eq!(
            tokens,
            [START_OF_ROLE, USER, END_OF_ROLE, HELLO, END_OF_TEXT]
        );
        Ok(())
    }

    #[test]
    fn assistant() -> Result<(), ::std::sync::Arc<tokenizers::tokenizer::Error>> {
        assert_eq!(
            tok().map(|tok| tok.assistant("hello"))?,
            "<|start_of_role|>assistant<|end_of_role|>hello<|end_of_text|>"
        );
        Ok(())
    }

    #[test]
    fn assistanttok_fully_cropped() -> Result<(), ::std::sync::Arc<tokenizers::tokenizer::Error>> {
        let mut tokens = vec![];
        tok()?.assistanttok("hello", &mut tokens)?;
        let empty: &[u32] = &[];
        assert_eq!(tokens, empty);
        Ok(())
    }

    #[test]
    fn assistanttok_partially_cropped() -> Result<(), ::std::sync::Arc<tokenizers::tokenizer::Error>>
    {
        let repeat_input = 17; // repeat this many times for the input message
        let repeat_output = 12; // expect this many repetitions after cropping
        let mut tokens = vec![];
        tok()?.assistanttok(
            format!(
                "hello {}",
                ::std::iter::repeat_n("longer", repeat_input).join(" ")
            )
            .as_str(),
            &mut tokens,
        )?;
        assert_eq!(
            tokens,
            [START_OF_ROLE, ASSISTANT, END_OF_ROLE, HELLO]
                .into_iter()
                .chain(::std::iter::repeat_n(LONGER, repeat_output))
                // .chain([END_OF_TEXT])
                .collect::<Vec<u32>>(),
        );
        Ok(())
    }
}

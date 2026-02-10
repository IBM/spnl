//! Extract tokenizer from GGUF file metadata
//!
//! GGUF files contain embedded tokenizer data in their metadata, similar to how Ollama handles them.
//! This module extracts that data and builds a tokenizers::Tokenizer instance by creating
//! a tokenizer.json structure and loading it.

use anyhow::{Result, anyhow};
use candle_core::quantized::gguf_file;
use serde_json::json;
use tokenizers::Tokenizer;

/// Extract tokenizer from GGUF file metadata
pub fn extract_tokenizer_from_gguf(content: &gguf_file::Content) -> Result<Tokenizer> {
    // 1. Read tokenizer type
    let tokenizer_model =
        get_string(content, "tokenizer.ggml.model").unwrap_or_else(|_| "llama".to_string()); // Default to llama/sentencepiece

    // eprintln!("Extracting tokenizer from GGUF (type: {})", tokenizer_model);

    // 2. Extract vocabulary
    let tokens = get_string_array(content, "tokenizer.ggml.tokens")?;
    // eprintln!("  Vocabulary size: {}", tokens.len());

    // 3. Extract special tokens (with defaults)
    let bos_token_id = get_u32(content, "tokenizer.ggml.bos_token_id").ok();
    let eos_token_id = get_u32(content, "tokenizer.ggml.eos_token_id").ok();
    let unk_token_id = get_u32(content, "tokenizer.ggml.unknown_token_id")
        .or_else(|_| get_u32(content, "tokenizer.ggml.unk_token_id"))
        .ok();

    /* eprintln!("  Special tokens: BOS={:?}, EOS={:?}, UNK={:?}",
    bos_token_id, eos_token_id, unk_token_id); */

    // 4. Build tokenizer JSON based on type
    let tokenizer_json = match tokenizer_model.as_str() {
        "gpt2" => {
            let merges = get_string_array(content, "tokenizer.ggml.merges")?;
            build_bpe_json(tokens, merges, bos_token_id, eos_token_id, unk_token_id)?
        }
        "llama" | "sentencepiece" => {
            // SentencePiece - use BPE with empty merges as approximation
            build_sentencepiece_json(tokens, bos_token_id, eos_token_id, unk_token_id)?
        }
        "bert" => build_wordpiece_json(tokens, bos_token_id, eos_token_id, unk_token_id)?,
        _ => {
            return Err(anyhow!("Unsupported tokenizer type: {}", tokenizer_model));
        }
    };

    // 5. Load tokenizer from JSON string (using from_bytes)
    let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
        .map_err(|e| anyhow!("Failed to load tokenizer from JSON: {}", e))?;

    // eprintln!("✓ Successfully extracted tokenizer from GGUF");
    Ok(tokenizer)
}

/// Build a BPE tokenizer JSON (used by GPT-2, GPT-3, etc.)
fn build_bpe_json(
    tokens: Vec<String>,
    merges: Vec<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
) -> Result<String> {
    // Create vocab map
    let vocab: serde_json::Map<String, serde_json::Value> = tokens
        .iter()
        .enumerate()
        .map(|(i, token)| (token.clone(), json!(i)))
        .collect();

    // Get special token strings
    let unk_token = unk_token_id
        .and_then(|id| tokens.get(id as usize))
        .map(|s| s.as_str())
        .unwrap_or("<|endoftext|>");

    let _bos_token = bos_token_id
        .and_then(|id| tokens.get(id as usize))
        .map(|s| s.as_str());

    let _eos_token = eos_token_id
        .and_then(|id| tokens.get(id as usize))
        .map(|s| s.as_str());

    let json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": build_added_tokens(&tokens, bos_token_id, eos_token_id, unk_token_id),
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": unk_token,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocab,
            "merges": merges
        }
    });

    Ok(serde_json::to_string(&json)?)
}

/// Build a SentencePiece tokenizer JSON (used by LLaMA, Qwen, etc.)
fn build_sentencepiece_json(
    tokens: Vec<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
) -> Result<String> {
    let json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": build_added_tokens(&tokens, bos_token_id, eos_token_id, unk_token_id),
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": true,
            "prepend_scheme": "always"
        },
        "post_processor": null,
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": true,
            "prepend_scheme": "always"
        },
        "model": {
            "type": "Unigram",
            "unk_id": unk_token_id,
            "vocab": tokens.iter().map(|token| {
                json!([token.clone(), 0.0]) // Score of 0.0 for all tokens
            }).collect::<Vec<_>>()
        }
    });

    Ok(serde_json::to_string(&json)?)
}

/// Build a WordPiece tokenizer JSON (used by BERT, etc.)
fn build_wordpiece_json(
    tokens: Vec<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
) -> Result<String> {
    // Create vocab map
    let vocab: serde_json::Map<String, serde_json::Value> = tokens
        .iter()
        .enumerate()
        .map(|(i, token)| (token.clone(), json!(i)))
        .collect();

    // Get special token strings
    let unk_token = unk_token_id
        .and_then(|id| tokens.get(id as usize))
        .map(|s| s.as_str())
        .unwrap_or("[UNK]");

    let json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": build_added_tokens(&tokens, bos_token_id, eos_token_id, unk_token_id),
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Whitespace"
        },
        "post_processor": null,
        "decoder": {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": true
        },
        "model": {
            "type": "WordPiece",
            "unk_token": unk_token,
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab
        }
    });

    Ok(serde_json::to_string(&json)?)
}

/// Build added_tokens array for special tokens
fn build_added_tokens(
    tokens: &[String],
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_token_id: Option<u32>,
) -> Vec<serde_json::Value> {
    let mut added_tokens = Vec::new();

    if let Some(id) = bos_token_id
        && let Some(token) = tokens.get(id as usize)
    {
        added_tokens.push(json!({
            "id": id,
            "content": token,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }));
    }

    if let Some(id) = eos_token_id
        && let Some(token) = tokens.get(id as usize)
    {
        added_tokens.push(json!({
            "id": id,
            "content": token,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }));
    }

    if let Some(id) = unk_token_id
        && let Some(token) = tokens.get(id as usize)
    {
        added_tokens.push(json!({
            "id": id,
            "content": token,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }));
    }

    added_tokens
}

// Helper functions to extract metadata from GGUF

fn get_string(content: &gguf_file::Content, key: &str) -> Result<String> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(anyhow!("Key '{}' is not a string", key)),
        None => Err(anyhow!("Key '{}' not found in GGUF metadata", key)),
    }
}

fn get_string_array(content: &gguf_file::Content, key: &str) -> Result<Vec<String>> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::Array(arr)) => {
            let mut result = Vec::new();
            for item in arr {
                match item {
                    gguf_file::Value::String(s) => result.push(s.clone()),
                    _ => return Err(anyhow!("Array '{}' contains non-string elements", key)),
                }
            }
            Ok(result)
        }
        Some(_) => Err(anyhow!("Key '{}' is not an array", key)),
        None => Err(anyhow!("Key '{}' not found in GGUF metadata", key)),
    }
}

fn get_u32(content: &gguf_file::Content, key: &str) -> Result<u32> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::U32(v)) => Ok(*v),
        Some(gguf_file::Value::U64(v)) => Ok(*v as u32),
        Some(gguf_file::Value::I32(v)) => Ok(*v as u32),
        Some(gguf_file::Value::I64(v)) => Ok(*v as u32),
        Some(_) => Err(anyhow!("Key '{}' is not a numeric type", key)),
        None => Err(anyhow!("Key '{}' not found in GGUF metadata", key)),
    }
}

// Made with Bob

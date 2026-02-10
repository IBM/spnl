# GGUF Tokenizer Extraction Implementation

## Overview

This implementation extracts tokenizers directly from GGUF file metadata, eliminating the need to download separate `tokenizer.json` files from HuggingFace. This approach is inspired by how Ollama handles GGUF files.

## Implementation Details

### Files Modified/Created

1. **`gguf_tokenizer.rs`** (NEW) - Core tokenizer extraction module
   - Extracts tokenizer metadata from GGUF files
   - Builds tokenizer.json structures for different tokenizer types
   - Supports BPE, SentencePiece/Unigram, and WordPiece tokenizers

2. **`loader.rs`** (MODIFIED) - Updated `load_gguf_model()` function
   - Now attempts to extract tokenizer from GGUF first
   - Falls back to HuggingFace download if extraction fails
   - Maintains backward compatibility

3. **`mod.rs`** (MODIFIED) - Added `gguf_tokenizer` module

### How It Works

1. **Read GGUF Metadata**: When loading a GGUF model, the loader opens the file and reads its metadata using `candle_core::quantized::gguf_file::Content`.

2. **Extract Tokenizer Data**: The `extract_tokenizer_from_gguf()` function extracts:
   - Tokenizer type (`tokenizer.ggml.model`)
   - Vocabulary tokens (`tokenizer.ggml.tokens`)
   - Merges for BPE (`tokenizer.ggml.merges`)
   - Special token IDs (BOS, EOS, UNK, PAD)

3. **Build Tokenizer JSON**: Based on the tokenizer type, we construct a complete `tokenizer.json` structure:
   - **BPE** (GPT-2 style): Includes vocab, merges, ByteLevel pre-tokenizer/decoder
   - **SentencePiece/Unigram** (LLaMA, Qwen): Uses Unigram model with Metaspace pre-tokenizer
   - **WordPiece** (BERT): Uses WordPiece model with Whitespace pre-tokenizer

4. **Load Tokenizer**: The JSON string is loaded using `Tokenizer::from_bytes()`.

5. **Fallback**: If extraction fails (e.g., missing metadata keys), the system falls back to the original behavior of downloading `tokenizer.json` from HuggingFace.

### Supported Tokenizer Types

| Type | GGUF Value | Used By | Status |
|------|------------|---------|--------|
| BPE | `gpt2` | GPT-2, GPT-3, etc. | ✅ Implemented |
| SentencePiece | `llama`, `sentencepiece` | LLaMA, Qwen, Mistral | ✅ Implemented |
| WordPiece | `bert` | BERT, etc. | ✅ Implemented |

### Benefits

1. **No Network Required**: Tokenizer is extracted locally from the GGUF file
2. **Faster Loading**: No need to download separate files
3. **Self-Contained**: GGUF files work standalone
4. **Backward Compatible**: Falls back to HuggingFace if needed
5. **Fixes Issues**: Resolves problems with models like `unsloth/gemma-3-27b-it-GGUF` where no separate tokenizer exists

### GGUF Metadata Keys Used

```
tokenizer.ggml.model          - Tokenizer type (string)
tokenizer.ggml.tokens         - Array of token strings
tokenizer.ggml.scores         - Array of token scores (optional)
tokenizer.ggml.token_type     - Array of token types (optional)
tokenizer.ggml.merges         - BPE merge rules (for BPE tokenizers)
tokenizer.ggml.bos_token_id   - Beginning-of-sequence token ID
tokenizer.ggml.eos_token_id   - End-of-sequence token ID
tokenizer.ggml.unk_token_id   - Unknown token ID
tokenizer.ggml.pad_token_id   - Padding token ID (optional)
tokenizer.ggml.add_bos_token  - Whether to add BOS token (optional)
tokenizer.ggml.add_eos_token  - Whether to add EOS token (optional)
```

## Testing

To test the implementation:

```bash
# Test with a GGUF model that has embedded tokenizer
cargo run --features candle --bin spnl -- run "Qwen/Qwen2.5-1.5B-Instruct-GGUF" "Hello, how are you?"

# Test with the problematic model mentioned in the plan
cargo run --features candle --bin spnl -- run "hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_XL" "Hello"
```

Expected output should include:
```
Extracting tokenizer from GGUF (type: llama)
  Vocabulary size: 151936
  Special tokens: BOS=Some(151643), EOS=Some(151645), UNK=Some(151643)
✓ Successfully extracted tokenizer from GGUF
```

## Future Improvements

1. **Add Scores Support**: Currently, we use default scores (0.0). Could extract actual scores from `tokenizer.ggml.scores` if available.

2. **Better Pre-tokenizer Detection**: The pre-tokenizer choice could be refined based on additional GGUF metadata.

3. **Post-processor Support**: Add template-based post-processors for chat models (e.g., adding BOS/EOS tokens automatically).

4. **Contribute to Candle**: This functionality could be contributed upstream to `candle-transformers` to benefit the entire ecosystem.

5. **Remove Hardcoded Mappings**: Once this is stable, we can remove the `infer_base_model_from_gguf()` function and all its hardcoded model mappings (lines 306-369 in `loader.rs`).

## References

- Ollama GGUF handling: `/Users/nickm/git/ollama/model/models/llama/model.go`
- GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Tokenizers crate: https://github.com/huggingface/tokenizers
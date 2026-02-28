/// Does the given provider support PIC (Position-Independent Caching)?
///
/// This includes:
/// - `spnl/` prefixed models (vLLM spans backend)
/// - `local/` prefixed models (mistral.rs backend)
/// - Pretty names that resolve to the local backend (e.g. `llama3.1:8b`)
pub fn supports_spans(provider_slash_model: &str) -> bool {
    if provider_slash_model.starts_with("spnl/") || provider_slash_model.starts_with("local/") {
        return true;
    }
    // No recognized prefix → falls through to prettynames → local backend
    !provider_slash_model.contains('/')
}

/// Does the given provider support the bulk-repeat API (generate with `n`)?
pub fn supports_bulk_repeat(provider_slash_model: &str) -> bool {
    // @starpit 20251117 Tried gemini2.0-flash and gemini2.5-flash and gemini2.5-pro and none of these supports `n`
    // @starpit 20251117 re: Ollama: https://github.com/ollama/ollama/issues/13111
    !provider_slash_model.starts_with("disable_bulk_repeat/")
        && !provider_slash_model.starts_with("gemini/")
        && !provider_slash_model.starts_with("ollama/")
        && !provider_slash_model.starts_with("spnl/") // the vLLM spans stuff seems to have a bug? not sure yet. can't get good cache locality with Bulk::Repeat
}

/// Does the given provider support the bulk-map API (non-chat completion across a vector of string prompts)?
pub fn supports_bulk_map(provider_slash_model: &str) -> bool {
    // @starpit 20251117 Tried gemini2.0-flash and gemini2.5-flash and gemini2.5-pro and none of these supports bulk map
    // @starpit 20251117 re: Ollama: https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx#notes
    !provider_slash_model.starts_with("disable_bulk_map/")
        && !provider_slash_model.starts_with("gemini/")
        && !provider_slash_model.starts_with("ollama/")
}

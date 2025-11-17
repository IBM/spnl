/// Does the given provider support the spnl REST API?
pub fn supports_spans(provider_slash_model: &str) -> bool {
    // for now...
    provider_slash_model.starts_with("spnl/")
}

/// Does the given provider support the bulk-repeat API (generate with `n`)?
pub fn supports_bulk_repeat(provider_slash_model: &str) -> bool {
    // @starpit 20251117 Tried gemini2.0-flash and gemini2.5-flash and gemini2.5-pro and none of these supports `n`
    // @starpit 20251117 re: Ollama: https://github.com/ollama/ollama/issues/13111
    !provider_slash_model.starts_with("disable_bulk_repeat/")
        && !provider_slash_model.starts_with("gemini/")
        && !provider_slash_model.starts_with("ollama/")
}

/// Does the given provider support the bulk-map API (non-chat completion across a vector of string prompts)?
pub fn supports_bulk_map(provider_slash_model: &str) -> bool {
    // @starpit 20251117 Tried gemini2.0-flash and gemini2.5-flash and gemini2.5-pro and none of these supports bulk map
    // @starpit 20251117 re: Ollama: https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx#notes
    !provider_slash_model.starts_with("disable_bulk_map/")
        && !provider_slash_model.starts_with("gemini/")
        && !provider_slash_model.starts_with("ollama/")
}

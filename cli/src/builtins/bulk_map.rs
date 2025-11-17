use crate::args::Args;
use spnl::ir::{Bulk, GenerateMetadata, Map, Query};

pub fn query(args: Args) -> Query {
    let Args {
        model,
        temperature,
        max_tokens,
        ..
    } = args;

    Query::Bulk(Bulk::Map(Map {
        inputs: vec![
            "What color is the sky?".to_string(),
            "What color is clay?".to_string(),
        ],
        metadata: GenerateMetadata {
            model,
            max_tokens: max_tokens.into(),
            temperature: temperature.into(),
        },
    }))
}

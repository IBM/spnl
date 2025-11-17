use super::{Generate, GenerateBuilder, GenerateMetadata, GenerateMetadataBuilder};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Bulk {
    Repeat(Repeat),

    Map(Map),
}

/// Bulk operation: generate `n` outputs for the given `generate` specification
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Repeat {
    /// The number of outputs to generate
    pub n: u8,

    /// The specification of what to generate
    pub generate: Generate,
}

impl Repeat {
    pub fn with_model(&self, model: &str) -> anyhow::Result<Self> {
        Ok(Repeat {
            n: self.n,
            generate: GenerateBuilder::from(self.generate.clone())
                .metadata(
                    GenerateMetadataBuilder::from(self.generate.metadata.clone())
                        .model(model.to_string())
                        .build()?,
                )
                .build()?,
        })
    }
}

/// Bulk operation: map the generate operation across the given inputs, using the given metadata
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Map {
    /// The metadata governing the content generation
    pub metadata: GenerateMetadata,

    /// Generate one output for each input in this list
    pub inputs: Vec<String>,
}

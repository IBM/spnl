use super::{Generate, GenerateBuilder, GenerateMetadata, GenerateMetadataBuilder};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Bulk {
    Repeat(Repeat),

    Map(Map),
}

/// Bulk operation: generate `n` outputs for the given `generate` specification
#[derive(
    Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, derive_builder::Builder,
)]
pub struct Repeat {
    /// The number of outputs to generate
    #[builder(setter(into), default = 1u8)]
    pub n: u8,

    /// The specification of what to generate
    #[serde(rename = "g")]
    pub generate: Generate,
}

impl From<&Repeat> for RepeatBuilder {
    fn from(other: &Repeat) -> Self {
        RepeatBuilder::default()
            .n(other.n)
            .generate(other.generate.clone())
            .clone()
    }
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

impl Map {
    pub fn with_model(&self, model: &str) -> anyhow::Result<Self> {
        Ok(Map {
            inputs: self.inputs.clone(),
            metadata: GenerateMetadataBuilder::from(self.metadata.clone())
                .model(model.to_string())
                .build()?,
        })
    }
}

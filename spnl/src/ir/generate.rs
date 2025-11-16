use super::Query;

#[derive(
    Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, derive_builder::Builder,
)]
#[builder(derive(serde::Serialize))]
pub struct GenerateMetadata {
    #[builder(setter(into))]
    pub model: String,

    #[serde(
        skip_serializing_if = "Option::is_none",
        default = "default_max_tokens"
    )]
    #[builder(setter(into), default = Some(0))]
    pub max_tokens: Option<i32>,

    #[serde(
        skip_serializing_if = "Option::is_none",
        default = "default_temperature"
    )]
    #[builder(setter(into), default = Some(0.6))]
    pub temperature: Option<f32>,
}

fn default_max_tokens() -> Option<i32> {
    Some(0)
}

fn default_temperature() -> Option<f32> {
    Some(0.6)
}

impl From<GenerateMetadata> for GenerateMetadataBuilder {
    fn from(other: GenerateMetadata) -> Self {
        GenerateMetadataBuilder::default()
            .model(other.model)
            .max_tokens(other.max_tokens)
            .temperature(other.temperature)
            .clone()
    }
}

impl From<&GenerateMetadata> for GenerateMetadataBuilder {
    fn from(other: &GenerateMetadata) -> Self {
        GenerateMetadataBuilder::default()
            .model(other.model.clone())
            .max_tokens(other.max_tokens)
            .temperature(other.temperature)
            .clone()
    }
}

#[derive(
    Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, derive_builder::Builder,
)]
pub struct Generate {
    #[serde(flatten)]
    pub metadata: GenerateMetadata,

    pub input: Box<Query>,
}

impl Generate {
    /// Return self, but with input wrapped according to the given function
    pub fn wrap(&self, f: fn(Query) -> Query) -> Self {
        let mut g = self.clone();
        g.input = Box::new(f(*g.input));
        g
    }

    /// Return self, but with input wrapped with a Plus
    pub fn wrap_plus(&self) -> Self {
        self.wrap(|input| Query::Plus(vec![input]))
    }
}

impl From<&Generate> for GenerateBuilder {
    fn from(other: &Generate) -> Self {
        GenerateBuilder::default()
            .metadata(other.metadata.clone())
            .input(other.input.clone())
            .clone()
    }
}

impl Generate {
    pub fn with_model(&self, model: &str) -> anyhow::Result<Self> {
        Ok(GenerateBuilder::from(self)
            .metadata(
                GenerateMetadataBuilder::from(self.metadata.clone())
                    .model(model.to_string())
                    .build()?,
            )
            .build()?)
    }
}

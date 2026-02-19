use super::Query;

#[derive(
    Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, derive_builder::Builder,
)]
#[builder(derive(serde::Serialize))]
pub struct GenerateMetadata {
    /// This is of the form `{provider}/{model}`
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
        let mut b = GenerateMetadataBuilder::default();
        b.model(other.model);
        b.max_tokens(other.max_tokens);
        b.temperature(other.temperature);
        b
    }
}

impl From<&GenerateMetadata> for GenerateMetadataBuilder {
    fn from(other: &GenerateMetadata) -> Self {
        let mut b = GenerateMetadataBuilder::default();
        b.model(other.model.clone());
        b.max_tokens(other.max_tokens);
        b.temperature(other.temperature);
        b
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
        self.wrap(|input| {
            Query::Plus(match input {
                Query::Seq(s) => s,
                _ => vec![input],
            })
        })
    }
}

impl From<Generate> for GenerateBuilder {
    fn from(other: Generate) -> Self {
        let mut b = GenerateBuilder::default();
        b.metadata(other.metadata);
        b.input(other.input);
        b
    }
}

impl From<&Generate> for GenerateBuilder {
    fn from(other: &Generate) -> Self {
        let mut b = GenerateBuilder::default();
        b.metadata(other.metadata.clone());
        b.input(other.input.clone());
        b
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

use crate::ir::Query;

#[derive(
    Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, derive_builder::Builder,
)]
pub struct Generate {
    #[builder(setter(into))]
    pub model: String,

    pub input: Box<Query>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default = Some(0))]
    pub max_tokens: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default = Some(0.6))]
    pub temperature: Option<f32>,
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
            .model(other.model.clone())
            .input(other.input.clone())
            .max_tokens(other.max_tokens)
            .temperature(other.temperature)
            .clone()
    }
}

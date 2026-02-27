#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Document {
    Text(String),
    Binary(Vec<u8>),
}

#[cfg(feature = "rag")]
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Augment {
    pub embedding_model: String,
    pub body: Box<crate::ir::Query>,
    pub doc: (String, Document),
}

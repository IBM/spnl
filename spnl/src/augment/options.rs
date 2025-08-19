#[derive(clap::ValueEnum, Clone, Debug, Default, serde::Serialize)]
pub enum Indexer {
    /// Only perform the initial embedding without any further
    /// knowledge graph formation
    SimpleEmbedRetrieve,

    /// Use the RAPTOR algorithm https://github.com/parthsarthi03/raptor
    #[default]
    Raptor,
}

#[derive(Clone, Default, derive_builder::Builder)]
pub struct AugmentOptions {
    /// Max augmentations to add to the query
    pub max_aug: Option<usize>,

    /// URI of vector database, which can be a local filepath
    pub vecdb_uri: String,

    /// Name of table to use in vector database
    pub vecdb_table: String,

    /// Scheme to use for indexing the corpus
    #[builder(default)]
    pub indexer: Indexer,

    /// Scheme to use for indexing the corpus
    #[builder(default)]
    pub verbose: bool,
}

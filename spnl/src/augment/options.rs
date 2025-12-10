#[derive(clap::ValueEnum, Clone, Debug, Default, serde::Serialize)]
pub enum Indexer {
    /// Only perform the initial embedding without any further
    /// knowledge graph formation
    #[default]
    SimpleEmbedRetrieve,

    /// Use the RAPTOR algorithm https://github.com/parthsarthi03/raptor
    Raptor,
}

#[derive(Clone, Debug, derive_builder::Builder)]
pub struct AugmentOptions {
    /// Max augmentations to add to the query
    #[builder(default)]
    pub max_aug: Option<usize>,

    /// URI of vector database, which can be a local filepath
    #[builder(default = "data/spnl".to_string())]
    pub vecdb_uri: String,

    /// Name of table to use in vector database
    #[builder(default = "default".to_string())]
    pub vecdb_table: String,

    /// Scheme to use for indexing the corpus
    #[builder(default)]
    pub indexer: Indexer,

    /// Randomly shuffle order of fragments
    #[builder(default)]
    pub shuffle: bool,

    /// Scheme to use for indexing the corpus
    #[builder(default)]
    pub verbose: bool,
}

impl Default for AugmentOptions {
    fn default() -> Self {
        AugmentOptionsBuilder::default().build().unwrap()
    }
}

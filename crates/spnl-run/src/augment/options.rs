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

    /// Directory where HNSW indexes are stored
    #[builder(default = "\"data/spnl\".to_string()")]
    pub index_dir: String,

    /// Chunk size for sentence-based chunking (characters)
    #[builder(default = "512")]
    pub chunk_size: usize,

    /// Chunk overlap for sentence-based chunking (characters)
    #[builder(default = "50")]
    pub chunk_overlap: usize,

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

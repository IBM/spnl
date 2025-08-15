#[derive(Default)]
pub struct AugmentOptions {
    /// Max augmentations to add to the query
    pub max_aug: Option<usize>,

    /// URI of vector database. Could be a local filepath.
    pub vecdb_uri: String,

    /// Name of table to use in vector database.
    pub vecdb_table: String,
}

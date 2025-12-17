use super::Query;

/// Interleave the messages that emanate from the first
/// in-between every message that emanates from the second
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Zip {
    pub first: Box<Query>,
    pub second: Box<Query>,
}

/// Turn a pair into a Zip
impl From<(Query, Query)> for Query {
    fn from(pair: (Query, Query)) -> Self {
        Self::Zip(Zip {
            first: pair.0.into(),
            second: pair.1.into(),
        })
    }
}

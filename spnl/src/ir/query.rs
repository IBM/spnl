use super::{Bulk, Generate, Message};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Query {
    /// Execute in sequence
    Seq(Vec<Query>),

    /// Execute in parallel
    Par(Vec<Query>),

    /// Non-commutative
    Cross(Vec<Query>),

    /// Commutative
    Plus(Vec<Query>),

    /// Ignore the output, executed for server-side effect only (e.g. caching)
    Monad(Box<Query>),

    /// Generate new content via a given model
    #[serde(rename = "g")]
    Generate(Generate),

    /// Incorporate information relevant to the question gathered from
    /// the given docs
    #[cfg(feature = "rag")]
    Augment(crate::ir::Augment),

    /// Ask with a given message
    #[cfg(feature = "cli_support")]
    Ask(String),

    /// Print a helpful message to the console
    #[cfg(feature = "print")]
    Print(String),

    /// Some kind of bulk operation
    #[serde(untagged)]
    Bulk(Bulk),

    /// Some sort of chat message
    #[serde(untagged)]
    Message(Message),
}

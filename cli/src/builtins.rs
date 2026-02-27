pub mod bulk_map;
pub mod chat;
pub mod email;
pub mod email2;
pub mod email3;
pub mod gsm8k;
pub mod sweagent;

#[cfg(feature = "rag")]
pub mod rag;

#[cfg(feature = "spnl-api")]
pub mod spans;

#[derive(clap::ValueEnum, Clone, Debug, serde::Serialize)]
#[clap(rename_all = "lowercase")]
pub enum Builtin {
    Chat,
    BulkMap,
    Email,
    Email2,
    Email3,
    SWEAgent,
    GSM8k,
    #[cfg(feature = "rag")]
    Rag,
    #[cfg(feature = "spnl-api")]
    Spans,
}

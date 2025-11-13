#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Message {
    /// Assistant output
    Assistant(String),

    /// User prompt
    User(String),

    /// System prompt
    System(String),
}

impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Message::User(s.to_string())
    }
}

impl Message {
    pub fn role(&self) -> &'static str {
        match self {
            Message::Assistant(_) => "assistant",
            Message::User(_) => "user",
            Message::System(_) => "system",
        }
    }
    pub fn content(&self) -> String {
        match self {
            Message::Assistant(s) | Message::User(s) | Message::System(s) => s.to_string(),
        }
    }
}

impl ::std::fmt::Display for Message {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Message::Assistant(s) => s,
                Message::User(s) => s,
                Message::System(s) => s,
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Document {
    Text(String),
    Binary(Vec<u8>),
}

#[derive(
    Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, derive_builder::Builder,
)]
pub struct Generate {
    pub model: String,
    pub input: Box<Query>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Repeat {
    pub n: usize,
    pub query: Box<Query>,
}

#[cfg(feature = "rag")]
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Augment {
    pub embedding_model: String,
    pub body: Box<Query>,
    pub doc: (String, Document),
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Query {
    /// Assistant output
    Assistant(String),

    /// User prompt
    User(String),

    /// System prompt
    System(String),

    /// Reduce
    Cross(Vec<Query>),

    /// Map
    Plus(Vec<Query>),

    /// Helpful for repeating an operation n times in a Plus
    Repeat(Repeat),

    /// Generate new content via a given model
    #[serde(rename = "g")]
    Generate(Generate),

    /// Incorporate information relevant to the question gathered from
    /// the given docs
    #[cfg(feature = "rag")]
    Augment(Augment),

    /// Ask with a given message
    #[cfg(feature = "cli_support")]
    Ask(String),

    /// Print a helpful message to the console
    #[cfg(feature = "cli_support")]
    Print(String),
}

#[cfg(feature = "cli_support")]
fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() < max_chars {
        return s.to_string();
    }

    match s.char_indices().nth(max_chars) {
        None => s.to_string(),
        Some((idx, _)) => format!("{}…", &s[..idx]),
    }
}

#[cfg(feature = "cli_support")]
fn trim(s: &str, max_chars: usize) -> String {
    truncate(s, max_chars).trim().replace("\n", " ")
}

#[cfg(feature = "cli_support")]
impl ptree::TreeItem for Query {
    type Child = Self;
    fn write_self<W: ::std::io::Write>(
        &self,
        f: &mut W,
        style: &ptree::Style,
    ) -> ::std::io::Result<()> {
        write!(
            f,
            "{}",
            match self {
                Query::Assistant(s) =>
                    style.paint(format!("\x1b[32mAssistant\x1b[0m {}", trim(s, 700))),
                Query::User(s) => style.paint(format!("\x1b[33mUser\x1b[0m {}", trim(s, 700))),
                Query::System(s) => style.paint(format!("\x1b[34mSystem\x1b[0m {}", trim(s, 700))),
                Query::Plus(_) => style.paint("\x1b[31;1mPlus\x1b[0m".to_string()),
                Query::Cross(_) => style.paint("\x1b[31;1mCross\x1b[0m".to_string()),
                Query::Generate(Generate { model, .. }) =>
                    style.paint(format!("\x1b[31;1mGenerate\x1b[0m \x1b[2m{model}\x1b[0m",)),
                Query::Repeat(Repeat { n, .. }) => style.paint(format!("Repeat {n}")),
                Query::Ask(m) => style.paint(format!("Ask {m}")),
                Query::Print(m) => style.paint(format!("Print {}", truncate(m, 700))),
                #[cfg(feature = "rag")]
                Query::Augment(_) => style.paint("\x1b[34;1mAugment\x1b[0m".to_string()),
            }
        )
    }
    fn children(&self) -> ::std::borrow::Cow<'_, [Self::Child]> {
        ::std::borrow::Cow::from(match self {
            Query::Ask(_)
            | Query::Assistant(_)
            | Query::User(_)
            | Query::System(_)
            | Query::Print(_) => vec![],
            Query::Plus(v) | Query::Cross(v) => v.clone(),
            Query::Repeat(Repeat { query, .. }) => vec![*query.clone()],
            Query::Generate(Generate { input, .. }) => vec![*input.clone()],
            #[cfg(feature = "rag")]
            Query::Augment(Augment {
                body,
                doc: (filename, _),
                ..
            }) => vec![
                *body.clone(),
                Query::User(format!("\x1b[35m<{filename}>\x1b[0m")),
            ],
        })
    }
}

impl ::std::fmt::Display for Query {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match self {
            Query::Cross(v) | Query::Plus(v) => write!(
                f,
                "{}",
                v.iter()
                    .map(|u| format!("{u}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            Query::System(s) | Query::User(s) => write!(f, "{s}"),
            _ => Ok(()),
        }
    }
}

impl From<&str> for Query {
    fn from(s: &str) -> Self {
        Self::User(s.into())
    }
}

impl ::std::str::FromStr for Query {
    type Err = Box<dyn ::std::error::Error>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::User(s.to_string()))
    }
}

impl From<&String> for Query {
    fn from(s: &String) -> Self {
        Self::User(s.clone())
    }
}

/// Pretty print a query
pub fn pretty_print(u: &Query) -> serde_json::Result<()> {
    println!("{}", serde_json::to_string(u)?);
    Ok(())
}

/// Serialize to JSON
pub fn to_string(q: &Query) -> serde_json::Result<String> {
    serde_json::to_string(q)
}

/// Deserialize a SPNL query from a string
pub fn from_str(s: &str) -> serde_json::Result<Query> {
    serde_json::from_str(s)
}

#[cfg(feature = "yaml")]
#[derive(Debug, Clone)]
pub struct FromYamlError {
    message: String,
}

#[cfg(feature = "yaml")]
impl From<serde::de::value::Error> for FromYamlError {
    fn from(e: serde::de::value::Error) -> Self {
        Self {
            message: e.to_string(),
        }
    }
}

#[cfg(feature = "yaml")]
impl ::std::error::Error for FromYamlError {
    fn description(&self) -> &str {
        self.message.as_str()
    }
}

#[cfg(feature = "yaml")]
impl ::std::fmt::Display for FromYamlError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[cfg(feature = "yaml")]
/// Deserialize a SPNL query from a YAML string
pub fn from_yaml_str(s: &str) -> Result<Query, FromYamlError> {
    Ok(serde_yaml2::from_str(s)?)
}

/// Deserialize a SPNL query from a reader
pub fn from_reader(r: impl ::std::io::Read) -> serde_json::Result<Query> {
    serde_json::from_reader(r)
}

/// Deserialize a SPNL query from a file path
pub fn from_file(f: &str) -> Result<Query, Box<dyn ::std::error::Error>> {
    Ok(serde_json::from_reader(::std::fs::File::open(f)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_user() -> serde_json::Result<()> {
        let result = from_str(r#"{"user": "hello"}"#)?;
        assert_eq!(result, Query::User("hello".to_string()));
        Ok(())
    }

    #[test]
    fn serde_system() -> serde_json::Result<()> {
        let result = from_str(r#"{"system": "hello"}"#)?;
        assert_eq!(result, Query::System("hello".to_string()));
        Ok(())
    }

    #[test]
    fn serde_ask() -> serde_json::Result<()> {
        let result = from_str(r#"{"ask": "hello"}"#)?;
        assert_eq!(result, Query::Ask("hello".to_string()));
        Ok(())
    }

    #[test]
    fn serde_plus_1() -> serde_json::Result<()> {
        let result = from_str(r#"{"plus": [{"user": "hello"}]}"#)?;
        assert_eq!(result, Query::Plus(vec![Query::User("hello".to_string())]));
        Ok(())
    }

    #[test]
    fn serde_plus_2() -> serde_json::Result<()> {
        let result = from_str(r#"{"plus": [{"user": "hello"},{"system": "world"}]}"#)?;
        assert_eq!(
            result,
            Query::Plus(vec![
                Query::User("hello".to_string()),
                Query::System("world".to_string())
            ])
        );
        Ok(())
    }

    #[test]
    fn serde_cross_1() -> serde_json::Result<()> {
        let result = from_str(r#"{"cross": [{"user": "hello"}]}"#)?;
        assert_eq!(result, Query::Cross(vec![Query::User("hello".to_string())]));
        Ok(())
    }

    #[test]
    fn serde_cross_3() -> serde_json::Result<()> {
        let result = from_str(
            r#"{"cross": [{"user": "hello"},{"system": "world"},{"plus": [{"user": "sloop"}]}]}"#,
        )?;
        assert_eq!(
            result,
            Query::Cross(vec![
                Query::User("hello".to_string()),
                Query::System("world".to_string()),
                Query::Plus(vec![Query::User("sloop".to_string())])
            ])
        );
        Ok(())
    }

    #[test]
    fn serde_gen() -> Result<(), Box<dyn ::std::error::Error>> {
        let result =
            from_str(r#"{"g": {"model": "ollama/granite3.2:2b", "input": {"user": "hello"}}}"#)?;
        assert_eq!(
            result,
            Query::Generate(
                GenerateBuilder::default()
                    .model("ollama/granite3.2:2b".into())
                    .input(Query::User("hello".to_string()).into())
                    .max_tokens(None)
                    .temperature(None)
                    .build()?
            )
        );
        Ok(())
    }
}

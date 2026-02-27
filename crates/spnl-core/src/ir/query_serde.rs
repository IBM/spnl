use super::{Message, Query};

impl From<&str> for Query {
    fn from(s: &str) -> Self {
        Self::Message(Message::User(s.into()))
    }
}

impl ::std::str::FromStr for Query {
    type Err = Box<dyn ::std::error::Error>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::Message(Message::User(s.to_string())))
    }
}

impl From<&String> for Query {
    fn from(s: &String) -> Self {
        Self::Message(Message::User(s.clone()))
    }
}

/// Turn a list of Query into a Seq
impl From<Vec<Query>> for Query {
    fn from(v: Vec<Query>) -> Self {
        match &v[..] {
            [q] => q.clone(),  // single-entry list doesn't need a Seq
            _ => Self::Seq(v), // otherwise, Seq is needed
        }
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
    use crate::ir::{GenerateBuilder, GenerateMetadataBuilder};

    #[test]
    fn serde_user() -> serde_json::Result<()> {
        let result = from_str(r#"{"user": "hello"}"#)?;
        assert_eq!(result, Query::Message(Message::User("hello".to_string())));
        Ok(())
    }

    #[test]
    fn serde_system() -> serde_json::Result<()> {
        let result = from_str(r#"{"system": "hello"}"#)?;
        assert_eq!(result, Query::Message(Message::System("hello".to_string())));
        Ok(())
    }

    #[test]
    fn serde_plus_1() -> serde_json::Result<()> {
        let result = from_str(r#"{"plus": [{"user": "hello"}]}"#)?;
        assert_eq!(
            result,
            Query::Plus(vec![Query::Message(Message::User("hello".to_string()))])
        );
        Ok(())
    }

    #[test]
    fn serde_plus_2() -> serde_json::Result<()> {
        let result = from_str(r#"{"plus": [{"user": "hello"},{"system": "world"}]}"#)?;
        assert_eq!(
            result,
            Query::Plus(vec![
                Query::Message(Message::User("hello".to_string())),
                Query::Message(Message::System("world".to_string()))
            ])
        );
        Ok(())
    }

    #[test]
    fn serde_cross_1() -> serde_json::Result<()> {
        let result = from_str(r#"{"cross": [{"user": "hello"}]}"#)?;
        assert_eq!(
            result,
            Query::Cross(vec![Query::Message(Message::User("hello".to_string()))])
        );
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
                Query::Message(Message::User("hello".to_string())),
                Query::Message(Message::System("world".to_string())),
                Query::Plus(vec![Query::Message(Message::User("sloop".to_string()))])
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
                    .input(Query::Message(Message::User("hello".to_string())).into())
                    .metadata(
                        GenerateMetadataBuilder::default()
                            .model("ollama/granite3.2:2b")
                            .build()?
                    )
                    .build()?
            )
        );
        Ok(())
    }
}

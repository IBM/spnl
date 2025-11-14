use super::*;

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
            Query::Message(m) => write!(f, "{m}"),
            _ => Ok(()),
        }
    }
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
                Query::Message(Message::Assistant(s)) =>
                    style.paint(format!("\x1b[32mAssistant\x1b[0m {}", trim(s, 700))),
                Query::Message(Message::User(s)) =>
                    style.paint(format!("\x1b[33mUser\x1b[0m {}", trim(s, 700))),
                Query::Message(Message::System(s)) =>
                    style.paint(format!("\x1b[34mSystem\x1b[0m {}", trim(s, 700))),
                Query::Seq(_) => style.paint("\x1b[31;1mSequence\x1b[0m".to_string()),
                Query::Par(_) => style.paint("\x1b[31;1mParallel\x1b[0m".to_string()),
                Query::Plus(_) => style.paint("\x1b[31;1mPlus\x1b[0m".to_string()),
                Query::Cross(_) => style.paint("\x1b[31;1mCross\x1b[0m".to_string()),
                Query::Generate(Generate {
                    metadata:
                        GenerateMetadata {
                            model, max_tokens, ..
                        },
                    ..
                }) => style.paint(format!(
                    "\x1b[31;1mGenerate\x1b[0m \x1b[2m{}model={model}\x1b[0m",
                    if let Some(mt) = max_tokens
                        && *mt != 0
                    {
                        format!("max_tokens={mt} ")
                    } else {
                        "".to_string()
                    }
                )),
                Query::Monad(_) => style.paint("\x1b[2mMonad\x1b[0m".to_string()),
                Query::Bulk(Bulk::Repeat(Repeat { n, .. })) => style.paint(format!("Repeat {n}")),
                Query::Bulk(Bulk::Map(Map { inputs, .. })) =>
                    style.paint(format!("Map {}", inputs.len())),
                Query::Ask(m) => style.paint(format!("Ask {m}")),
                Query::Print(m) => style.paint(format!("Print {}", truncate(m, 700))),
                #[cfg(feature = "rag")]
                Query::Augment(_) => style.paint("\x1b[34;1mAugment\x1b[0m".to_string()),
            }
        )
    }
    fn children(&self) -> ::std::borrow::Cow<'_, [Self::Child]> {
        ::std::borrow::Cow::from(match self {
            Query::Ask(_) | Query::Message(_) | Query::Print(_) | Query::Bulk(Bulk::Map(_)) => {
                vec![]
            }
            Query::Par(v) | Query::Seq(v) | Query::Plus(v) | Query::Cross(v) => v.clone(),
            Query::Monad(q) => vec![*q.clone()],
            Query::Bulk(Bulk::Repeat(Repeat { generate, .. })) => vec![*generate.input.clone()],
            Query::Generate(Generate { input, .. }) => vec![*input.clone()],
            #[cfg(feature = "rag")]
            Query::Augment(Augment {
                body,
                doc: (filename, _),
                ..
            }) => vec![
                *body.clone(),
                Query::Message(Message::User(format!("\x1b[35m<{filename}>\x1b[0m"))),
            ],
        })
    }
}

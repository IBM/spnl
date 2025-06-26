mod backend;
mod generate;
pub mod plan;
pub mod result;
#[cfg(feature = "rag")]
mod with;

use indicatif::MultiProgress;

use crate::{Generate, Query, run::result::SpnlResult};

pub struct RunParameters {
    /// URI of vector database. Could be a local filepath.
    pub vecdb_uri: String,

    /// Name of table to use in vector database.
    pub vecdb_table: String,

    /// Prepare query
    pub prepare: Option<bool>,
}

async fn cross(units: &[Query], rp: &RunParameters, mm: Option<&MultiProgress>) -> SpnlResult {
    let mym = MultiProgress::new();
    let m = if let Some(m) = mm { m } else { &mym };

    let mut evaluated = vec![];
    for u in units.iter() {
        evaluated.push(run(u, rp, Some(m)).await?);
    }

    Ok(Query::Cross(evaluated))
}

async fn plus(units: &[Query], rp: &RunParameters) -> SpnlResult {
    let m = MultiProgress::new();
    let evaluated =
        futures::future::try_join_all(units.iter().map(|u| run(u, rp, Some(&m)))).await?;

    if evaluated.len() == 1 {
        Ok(evaluated[0].clone())
    } else {
        Ok(Query::Plus(evaluated))
    }
}

#[async_recursion::async_recursion]
pub async fn run(unit: &Query, rp: &RunParameters, m: Option<&MultiProgress>) -> SpnlResult {
    #[cfg(feature = "pull")]
    crate::pull::pull_if_needed(unit).await?;

    match unit {
        Query::Print(m) => {
            println!("{}", m);
            Ok(Query::Print(m.clone()))
        }
        Query::User(s) => Ok(Query::User(s.clone())),
        Query::System(s) => Ok(Query::System(s.clone())),

        #[cfg(feature = "rag")]
        Query::Retrieve(crate::Retrieve {
            embedding_model,
            body,
            doc,
        }) => {
            crate::run::with::embed_and_retrieve(
                embedding_model,
                body,
                doc,
                rp.vecdb_uri.as_str(),
                rp.vecdb_table.as_str(),
            )
            .await
        }

        Query::Cross(u) => cross(u, rp, m).await,
        Query::Plus(u) => plus(u, rp).await,
        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
            accumulate,
        }) => match accumulate {
            None | Some(false) => {
                crate::run::generate::generate(
                    model.as_str(),
                    &run(input, rp, m).await?,
                    max_tokens,
                    temperature,
                    m,
                    rp.prepare.unwrap_or_default(),
                )
                .await
            }
            Some(true) => {
                let mut accum = match &**input {
                    Query::Cross(v) => v.clone(),
                    _ => vec![*input.clone()],
                };
                loop {
                    let program = Query::Generate(Generate {
                        model: model.clone(),
                        input: Box::new(Query::Cross(accum.clone())),
                        max_tokens: *max_tokens,
                        temperature: *temperature,
                        accumulate: None,
                    });
                    let out = run(&program, rp, m).await?;
                    accum.push(out.clone());
                }
            }
        },

        #[cfg(not(feature = "cli_support"))]
        Query::Ask(message) => todo!("ask"),
        #[cfg(feature = "cli_support")]
        Query::Ask(message) => {
            use rustyline::error::ReadlineError;
            let mut rl = rustyline::DefaultEditor::new().unwrap();
            let _ = rl.load_history("history.txt");
            let prompt = match rl.readline(message.as_str()) {
                Ok(line) => {
                    rl.add_history_entry(line.as_str()).unwrap();
                    line
                }
                Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                    ::std::process::exit(0) // TODO this only works in a CLI
                }
                Err(err) => panic!("{}", err), // TODO this only works in a CLI
            };
            rl.append_history("history.txt").unwrap();
            Ok(Query::User(prompt))
        }

        // should not happen
        Query::Repeat(_) => todo!("repeat"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::result::SpnlError;

    #[tokio::test]
    async fn it_works() -> Result<(), SpnlError> {
        let result = run(
            &"hello".into(),
            &RunParameters {
                prepare: None,
                vecdb_table: "".into(),
                vecdb_uri: "".into(),
            },
            None,
        )
        .await?;
        assert_eq!(result, Query::User("hello".to_string()));
        Ok(())
    }
}

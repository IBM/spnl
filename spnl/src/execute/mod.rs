use crate::ir::{Bulk, Generate, GenerateBuilder, Message::*, Query, Repeat};
use crate::optimizer::hlo::simplify;
use indicatif::MultiProgress;

#[cfg(feature = "pull")]
pub mod pull;

pub type ExecuteOptions = crate::generate::GenerateOptions;

pub type SpnlError = anyhow::Error;
pub type SpnlResult = anyhow::Result<Query>;

async fn seq(
    units: &[Query],
    rp: &ExecuteOptions,
    mm: Option<&MultiProgress>,
) -> anyhow::Result<Vec<Query>> {
    let mym = MultiProgress::new();
    let m = if let Some(m) = mm { m } else { &mym };

    let mut evaluated = vec![];
    for u in units.iter() {
        evaluated.push(run_subtree(u, rp, Some(m)).await?);
    }

    Ok(evaluated)
}

async fn par(units: &[Query], rp: &ExecuteOptions) -> SpnlResult {
    let m = MultiProgress::new();
    let evaluated =
        futures::future::try_join_all(units.iter().map(|u| run_subtree(u, rp, Some(&m)))).await?;

    if evaluated.len() == 1 {
        // the unwrap() is safe here, due to the len() == 1 guard
        Ok(evaluated.into_iter().next().unwrap())
    } else {
        Ok(Query::Par(evaluated))
    }
}

async fn plus(units: &[Query], rp: &ExecuteOptions) -> SpnlResult {
    let m = MultiProgress::new();
    let evaluated =
        futures::future::try_join_all(units.iter().map(|u| run_subtree(u, rp, Some(&m)))).await?;

    if evaluated.len() == 1 {
        // the unwrap() is safe here, due to the len() == 1 guard
        Ok(evaluated.into_iter().next().unwrap())
    } else {
        Ok(Query::Plus(evaluated))
    }
}

pub async fn execute(query: &Query, rp: &ExecuteOptions) -> SpnlResult {
    run_subtree(query, rp, None).await
}

#[async_recursion::async_recursion]
async fn run_subtree(query: &Query, rp: &ExecuteOptions, m: Option<&MultiProgress>) -> SpnlResult {
    Ok(simplify(&run_subtree_(query, rp, m).await?))
}

async fn run_subtree_(query: &Query, rp: &ExecuteOptions, m: Option<&MultiProgress>) -> SpnlResult {
    #[cfg(feature = "pull")]
    crate::execute::pull::pull_if_needed(query).await?;

    match query {
        Query::Message(_) => Ok(query.clone()),

        Query::Par(u) => par(u, rp).await,
        Query::Seq(u) => Ok(Query::Seq(seq(u, rp, m).await?)),
        Query::Cross(u) => Ok(Query::Cross(seq(u, rp, m).await?)),
        Query::Plus(u) => plus(u, rp).await,

        Query::Monad(q) => {
            // ignore output
            let _ = run_subtree(q, rp, m).await?;
            Ok("".into())
        }

        Query::Bulk(Bulk::Repeat(repeat)) => crate::generate::generate(repeat.clone(), m, rp).await,

        Query::Bulk(Bulk::Map(map)) => crate::generate::map(map, m, rp).await,

        Query::Generate(Generate { metadata, input }) => {
            crate::generate::generate(
                Repeat {
                    n: 1,
                    generate: GenerateBuilder::default()
                        .metadata(metadata.clone())
                        .input(Box::from(run_subtree(input, rp, m).await?))
                        .build()?,
                },
                m,
                rp,
            )
            .await
        }

        #[cfg(feature = "print")]
        Query::Print(m) => {
            if rp.time.is_none() {
                println!("{m}");
            }
            Ok(Query::Message(User("".into())))
        }
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
            Ok(Query::Message(User(prompt)))
        }

        // TODO: should not happen; we need to improve the typing of runnable queries
        #[cfg(feature = "rag")]
        Query::Augment(_) => todo!("augment"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn it_works() -> Result<(), SpnlError> {
        let result = execute(&"hello".into(), &ExecuteOptions::default()).await?;
        assert_eq!(result, Query::Message(User("hello".to_string())));
        Ok(())
    }

    #[tokio::test]
    async fn simplfiy_monad() -> Result<(), SpnlError> {
        let result = execute(
            &Query::Plus(vec![
                Query::Monad(Box::new("ignored".into())),
                "not ignored".into(),
            ]),
            &ExecuteOptions::default(),
        )
        .await?;
        assert_eq!(
            result,
            Query::Plus(vec![Query::Message(User("not ignored".to_string()))])
        );
        Ok(())
    }

    #[tokio::test]
    async fn simplfiy_nested_monad() -> Result<(), SpnlError> {
        let result = execute(
            &Query::Cross(vec![Query::Plus(vec![
                Query::Monad(Box::new("ignored".into())),
                "not ignored".into(),
            ])]),
            &ExecuteOptions::default(),
        )
        .await?;
        assert_eq!(
            result,
            Query::Cross(vec![Query::Plus(vec![Query::Message(User(
                "not ignored".to_string()
            ))])])
        );
        Ok(())
    }
}

use crate::{Generate, Message::*, Query};
use ::std::time::Instant;
use indicatif::MultiProgress;

pub struct ExecuteOptions {
    /// Prepare query?
    pub prepare: Option<bool>,
}

pub struct TimestampedQuery {
    pub finish_time: Instant,
    pub result: Query,
}

impl From<&Query> for TimestampedQuery {
    fn from(result: &Query) -> Self {
        Self {
            finish_time: Instant::now(),
            result: result.clone(),
        }
    }
}

impl From<Query> for TimestampedQuery {
    fn from(result: Query) -> Self {
        Self {
            finish_time: Instant::now(),
            result,
        }
    }
}

pub type SpnlError = anyhow::Error;
pub type SpnlResult = anyhow::Result<TimestampedQuery>;

async fn run_sequentially(
    units: &[Query],
    rp: &ExecuteOptions,
    mm: Option<&MultiProgress>,
    f: fn(Vec<Query>) -> Query,
) -> anyhow::Result<TimestampedQuery> {
    let mym = MultiProgress::new();
    let m = if let Some(m) = mm { m } else { &mym };

    let mut evaluated = vec![];
    for u in units.iter() {
        evaluated.push(run_subtree(u, rp, Some(m)).await?);
    }

    if evaluated.len() == 1 {
        // the unwrap() is safe here, due to the len() == 1 guard
        Ok(evaluated.into_iter().next().unwrap())
    } else {
        Ok(f(evaluated.into_iter().map(|q| q.result).collect()).into())
    }
}

async fn run_in_parallel(
    units: &[Query],
    rp: &ExecuteOptions,
    f: fn(Vec<Query>) -> Query,
) -> SpnlResult {
    let m = MultiProgress::new();
    let mut evaluated =
        futures::future::try_join_all(units.iter().map(|u| run_subtree(u, rp, Some(&m)))).await?;

    if evaluated.len() == 1 {
        // the unwrap() is safe here, due to the len() == 1 guard
        Ok(evaluated.into_iter().next().unwrap())
    } else {
        // Reverse sort the children output so that the first to finish is at the end
        evaluated.sort_by_key(|q| ::std::cmp::Reverse(q.finish_time));
        let max_finish_time = evaluated
            .first()
            .map(|q| q.finish_time)
            .unwrap_or_else(Instant::now);
        Ok(TimestampedQuery {
            finish_time: max_finish_time,
            result: f(evaluated.into_iter().map(|q| q.result).collect()),
        })
    }
}

pub async fn execute(query: &Query, rp: &ExecuteOptions) -> SpnlResult {
    run_subtree(query, rp, None).await
}

#[async_recursion::async_recursion]
async fn run_subtree(query: &Query, rp: &ExecuteOptions, m: Option<&MultiProgress>) -> SpnlResult {
    #[cfg(feature = "pull")]
    crate::pull::pull_if_needed(query).await?;

    match query {
        Query::Message(_) => Ok(query.into()),

        Query::Par(u) => run_in_parallel(u, rp, Query::Par).await,
        Query::Plus(u) => run_in_parallel(u, rp, Query::Plus).await,
        Query::Seq(u) => run_sequentially(u, rp, m, Query::Seq).await,
        Query::Cross(u) => run_sequentially(u, rp, m, Query::Cross).await,

        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
        }) => {
            crate::generate::generate(
                model.as_str(),
                &run_subtree(input, rp, m).await?.result,
                max_tokens,
                temperature,
                m,
                rp.prepare.unwrap_or_default(),
            )
            .await
        }

        #[cfg(feature = "cli_support")]
        Query::Print(m) => {
            println!("{m}");
            Ok(Query::Message(User("".into())).into())
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
            Ok(Query::Message(User(prompt)).into())
        }

        // TODO: should not happen; we need to improve the typing of runnable queries
        Query::Repeat(_) => todo!("repeat"),
        #[cfg(feature = "rag")]
        Query::Augment(_) => todo!("augment"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn it_works() -> Result<(), SpnlError> {
        let result = execute(&"hello".into(), &ExecuteOptions { prepare: None }).await?;
        assert_eq!(result.result, Query::Message(User("hello".to_string())));
        Ok(())
    }
}

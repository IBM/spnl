use crate::{Generate, Query};
use indicatif::MultiProgress;

pub struct ExecuteOptions {
    /// Prepare query?
    pub prepare: Option<bool>,
}

pub type SpnlError = anyhow::Error;
pub type SpnlResult = anyhow::Result<crate::Query>;

async fn cross(units: &[Query], rp: &ExecuteOptions, mm: Option<&MultiProgress>) -> SpnlResult {
    let mym = MultiProgress::new();
    let m = if let Some(m) = mm { m } else { &mym };

    let mut evaluated = vec![];
    for u in units.iter() {
        evaluated.push(run_subtree(u, rp, Some(m)).await?);
    }

    Ok(Query::Cross(evaluated))
}

async fn plus(units: &[Query], rp: &ExecuteOptions) -> SpnlResult {
    let m = MultiProgress::new();
    let evaluated =
        futures::future::try_join_all(units.iter().map(|u| run_subtree(u, rp, Some(&m)))).await?;

    if evaluated.len() == 1 {
        Ok(evaluated[0].clone())
    } else {
        Ok(Query::Plus(evaluated))
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
        Query::Assistant(_) | Query::User(_) | Query::System(_) => Ok(query.clone()),

        Query::Cross(u) => cross(u, rp, m).await,
        Query::Plus(u) => plus(u, rp).await,
        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
        }) => {
            crate::generate::generate(
                model.as_str(),
                &run_subtree(input, rp, m).await?,
                max_tokens,
                temperature,
                m,
                rp.prepare.unwrap_or_default(),
            )
            .await
        }

        #[cfg(not(feature = "cli_support"))]
        Query::Print(_) => todo!("Print capability is not enabled"),
        #[cfg(feature = "cli_support")]
        Query::Print(m) => {
            println!("{m}");
            Ok(Query::User(m.clone()))
        }
        #[cfg(not(feature = "cli_support"))]
        Query::Ask(_) => todo!("Ask capability is not enabled"),
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
        assert_eq!(result, Query::User("hello".to_string()));
        Ok(())
    }
}

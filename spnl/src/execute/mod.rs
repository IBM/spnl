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

    // Always keep Plus wrapping — even for single elements — so that PIC
    // tagging is preserved (prepare_fragment generates need the Plus sentinel).
    Ok(Query::Plus(evaluated))
}

/// Intersperse a in-between every element of b
fn intersperse(a: Query, b: Vec<Query>) -> Vec<Query> {
    match a {
        Query::Plus(p) => b
            .into_iter()
            .map(|bb| Query::Plus(p.iter().cloned().chain(std::iter::once(bb)).collect()))
            .collect(),
        _ => ::std::iter::repeat_n(a, b.len())
            .zip(b) // [(a1,b1),(a1,b2),...]
            .flat_map(|(a, b)| [a, b]) // [a1,b1,a1,b2,...]
            .collect(),
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

        Query::Zip(z) => {
            let f = run_subtree(&z.first, rp, m);
            let s = run_subtree(&z.second, rp, m);
            Ok(match s.await? {
                Query::Message(second) => {
                    Query::Seq(intersperse(f.await?, vec![Query::Message(second)]))
                }
                Query::Seq(second_list) => Query::Seq(intersperse(f.await?, second_list)),
                Query::Par(second_list) => Query::Par(intersperse(f.await?, second_list)),
                Query::Plus(second_list) => Query::Plus(intersperse(f.await?, second_list)),
                Query::Cross(second_list) => Query::Cross(intersperse(f.await?, second_list)),
                _ => todo!(),
            })
        }

        Query::Monad(q) => {
            // Execute for side effects (e.g. populating PIC cache), discard output.
            // Return empty Seq to avoid phantom empty-message tokens.
            let _ = run_subtree(q, rp, m).await?;
            Ok(Query::Seq(vec![]))
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
            if !rp.time {
                println!("{m}");
            }
            Ok(Query::Message(User("".into())))
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
    async fn zip_message_message() -> Result<(), SpnlError> {
        let a: Query = "aaa".into();
        let b: Query = "bbb".into();
        let expected = Query::Seq(vec![a.clone(), b.clone()]);
        let result = execute(&(a, b).into(), &ExecuteOptions::default()).await?;
        assert_eq!(result, expected);
        Ok(())
    }

    async fn zip_message_list(f: fn(Vec<Query>) -> Query) -> Result<(), SpnlError> {
        let a: Query = "aaa".into();
        let b: Query = "bbb".into();
        let c: Query = "ccc".into();
        let expected = f(vec![a.clone(), b.clone(), a.clone(), c.clone()]);
        let p = f(vec![b, c]);
        let result = execute(&(a, p).into(), &ExecuteOptions::default()).await?;
        assert_eq!(result, expected);
        Ok(())
    }

    #[tokio::test]
    async fn zip_message_par() -> Result<(), SpnlError> {
        zip_message_list(Query::Par).await
    }

    #[tokio::test]
    async fn zip_message_seq() -> Result<(), SpnlError> {
        zip_message_list(Query::Seq).await
    }

    #[tokio::test]
    async fn zip_message_plus() -> Result<(), SpnlError> {
        zip_message_list(Query::Plus).await
    }

    #[tokio::test]
    async fn zip_message_cross() -> Result<(), SpnlError> {
        zip_message_list(Query::Cross).await
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

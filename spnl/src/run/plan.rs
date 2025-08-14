use crate::{Generate, Query, Repeat};

pub struct PlanOptions {
    /// Max augmentations to add to the query
    pub max_aug: Option<usize>,

    /// URI of vector database. Could be a local filepath.
    pub vecdb_uri: String,

    /// Name of table to use in vector database.
    pub vecdb_table: String,
}

async fn plan_vec_iter(v: &[Query], po: &PlanOptions) -> anyhow::Result<Vec<Query>> {
    // TODO: this can't be the most efficient way to do this
    Ok(
        futures::future::try_join_all(v.iter().map(|u| plan_iter(u, po)))
            .await?
            .into_iter()
            .flatten()
            .collect(),
    )
}

fn cross_if_needed(v: Vec<Query>) -> Query {
    match &v[..] {
        [query] => query.clone(),
        _ => Query::Cross(v),
    }
}

#[async_recursion::async_recursion]
async fn plan_iter(query: &Query, po: &PlanOptions) -> anyhow::Result<Vec<Query>> {
    // this is probably the wrong place for this, but here we expand any Repeats under Plus or Cross
    match query {
        Query::Plus(v) => Ok(vec![Query::Plus(plan_vec_iter(v, po).await?)]),
        Query::Cross(v) => Ok(vec![Query::Cross(plan_vec_iter(v, po).await?)]),

        #[cfg(feature = "rag")]
        Query::Augment(a) => Ok(vec![
            crate::run::with::retrieve(&a.embedding_model, &a.body, &a.doc, po).await?,
        ]),

        Query::Repeat(Repeat { n, query }) => {
            let q = plan_iter(query, po).await?;
            Ok(::std::iter::repeat_n(q, *n).flatten().collect::<Vec<_>>())
        }

        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
            accumulate,
        }) => Ok(vec![Query::Generate(Generate {
            model: model.clone(),
            input: Box::new(cross_if_needed(plan_iter(input, po).await?)),
            max_tokens: *max_tokens,
            temperature: *temperature,
            accumulate: *accumulate,
        })]),

        otherwise => Ok(vec![otherwise.clone()]),
    }
}

/// This tries to remove some unnecessary syntactic complexities,
/// e.g. Plus-of-Plus or Cross with a tail Cross.
fn simplify(query: &Query) -> Query {
    match query {
        Query::Plus(v) => Query::Plus(match &v[..] {
            // Plus of Plus
            [Query::Plus(v2)] => v2.iter().map(simplify).collect(),

            otherwise => otherwise.iter().map(simplify).collect(),
        }),
        Query::Cross(v) => Query::Cross(match &v[..] {
            // Cross of tail Cross
            [.., Query::Cross(v2)] => v
                .iter()
                .take(v.len() - 1)
                .chain(v2.iter())
                .map(simplify)
                .collect(),

            otherwise => otherwise.iter().map(simplify).collect(),
        }),
        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
            accumulate,
        }) => Query::Generate(Generate {
            model: model.clone(),
            input: Box::new(simplify(input)),
            max_tokens: *max_tokens,
            temperature: *temperature,
            accumulate: *accumulate,
        }),
        otherwise => otherwise.clone(),
    }
}

pub async fn plan(query: &Query, po: &PlanOptions) -> anyhow::Result<Query> {
    #[cfg(feature = "rag")]
    crate::run::with::index::run(query, po).await?;

    Ok(simplify(&cross_if_needed(plan_iter(query, po).await?)))
}

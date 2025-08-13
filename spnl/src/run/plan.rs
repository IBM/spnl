use crate::{Generate, Query, Repeat};

pub struct PlanOptions {
    /// URI of vector database. Could be a local filepath.
    pub vecdb_uri: String,

    /// Name of table to use in vector database.
    pub vecdb_table: String,
}

async fn plan_vec_iter(v: &[Query], rp: &PlanOptions) -> anyhow::Result<Vec<Query>> {
    // TODO: this can't be the most efficient way to do this
    Ok(
        futures::future::try_join_all(v.iter().map(|u| plan_iter(u, rp)))
            .await?
            .into_iter()
            .flatten()
            .collect(),
    )
}

fn cross_if_needed(v: Vec<Query>) -> Query {
    match &v[..] {
        [query] => query.clone(),
        _ => Query::Cross(v.to_vec()),
    }
}

#[async_recursion::async_recursion]
async fn plan_iter(query: &Query, rp: &PlanOptions) -> anyhow::Result<Vec<Query>> {
    // this is probably the wrong place for this, but here we expand any Repeats under Plus or Cross
    match query {
        Query::Plus(v) => Ok(vec![Query::Plus(plan_vec_iter(v, rp).await?)]),
        Query::Cross(v) => Ok(vec![Query::Cross(plan_vec_iter(v, rp).await?)]),

        #[cfg(feature = "rag")]
        Query::Augment(a) => Ok(vec![
            crate::run::with::retrieve(
                &a.embedding_model,
                &a.body,
                &a.doc,
                rp.vecdb_uri.as_str(),
                rp.vecdb_table.as_str(),
            )
            .await?,
        ]),

        Query::Repeat(Repeat { n, query }) => {
            let q = plan_iter(query, rp).await?;
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
            input: Box::new(cross_if_needed(plan_iter(input, rp).await?)),
            max_tokens: *max_tokens,
            temperature: *temperature,
            accumulate: *accumulate,
        })]),

        otherwise => Ok(vec![otherwise.clone()]),
    }
}

pub async fn plan(query: &Query, rp: &PlanOptions) -> anyhow::Result<Query> {
    #[cfg(feature = "rag")]
    crate::run::with::index::run(query, rp.vecdb_uri.as_str(), rp.vecdb_table.as_str()).await?;

    Ok(cross_if_needed(plan_iter(query, rp).await?))
}

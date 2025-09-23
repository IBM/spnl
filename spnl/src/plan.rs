use crate::{Generate, Query, Repeat};

#[derive(Default)]
pub struct PlanOptions {
    #[cfg(feature = "rag")]
    pub aug: crate::augment::AugmentOptions,
}

/// Apply `plan_iter()` across the given list of Query
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

#[async_recursion::async_recursion]
async fn plan_iter(query: &Query, po: &PlanOptions) -> anyhow::Result<Vec<Query>> {
    match query {
        Query::Plus(v) => Ok(vec![Query::Plus(plan_vec_iter(v, po).await?)]),
        Query::Cross(v) => Ok(vec![Query::Cross(plan_vec_iter(v, po).await?)]),

        #[cfg(feature = "rag")]
        // Inline the retrieved fragments
        Query::Augment(a) => Ok(vec![Query::Plus(
            crate::augment::retrieve(&a.embedding_model, &a.body, &a.doc, &po.aug)
                .await?
                .map(|s| Query::Message(crate::Message::User(s)))
                .collect(),
        )]),

        // Unroll repeats
        Query::Repeat(Repeat { n, query }) => {
            let q = plan_iter(query, po).await?;
            Ok(::std::iter::repeat_n(q, *n).flatten().collect::<Vec<_>>())
        }

        // Nothing, except pass-through plan of the `input` field
        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
        }) => {
            let planned_input = plan_iter(input, po).await?;

            let nested_gen_input: Option<Query> = match &planned_input[..] {
                [Query::Seq(seq)] => match &seq[..] {
                    [Query::Message(m), Query::Plus(plus)] => {
                        match plus.iter().all(|q| matches!(q, Query::Generate(_))) {
                            true => Some(Query::Seq(vec![
                                Query::Message(m.clone()),
                                Query::Par(
                                    plus.iter()
                                        .filter_map(|q| match q {
                                            Query::Generate(g) => Some(Query::Plus(vec![
                                                *g.input.clone(),
                                                Query::Generate(g.clone()),
                                            ])),
                                            _ => None,
                                        })
                                        .collect(),
                                ),
                            ])),
                            false => None,
                        }
                    }
                    _ => None,
                },
                _ => None,
            };

            Ok(vec![Query::Generate(Generate {
                model: model.clone(),
                input: Box::new(nested_gen_input.unwrap_or_else(|| planned_input.into())),
                max_tokens: *max_tokens,
                temperature: *temperature,
            })])
        }

        otherwise => Ok(vec![otherwise.clone()]),
    }
}

/// This tries to remove some unnecessary syntactic complexities,
/// e.g. Plus-of-Plus or Cross with a tail Cross.
fn simplify(query: &Query) -> Query {
    match query {
        Query::Seq(v) => match &v[..] {
            // One-entry sequence
            [q] => simplify(q),

            otherwise => Query::Seq(
                otherwise
                    .iter()
                    .map(simplify)
                    .flat_map(|q| match q {
                        Query::Seq(v) => v, // Seq inside a Seq? flatten
                        _ => vec![q],
                    })
                    .collect(),
            ),
        },
        Query::Par(v) => match &v[..] {
            // One-entry parallel
            [q] => simplify(q),

            otherwise => Query::Par(otherwise.iter().map(simplify).collect()),
        },
        Query::Plus(v) => Query::Plus(match &v[..] {
            // Plus where the first element is either a Seq or
            // Plus. We can flatten that first element.
            [Query::Seq(v2), ..] | [Query::Plus(v2), ..] => {
                v2.iter().chain(&v[1..]).map(simplify).collect()
            }

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
        }) => Query::Generate(Generate {
            model: model.clone(),
            input: Box::new(simplify(input)),
            max_tokens: *max_tokens,
            temperature: *temperature,
        }),
        otherwise => otherwise.clone(),
    }
}

pub async fn plan(query: &Query, po: &PlanOptions) -> anyhow::Result<Query> {
    #[cfg(feature = "rag")]
    // Index the corpus (if needed)
    crate::augment::index(query, &po.aug).await?;

    // iterate the simplify a few times
    Ok(simplify(&simplify(&simplify(&simplify(&simplify(
        &plan_iter(query, po).await?.into(),
    ))))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerateBuilder, Message::*, Query::*, Repeat as Rep};

    #[test]
    fn simplify1() {
        // Message -> Message (i.e. no change)
        let q = Message(User("hello".into()));
        assert_eq!(simplify(&q), q);
    }

    #[test]
    fn simplify2() {
        // Seq(a) = a
        let a = Message(User("a".into()));
        let q = Seq(vec![a.clone()]);
        assert_eq!(simplify(&q), a);
    }

    #[test]
    fn simplify3() {
        // Seq(Seq(a)) = a
        let a = Message(User("a".into()));
        let qq = Seq(vec![a.clone()]);
        let q = Seq(vec![qq]);
        assert_eq!(simplify(&q), a);
    }

    #[test]
    fn simplify4() {
        // Plus(Seq(a,b), c, d) -> Plus(a,b,c,d)
        let a = Message(User("a".into()));
        let b = Message(User("b".into()));
        let c = Message(User("c".into()));
        let d = Message(User("d".into()));
        let qq = Seq(vec![a.clone(), b.clone()]);
        let q = Plus(vec![qq, c.clone(), d.clone()]);
        assert_eq!(simplify(&q), Plus(vec![a, b, c, d]));
    }

    #[tokio::test] // <-- needed for async tests
    async fn plan_repeat_expansion() -> anyhow::Result<()> {
        let n = 2;
        let m = Message(User("hello".into()));
        let q = Repeat(Rep {
            n,
            query: Box::new(m.clone()),
        });
        assert_eq!(
            plan(&q, &PlanOptions::default()).await?,
            Seq(::std::iter::repeat_n(m, n).collect())
        );
        Ok(())
    }

    #[tokio::test] // <-- needed for async tests
    async fn nested_gen() -> anyhow::Result<()> {
        let s2 = Message(System("outer system".into()));
        let s1 = Message(System("inner system".into()));
        let u1 = Message(User("inner user".into()));
        let inner_generate = Generate(
            GenerateBuilder::default()
                .model("m")
                .input(Box::new(Seq(vec![s1.clone(), u1.clone()])))
                .build()?,
        );
        let outer_generate = Generate(
            GenerateBuilder::default()
                .model("m")
                .input(Box::new(Seq(vec![
                    s2.clone(),
                    Plus(vec![inner_generate.clone()]),
                ])))
                .build()?,
        );
        assert_eq!(
            plan(&outer_generate, &PlanOptions::default()).await?,
            Generate(
                GenerateBuilder::default()
                    .model("m")
                    .input(Box::new(Seq(vec![s2, Plus(vec![s1, u1, inner_generate])])))
                    .build()?
            )
        );
        Ok(())
    }
}

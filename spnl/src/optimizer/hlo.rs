use crate::{
    generate::backend::capabilities::supports_spans,
    ir::{Bulk, Generate, GenerateBuilder, Query, Repeat},
};

#[cfg(feature = "rag")]
use crate::augment;

mod simplify;
use simplify::simplify;

#[derive(Debug, Default)]
pub struct Options {
    #[cfg(feature = "rag")]
    pub aug: augment::AugmentOptions,
}

#[derive(derive_builder::Builder)]
struct InheritedAttributes<'a> {
    #[builder(default)]
    parent_generate: Option<&'a Generate>,

    options: &'a Options,
}

impl<'a> From<&'a InheritedAttributes<'a>> for InheritedAttributesBuilder<'a> {
    fn from(other: &'a InheritedAttributes<'a>) -> Self {
        InheritedAttributesBuilder::default()
            .parent_generate(other.parent_generate)
            .options(other.options)
            .clone()
    }
}

/// Apply `optimize_iter()` across the given list of Query
async fn optimize_vec_iter<'a>(
    v: &[Query],
    attrs: &'a InheritedAttributes<'a>,
) -> anyhow::Result<Vec<Query>> {
    // TODO: this can't be the most efficient way to do this
    Ok(
        futures::future::try_join_all(v.iter().map(|u| optimize_iter(u, attrs)))
            .await?
            .into_iter()
            .collect(),
    )
}

/// Wrap a 1-token inner generate around each fragment
#[cfg(feature = "rag")]
fn prepare_fragment(m: &Query, parent_generate: Option<&Generate>) -> Option<Query> {
    if let Some(g) = parent_generate
        && supports_spans(&g.metadata.model)
    {
        Some(Query::Generate(
            GenerateBuilder::from(g)
                .metadata(
                    crate::ir::GenerateMetadataBuilder::from(&g.metadata)
                        .max_tokens(1)
                        .temperature(0.0)
                        .build()
                        .unwrap(), // TODO...
                )
                .input(Box::new(m.clone()))
                .build()
                .unwrap(), // TODO...
        ))
    } else {
        None
    }
}

/// Wrap a list of queries into a monad
#[cfg(feature = "rag")]
fn prepare_all(prepares: Vec<Query>) -> Option<Query> {
    if !prepares.is_empty() {
        Some(Query::Monad(Query::Plus(prepares).into()))
    } else {
        None
    }
}

#[async_recursion::async_recursion]
async fn optimize_iter<'a>(
    query: &Query,
    attrs: &'a InheritedAttributes<'a>,
) -> anyhow::Result<Query> {
    match query {
        Query::Plus(v) => Ok(Query::Plus(optimize_vec_iter(v, attrs).await?)),
        Query::Cross(v) => Ok(Query::Cross(optimize_vec_iter(v, attrs).await?)),

        #[cfg(feature = "rag")]
        // Inline the retrieved fragments, wrapping them with a 1-token nested generate
        Query::Augment(a) => {
            let (prepares, fragments): (Vec<_>, Vec<_>) =
                augment::retrieve(&a.embedding_model, &a.body, &a.doc, &attrs.options.aug)
                    .await?
                    .into_iter()
                    .map(|s| Query::Message(crate::ir::Message::User(s))) // we don't currently have a special type for fragments
                    .map(|m| (prepare_fragment(&m, attrs.parent_generate), m))
                    .unzip();

            Ok(Query::Seq(
                [
                    prepare_all(prepares.into_iter().flatten().collect()),
                    Some(Query::Plus(fragments)),
                ]
                .into_iter()
                .flatten()
                .collect(),
            ))
        }

        // Optimize the input of the Repeat
        Query::Bulk(Bulk::Repeat(Repeat { n, generate })) => {
            Ok(Query::Bulk(Bulk::Repeat(Repeat {
                n: *n,
                generate: GenerateBuilder::from(generate)
                    .input(optimize_iter(&generate.input, attrs).await?.into())
                    .build()?,
            })))
        }

        // Optimize for nested generate; Generate(Seq(Message, Plus(Generate, Generate, Generate)))
        Query::Generate(g) => {
            let optimized_input = optimize_iter(
                &g.input,
                &InheritedAttributesBuilder::from(attrs)
                    .parent_generate(Some(g))
                    .build()?,
            )
            .await?;

            let nested_gen_input: Option<Query> = if !supports_spans(&g.metadata.model) {
                None
            } else {
                match &optimized_input {
                    Query::Seq(seq) => match &seq[..] {
                        [Query::Message(m), Query::Plus(plus)] => {
                            // Plus of (only) Generates? TODO: that's all we handle, at the moment, nested generate where the children are *only* generates
                            match plus.iter().all(|q| matches!(q, Query::Generate(_))) {
                                // yes, we have a Plus of only Generates
                                true => Some(Query::Seq(vec![
                                    Query::Message(m.clone()),
                                    Query::Par(
                                        plus.iter()
                                            .filter_map(|q| match q {
                                                Query::Generate(g) => Some(Query::Plus(vec![
                                                    *g.input.clone(),
                                                    Query::Generate(g.wrap_plus()),
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
                }
            };

            Ok(Query::Generate(Generate {
                metadata: g.metadata.clone(),
                input: Box::new(nested_gen_input.unwrap_or(optimized_input)),
            }))
        }

        otherwise => Ok(otherwise.clone()),
    }
}

pub async fn optimize(query: &Query, po: &Options) -> anyhow::Result<Query> {
    #[cfg(feature = "rag")]
    // Index the corpus (if needed)
    augment::index(query, &po.aug).await?;

    // iterate the simplify a few times
    Ok(simplify(&simplify(&simplify(&simplify(&simplify(
        &optimize_iter(
            query,
            &InheritedAttributesBuilder::default().options(po).build()?,
        )
        .await?,
    ))))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Augment, Document, GenerateMetadataBuilder, Message::*, Query::Message};

    fn nested_gen_query(model: &str) -> anyhow::Result<(Query, Generate, Query, Query, Query)> {
        let s2 = Message(System("outer system".into()));
        let s1 = Message(System("inner system".into()));
        let u1 = Message(User("inner user".into()));
        let inner_generate = GenerateBuilder::default()
            .metadata(GenerateMetadataBuilder::default().model(model).build()?)
            .input(Box::new(Query::Seq(vec![s1.clone(), u1.clone()])))
            .build()?;
        let outer_generate = Query::Generate(
            GenerateBuilder::default()
                .metadata(GenerateMetadataBuilder::default().model(model).build()?)
                .input(Box::new(Query::Seq(vec![
                    s2.clone(),
                    Query::Plus(vec![Query::Generate(inner_generate.clone())]),
                ])))
                .build()?,
        );

        Ok((outer_generate, inner_generate, s2, s1, u1))
    }

    #[tokio::test] // <-- needed for async tests
    async fn nested_gen_expect_no_span_optimization() -> anyhow::Result<()> {
        let (outer_generate, _, _, _, _) = nested_gen_query("m")?;
        assert_eq!(
            optimize(&outer_generate, &Options::default()).await?,
            outer_generate,
        );
        Ok(())
    }

    #[tokio::test] // <-- needed for async tests
    async fn nested_gen_expect_span_optimization() -> anyhow::Result<()> {
        let model = "spnl/m";
        let (outer_generate, inner_generate, s2, s1, u1) = nested_gen_query(model)?;
        assert_eq!(
            optimize(&outer_generate, &Options::default()).await?,
            simplify(&Query::Generate(
                GenerateBuilder::default()
                    .metadata(GenerateMetadataBuilder::default().model(model).build()?)
                    .input(Box::new(Query::Seq(vec![
                        s2,
                        Query::Plus(vec![s1, u1, Query::Generate(inner_generate.wrap_plus())])
                    ])))
                    .build()?
            ))
        );
        Ok(())
    }

    #[tokio::test] // <-- needed for async tests
    async fn retrieve() -> anyhow::Result<()> {
        let model = "spnl/m"; // This should work, because we use SimpleEmbedRetrieve which won't do any generation
        let q = Message(User("Hello".to_string()));
        let d = "I know all about Hello and stuff";
        let outer_generate = GenerateBuilder::default()
            .metadata(GenerateMetadataBuilder::default().model(model).build()?)
            .input(Box::new(Query::Augment(Augment {
                embedding_model: "ollama/mxbai-embed-large:335m".to_string(),
                body: Box::new(q),
                doc: ("path/to/doc.txt".to_string(), Document::Text(d.to_string())),
            })))
            .build()?;

        let fragment = Message(User(format!("Relevant Document @base-doc.txt-0: {d}")));

        assert_eq!(
            optimize(
                &Query::Generate(outer_generate.clone()),
                &Options {
                    aug: augment::AugmentOptionsBuilder::default()
                        .indexer(augment::Indexer::SimpleEmbedRetrieve)
                        .build()?
                }
            )
            .await?,
            Query::Generate(
                GenerateBuilder::default()
                    .metadata(GenerateMetadataBuilder::default().model(model).build()?)
                    .input(Box::new(Query::Seq(
                        [
                            prepare_all(
                                [prepare_fragment(&fragment, Some(&outer_generate))]
                                    .into_iter()
                                    .flatten()
                                    .collect()
                            ),
                            Some(Query::Plus(vec![fragment]))
                        ]
                        .into_iter()
                        .flatten()
                        .collect()
                    )))
                    .build()?
            ),
        );
        Ok(())
    }
}

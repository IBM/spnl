use crate::generate::backend::capabilities::{supports_bulk_map, supports_bulk_repeat};
use crate::ir::{Bulk, Generate, Map, Message, Query, Repeat};

/// If every query in the given list is a Generate with equivalent
/// metadata, and the input of every Generate is the same kind of
/// Message, then group these Generates together into a Bulk::Map
// - TODO we can probably use a fold or scan to avoid temporary Vecs?
// - TODO we could find continguous sub-sequences that can be folded
// into a Bulk::Map, rather than only Bulk:Mapifying all or nothing
fn bulk_mapify(qs: &[Query]) -> Option<Map> {
    if qs.len() > 1 {
        let generates = qs
            .iter()
            .filter_map(|q| match q {
                Query::Generate(g) => Some(g),
                _ => None,
            })
            .collect::<Vec<_>>();

        if qs.len() == generates.len() {
            // all Generates
            if let Some(first) = generates.iter().cloned().next()
                && let Query::Message(first_message) = &*first.input
                && generates.len()
                    == generates
                        .iter()
                        .filter(|g| first.metadata == g.metadata)
                        .count()
            {
                // all the same metadata
                let inputs = generates
                    .into_iter()
                    .filter_map(|g| match (first_message, &*g.input) {
                        (Message::Assistant(_), Query::Message(Message::Assistant(s))) => {
                            Some(s.clone())
                        }
                        (Message::Assistant(_), _) => None,
                        (Message::System(_), Query::Message(Message::System(s))) => Some(s.clone()),
                        (Message::System(_), _) => None,
                        (Message::User(_), Query::Message(Message::User(s))) => Some(s.clone()),
                        (Message::User(_), _) => None,
                    })
                    .collect::<Vec<String>>();

                if qs.len() == inputs.len() {
                    return Some(Map {
                        metadata: first.metadata.clone(),
                        inputs,
                    });
                }
            }
        }
    }

    None
}

pub fn simplify(query: &Query) -> Query {
    simplify_iter(query).into()
}

/// This tries to remove some unnecessary syntactic complexities,
/// e.g. Plus-of-Plus or Cross with a tail Cross.
fn simplify_iter(query: &Query) -> Vec<Query> {
    match query {
        // Unroll repeats if the backend does not support their direct execution
        Query::Bulk(Bulk::Repeat(Repeat { n, generate })) => {
            if supports_bulk_repeat(&generate.metadata.model) {
                vec![query.clone()]
            } else {
                vec![Query::Par(
                    ::std::iter::repeat_n(Query::Generate(generate.clone()), (*n).into())
                        .collect::<Vec<_>>(),
                )]
            }
        }

        // Unroll batch. TODO: move this into the query executor backend, to expose server-side support for Map
        Query::Bulk(Bulk::Map(Map { metadata, inputs })) => {
            if supports_bulk_map(&metadata.model) {
                vec![query.clone()]
            } else {
                vec![Query::Plus(
                    inputs
                        .iter()
                        .map(|input| {
                            Query::Generate(Generate {
                                metadata: metadata.clone(),
                                input: Query::Message(Message::User(input.clone())).into(),
                            })
                        })
                        .collect(),
                )]
            }
        }

        Query::Seq(v) => match &v[..] {
            // Empty sequence
            [] => vec![],

            // One-entry sequence
            [q] => simplify_iter(q),

            otherwise => vec![Query::Seq(
                otherwise
                    .iter()
                    .flat_map(simplify_iter)
                    .flat_map(|q| match q {
                        Query::Seq(v) => v, // Seq inside a Seq? flatten
                        _ => vec![q],
                    })
                    .collect(),
            )],
        },
        Query::Par(v) => match &v[..] {
            // One-entry parallel
            [q] => simplify_iter(q),

            otherwise => vec![Query::Par(
                otherwise.iter().flat_map(simplify_iter).collect(),
            )],
        },
        Query::Plus(v) => {
            if v.is_empty() {
                vec![]
            } else if let Some(map) = bulk_mapify(v) {
                vec![Query::Bulk(Bulk::Map(map))]
            } else {
                vec![Query::Plus(match &v[..] {
                    // Plus where the first element is either a Seq or
                    // Plus. We can flatten that first element.
                    [Query::Seq(v2), ..] | [Query::Plus(v2), ..] => {
                        v2.iter().chain(&v[1..]).flat_map(simplify_iter).collect()
                    }

                    otherwise => otherwise
                        .iter()
                        .flat_map(simplify_iter)
                        .flat_map(|child| match child {
                            Query::Plus(v2) => v2,
                            _ => vec![child],
                        })
                        .collect(),
                })]
            }
        }
        Query::Cross(v) => vec![Query::Cross(match &v[..] {
            // Cross of tail Cross
            [.., Query::Cross(v2)] => v
                .iter()
                .take(v.len() - 1)
                .chain(v2.iter())
                .flat_map(simplify_iter)
                .collect(),

            otherwise => otherwise.iter().flat_map(simplify_iter).collect(),
        })],
        Query::Generate(Generate { input, metadata }) => vec![Query::Generate(Generate {
            metadata: metadata.clone(),
            input: Box::new(simplify(input)),
        })],

        Query::Monad(q) => vec![Query::Monad(Box::new(simplify(q)))],
        Query::Message(m) => {
            if m.is_empty() {
                vec![]
            } else {
                vec![Query::Message(m.clone())]
            }
        }

        otherwise => vec![otherwise.clone()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        Bulk, GenerateBuilder, GenerateMetadata, GenerateMetadataBuilder, Message::*, Query::*,
    };

    #[test]
    // Message -> Message (i.e. no change)
    fn simplify_noop() {
        let q = Message(User("hello".into()));
        assert_eq!(simplify(&q), q);
    }

    #[test]
    // Seq(a) = a
    fn simplify_singleton_vec() {
        let a = Message(User("a".into()));
        let q = Seq(vec![a.clone()]);
        assert_eq!(simplify(&q), a);
    }

    #[test]
    // Seq(Seq(a)) = a
    fn simplify_seq_of_seq() {
        let a = Message(User("a".into()));
        let qq = Seq(vec![a.clone()]);
        let q = Seq(vec![qq]);
        assert_eq!(simplify(&q), a);
    }

    #[test]
    // Plus(Seq(a,b), c, d) -> Plus(a,b,c,d)
    fn simplify_plus_of_seq() {
        let a = Message(User("a".into()));
        let b = Message(User("b".into()));
        let c = Message(User("c".into()));
        let d = Message(User("d".into()));
        let qq = Seq(vec![a.clone(), b.clone()]);
        let q = Plus(vec![qq, c.clone(), d.clone()]);
        assert_eq!(simplify(&q), Plus(vec![a, b, c, d]));
    }

    #[test]
    fn simplify_repeat_expansion() {
        let n = 2u8;
        let g = GenerateBuilder::default()
            .metadata(
                GenerateMetadataBuilder::default()
                    .model("disable_bulk_map/xxx") // <-- important: indicate the backend DOES NOT support Batch::Map
                    .build()
                    .unwrap(),
            )
            .input(Message(User("hello".to_string())).into())
            .build()
            .unwrap();
        let q = Bulk(Bulk::Repeat(Repeat {
            n,
            generate: g.clone(),
        }));
        assert_eq!(simplify(&q), Bulk(Bulk::Repeat(Repeat { n, generate: g })));
    }

    fn bulk_form(inputs: &[String], metadata: &GenerateMetadata) -> Query {
        Bulk(Bulk::Map(Map {
            metadata: metadata.clone(),
            inputs: inputs.to_vec(),
        }))
    }

    fn expanded_form_of_bulk_form(inputs: &[String], metadata: &GenerateMetadata) -> Query {
        Plus(vec![
            Generate(
                GenerateBuilder::default()
                    .metadata(metadata.clone())
                    .input(Message(User(inputs[0].clone())).into())
                    .build()
                    .unwrap(),
            ),
            Generate(
                GenerateBuilder::default()
                    .metadata(metadata.clone())
                    .input(Message(User(inputs[1].clone())).into())
                    .build()
                    .unwrap(),
            ),
            Generate(
                GenerateBuilder::default()
                    .metadata(metadata.clone())
                    .input(Message(User(inputs[2].clone())).into())
                    .build()
                    .unwrap(),
            ),
        ])
    }

    // Note: this shouold be the inverse of simplify_batch_consolidation()
    #[test]
    fn simplify_batch_expansion() {
        let inputs = ["a".into(), "b".into(), "c".into()];
        let metadata = GenerateMetadataBuilder::default()
            .model("disable_bulk_map/xxx") // <-- important: indicate the backend DOES NOT support Batch::Map
            .build()
            .unwrap();
        assert_eq!(
            simplify(&bulk_form(&inputs, &metadata)),
            expanded_form_of_bulk_form(&inputs, &metadata)
        );
    }

    // Note: this shouold be the inverse of simplify_batch_expansion()
    #[test]
    fn simplify_batch_consolidation() {
        let inputs = ["a".into(), "b".into(), "c".into()];
        let metadata = GenerateMetadataBuilder::default()
            .model("openai/xxx") // <-- important: indicate the backend DOES support Batch::Map
            .build()
            .unwrap();
        assert_eq!(
            simplify(&expanded_form_of_bulk_form(&inputs, &metadata)),
            bulk_form(&inputs, &metadata),
        );
    }
}

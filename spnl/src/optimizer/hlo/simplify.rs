use crate::ir::{Generate, Query, Repeat};

pub fn simplify(query: &Query) -> Query {
    simplify_iter(query).into()
}

/// This tries to remove some unnecessary syntactic complexities,
/// e.g. Plus-of-Plus or Cross with a tail Cross.
fn simplify_iter(query: &Query) -> Vec<Query> {
    match query {
        // Unroll repeats
        Query::Bulk(Repeat { n, query }) => {
            let q = simplify_iter(query);
            ::std::iter::repeat_n(q, *n).flatten().collect::<Vec<_>>()
        }

        Query::Seq(v) => match &v[..] {
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
        Query::Plus(v) => vec![Query::Plus(match &v[..] {
            // Plus where the first element is either a Seq or
            // Plus. We can flatten that first element.
            [Query::Seq(v2), ..] | [Query::Plus(v2), ..] => {
                v2.iter().chain(&v[1..]).flat_map(simplify_iter).collect()
            }

            otherwise => otherwise.iter().flat_map(simplify_iter).collect(),
        })],
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
        otherwise => vec![otherwise.clone()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Message::*, Query::*};

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
        let n = 2;
        let m = Message(User("hello".to_string()));
        let q = Bulk(Repeat {
            n,
            query: Box::new(m.clone()),
        });
        assert_eq!(simplify(&q), Seq(::std::iter::repeat_n(m, n).collect()));
    }
}

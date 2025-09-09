// Inspiration: https://github.com/JunSuzukiJapan/macro-lisp
#[macro_export]
macro_rules! spnl {
    // Core: Generate text given $input using $model
    (g $model:tt $input:tt) => ($crate::spnl!(g $model $input 0.0 0));

    // Core: Generate text given $input using $model with temperature $temp
    (g $model:tt $input:tt $temp:tt) => ($crate::spnl!(g $model $input $temp 0));

    // Core: Generate text given $input using $model with temperature $temp and $max_tokens
    (g $model:tt $input:tt $temp:tt $max_tokens:tt) => (
        $crate::Query::Generate($crate::Generate {
            model: $crate::spnl_arg!($model).to_string(),
            input: Box::new($crate::spnl_arg!($input).into()),
            max_tokens: Some($crate::spnl_arg!($max_tokens)),
            temperature: Some($crate::spnl_arg!($temp)),
        })
    );

    // Core: execute serially
    (seq $e:tt) => ( $crate::Query::Seq($crate::spnl_arg!( $e )) );
    (seq $( $e:tt )+) => ( $crate::Query::Seq(vec![$( $crate::spnl_arg!( $e ).into() ),+]) );

    // Core: execute parallel
    (par $e:tt) => ( $crate::Query::Par($crate::spnl_arg!( $e )) );
    (par $( $e:tt )+) => ( $crate::Query::Par(vec![$( $crate::spnl_arg!( $e ).into() ),+]) );

    // Core: Dependent/needs-attention
    (cross $( $e:tt )+) => ( $crate::Query::Cross(vec![$( $crate::spnl_arg!( $e ).into() ),+]) );

    // Core: Independent/no-attention with one or more inputs provided directly as a vector
    (plus $e:tt) => ( $crate::Query::Plus($crate::spnl_arg!( $e )) );

    // Core: Independent/no-attention with multiple inputs provided inline
    (plus $( $e:tt )+) => ( $crate::Query::Plus(vec![$( $crate::spnl_arg!( $e ).into() ),+]) );

    // Core: A user message
    (user $e:tt) => ($crate::Query::Message($crate::Message::User($crate::spnl_arg!($e).clone().into())));

    // Core: A system message
    (system $e:tt) => ($crate::Query::Message($crate::Message::System($crate::spnl_arg!($e).into())));

    // Data: incorporate a file at compile time
    (file $f:tt) => (include_str!($crate::spnl_arg!($f)));

    // Data: incorporate a file at compile time, preserving file name
    (filen $f:tt) => (($crate::spnl_arg!($f).to_string(), include_str!($crate::spnl_arg!($f)).to_string()));

    // Data: incorporate a file at run time
    (fetch $f:tt) => (match $crate::spnl!(fetchn $f).1 { $crate::Document::Text(a) => a,  $crate::Document::Binary(b) => String::from_utf8(b).expect("string") });

    // Data: incorporate a file at run time, preserving file name
    (fetchn $f:tt) => {{
        let filename = ::std::path::Path::new(file!()).parent().expect("macro to have parent directory").join($crate::spnl_arg!($f));
        (filename.clone().into_os_string().into_string().expect("filename"), $crate::Document::Text(::std::fs::read_to_string(filename).expect("error reading file")))
    }};

    // Data: incorporate a binary file at run time, preserving file name
    (fetchb $f:tt) => {{
        let filename = ::std::path::Path::new(file!()).parent().expect("macro to have parent directory").join($crate::spnl_arg!($f));
        (filename.clone().into_os_string().into_string().expect("filename"), $crate::Document::Binary(::std::fs::read(filename).expect("error reading file")))
    }};

    // Data: peel off the first $n elements of the given serialized
    // json vector of strings (TODO: split this into multiple macros)
    (take $n:tt $s:tt) => (
        serde_json::from_str::<Vec<String>>($crate::spnl_arg!($s))?
            .into_iter()
            .take($crate::spnl_arg!($n).try_into().expect("usize"))
            .collect::<Vec<_>>()
    );

    // Data: prefix every string in $arr with $p
    (prefix $p:tt $arr:tt) => (
        $crate::spnl_arg!($arr)
            .into_iter()
            .enumerate()
            .map(|(idx, s)| ((1 + idx), s)) // (idx % $crate::spnl_arg!($chunk_size)), s))
            .map(|(idx, s)| $crate::spnl!(user (format "{}{idx}: {:?}" $p s)))
            .collect::<Vec<_>>()
    );

    // Data: break up the array $arr into chunks of maximum size
    // $chunk_size characters and send each chunk to the given
    // (lambda) $f.
    (chunk $chunk_size:tt $arr:tt $f:tt) => (
        $crate::spnl_arg!($arr)
            .chunks($crate::spnl_arg!($chunk_size))
            .map(|chunk| chunk.to_vec())
            .map($crate::spnl_arg!($f))
            .collect::<Vec<_>>()
    );

    // Data: augment a prompt with relevant fragments from one or more documents
    (with $embedding_model:tt $input:tt $docs:tt) => {{
        let docs: Vec<$crate::Query> = $crate::spnl_arg!($docs)
            .into_iter()
            .map(|doc| match ::std::path::Path::new(&doc)
                 .extension()
                 .and_then(std::ffi::OsStr::to_str) {
                     Some("txt") | Some("json") | Some("jsonl") => $crate::spnl!(fetchn doc),
                     _ => $crate::spnl!(fetchb doc),
                 })
            .map(|doc| $crate::spnl!(__spnl_retrieve $embedding_model $input doc))
            .collect();

        // Note how (with embedding_model "question" "document")
        // macro-expands to a "cross of plus" pattern. Each inside
        // "plus" uses (just above) an `Augment` tree node to defer
        // (beyond compile time, which is where we are now) the
        // indexing and retrieval logic. We could perhaps do the
        // indexing at compile time (probably not super useful, but a
        // possibility), but the retrieval in general by definition is
        // not a compile-time
        $crate::spnl!(
            cross
                (plus docs)
                (user "Please answer this question:")
                $input
        )
    }};

    // Internal
    (__spnl_retrieve $embedding_model:tt $input:tt $doc:tt) => (
        $crate::Query::Augment($crate::Augment {
            embedding_model: $crate::spnl_arg!($embedding_model).clone(),
            body: Box::new($crate::spnl_arg!($input)),
            doc: $crate::spnl_arg!( $doc ).into()
        })
    );

    // Sugar: this unfolds to a `(g $model (cross $body))` but with
    // special user and system messages geared at extracting,
    // simplifying, and summarizing the thought process of the output
    // of prior (g) calls.
    (extract $model:tt $n:tt $body:tt) => {{
        let n = $crate::spnl_arg!($n);
        $crate::spnl!(
            g $model (cross
                      (system "Your are an AI that combines prior outputs from other AIs, preferring no markdown or other exposition.")
                      $body
                      (user (format "Extract and simplify these {} final answers" n))))
    }};

    // Sugar: this unfolds to a `(g $model (cross $body))` but with
    // special user and system messages geared at combining the
    // output of prior (g) calls.
    (combine $model:tt $body:tt) => (
        $crate::spnl!(
            g $model (cross
                      (system "Your are an AI that combines prior outputs from other AIs, preferring no markdown or other exposition.")
                      $body
                      (user "Combine and flatten these into one JSON array, preserving order")))
    );

    // Sugar: this unfolds to repeating the given expression $e $n times.
    (repeat $n:tt $e:tt) => (spnl!(repeat i $n $e));

    // Sugar: this unfolds to repeating the given expression $e $n
    // times and makes available an index variable $i ranging from 0
    // to $n-1.
    (repeat $i:ident $n:tt $e:tt) => (spnl!(repeat $i 0 $n $e));

    // Sugar: this unfolds to repeating the given expression $e $n
    // times and makes available an index variable $i ranging from
    // $start to $n-$start-1.
    (repeat $i:ident $start:tt $n:tt $e:tt) => {{
        let mut args: Vec<$crate::Query> = vec![];
        let start = $crate::spnl_arg!($start);
        let end = $crate::spnl_arg!($n) + start;
        for $i in start..end {
            args.push($crate::spnl_arg!($e).clone());
        }
        args
    }};

    // Utility: Defines an n-ary function that accepts the given $name'd arguments
    (lambda ( $( $name:ident )* )
     $( ( $($e:tt)* ))*
    ) => (| $($name: Vec<Query>),* |{ $( $crate::spnl!( $($e)* ) );* });

    // Utility: the length of $list
    (length $list:tt) => ($crate::spnl_arg!($list).len());

    // Utility: read as string from stdin
    (ask $message:tt) => ( $crate::Query::Ask($crate::spnl_arg!($message).into()) );

    // Utility: print a helpful message to the console
    (print $message:tt) => ( $crate::Query::Print($crate::spnl_arg!($message).into()) );

    // Utility:
    (format $fmt:tt $( $e:tt )*) => ( &format!($fmt, $($crate::spnl_arg!($e)),* ) );

    // execute rust
    //(rust $( $st:stmt )* ) => ( $($st);* );
    // other
    //($e:expr) => ($e.into());
}

#[macro_export]
macro_rules! spnl_arg {
    ( ( $($e:tt)* ) ) => ( $crate::spnl!( $($e)* ) );
    ($e:expr) => ($e);
}

#[cfg(test)]
mod tests {
    use crate::{Message::*, Query};

    #[test]
    fn macro_user() {
        let result = spnl!(user "hello");
        assert_eq!(result, Query::Message(User("hello".to_string())));
    }

    #[test]
    fn macro_system() {
        let result = spnl!(system "hello");
        assert_eq!(result, Query::Message(System("hello".to_string())));
    }

    #[test]
    fn macro_ask() {
        let result = spnl!(ask "hello");
        assert_eq!(result, Query::Ask("hello".to_string()));
    }

    #[test]
    fn macro_plus_1() {
        let result = spnl!(plus (user "hello") (user "world"));
        assert_eq!(
            result,
            Query::Plus(vec![
                Query::Message(User("hello".to_string())),
                Query::Message(User("world".to_string()))
            ])
        );
    }

    #[test]
    fn macro_plus_2() {
        let result = spnl!(plus (user "hello") (system "world"));
        assert_eq!(
            result,
            Query::Plus(vec![
                Query::Message(User("hello".to_string())),
                Query::Message(System("world".to_string()))
            ])
        );
    }

    #[test]
    fn macro_cross_1() {
        let result = spnl!(cross (user "hello"));
        assert_eq!(
            result,
            Query::Cross(vec![Query::Message(User("hello".to_string()))])
        );
    }

    #[test]
    fn macro_cross_3() {
        let result =
            spnl!(cross (user "hello") (system "world") (plus (user "sloop") (user "boop")));
        assert_eq!(
            result,
            Query::Cross(vec![
                Query::Message(User("hello".to_string())),
                Query::Message(System("world".to_string())),
                Query::Plus(vec![
                    Query::Message(User("sloop".to_string())),
                    Query::Message(User("boop".to_string()))
                ])
            ])
        );
    }

    #[test]
    fn macro_gen() -> Result<(), Box<dyn ::std::error::Error>> {
        let result = spnl!(g "ollama/granite3.2:2b" (user "hello") 0.0 0);
        assert_eq!(
            result,
            Query::Generate(
                crate::GenerateBuilder::default()
                    .model("ollama/granite3.2:2b".to_string())
                    .input(Box::new(Query::Message(User("hello".to_string()))))
                    .max_tokens(Some(0))
                    .temperature(Some(0.0))
                    .build()?
            )
        );
        Ok(())
    }
}

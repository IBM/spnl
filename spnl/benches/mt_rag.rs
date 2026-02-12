#![cfg(feature = "rag")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use flate2::read::GzDecoder;
use spnl::{
    ExecuteOptions, execute,
    ir::{Document, Query},
    spnl,
};
use std::io::{BufRead, BufReader};

/// Load questions from the compressed questions file
fn load_questions(limit: Option<usize>) -> Vec<String> {
    let questions_gz = include_bytes!("questions.txt.gz");
    let decoder = GzDecoder::new(&questions_gz[..]);
    let reader = BufReader::new(decoder);

    let questions: Vec<String> = reader
        .lines()
        .filter_map(|line: Result<String, _>| line.ok())
        .collect();

    if let Some(limit) = limit {
        questions.into_iter().take(limit).collect()
    } else {
        questions
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum BenchmarkSize {
    Small,  // 10 questions
    Medium, // 50 questions
    Large,  // 200 questions
    Full,   // All 1467 questions
}

impl BenchmarkSize {
    fn limit(&self) -> Option<usize> {
        match self {
            BenchmarkSize::Small => Some(10),
            BenchmarkSize::Medium => Some(50),
            BenchmarkSize::Large => Some(200),
            BenchmarkSize::Full => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            BenchmarkSize::Small => "small",
            BenchmarkSize::Medium => "medium",
            BenchmarkSize::Large => "large",
            BenchmarkSize::Full => "full",
        }
    }
}

/// Creates a RAG query for a given question
fn create_rag_query(
    model: &str,
    embedding_model: &str,
    question: &str,
    docs: Vec<(String, Document)>,
    temperature: f32,
    max_tokens: i32,
) -> Query {
    let model = model.to_string();
    let embedding_model = embedding_model.to_string();
    let prompt = format!("Question: {}", question);

    let system_prompt = r#"
Your answer questions using information from the given Relevant Documents, and cite them. For example:

Question: How do trees grow?
Answer: Via carbon dioxide.
Citations: @base-foo-37, @raptor-bar-52

Question: How does hair grow?
Answer: Slowly.
Citations: @base-baz-2, @raptor-glam-8
"#;

    spnl!(
        g model
            (cross
             (system system_prompt)
             (with embedding_model (user prompt) docs))
            temperature max_tokens
    )
}

async fn run_rag_benchmark(
    model: &str,
    embedding_model: &str,
    question: &str,
    docs: Vec<(String, Document)>,
    temperature: f32,
    max_tokens: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let query = create_rag_query(
        model,
        embedding_model,
        question,
        docs,
        temperature,
        max_tokens,
    );
    execute(&query, &ExecuteOptions::default()).await?;
    Ok(())
}

fn mt_rag_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let model = "ollama/granite3.3:2b";
    let embedding_model = "ollama/mxbai-embed-large:335m";
    let temperature = 0.0;
    let max_tokens = 100; // Use small token limit for faster benchmarking

    // Create a simple document corpus for testing
    // In production, you would load actual documents from the datasets
    let docs = vec![
        (
            "doc1".to_string(),
            Document::Text("The Arizona Cardinals are a professional American football team based in the Phoenix metropolitan area.".to_string())
        ),
        (
            "doc2".to_string(),
            Document::Text("The NFL consists of 32 teams divided into two conferences: the American Football Conference (AFC) and the National Football Conference (NFC).".to_string())
        ),
        (
            "doc3".to_string(),
            Document::Text("The New England Patriots have won six Super Bowl championships, tied for the most in NFL history.".to_string())
        ),
        (
            "doc4".to_string(),
            Document::Text("In emergency preparedness, it's recommended to keep a three-day supply of water and non-perishable food in your safe room.".to_string())
        ),
        (
            "doc5".to_string(),
            Document::Text("California experiences more wildfires than most other states due to its climate and vegetation.".to_string())
        ),
    ];

    // Benchmark different dataset sizes
    for size in [BenchmarkSize::Small, BenchmarkSize::Medium] {
        let mut group = c.benchmark_group(format!("mt_rag_{}", size.name()));

        // Configure sample size based on benchmark size
        group.sample_size(match size {
            BenchmarkSize::Small => 10,
            BenchmarkSize::Medium => 5,
            BenchmarkSize::Large => 3,
            BenchmarkSize::Full => 1,
        });

        let questions = load_questions(size.limit());

        // Benchmark a subset of questions from this size
        let test_questions: Vec<_> = questions.iter().take(3).collect();

        for (idx, question) in test_questions.iter().enumerate() {
            group.bench_with_input(
                BenchmarkId::new("rag_query", idx),
                question,
                |b, question| {
                    b.to_async(&runtime).iter(|| {
                        let docs_clone = docs.clone();
                        let question = question.to_string();
                        async move {
                            run_rag_benchmark(
                                model,
                                embedding_model,
                                &question,
                                docs_clone,
                                temperature,
                                max_tokens,
                            )
                            .await
                            .unwrap()
                        }
                    });
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, mt_rag_benchmark);
criterion_main!(benches);

// Made with Bob

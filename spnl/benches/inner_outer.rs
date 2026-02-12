use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use spnl::{ExecuteOptions, execute, ir::Query, spnl};

/// Creates the email2 query with n inner generations
fn create_email2_query(model: &str, n: usize, temperature: f32, max_tokens: i32) -> Query {
    let model = model.to_string();
    spnl!(
        g model
         (seq
          (system "You compute an evaluation score from 0 to 100 that ranks given candidate introductory emails. Better emails are ones that mention specifics, such as names of people and companies. You present a list of the top 3 ordered by their rank showing the score and full content of each.")

          (print (format "Generate {n} candidate emails in parallel"))

          (repeat n
           model
            (seq
             (system (format "You write an introductory email for a job application, paying attention to the specifics of the application, and limited to at most {max_tokens} characters."))

             (user "My name is Shiloh. I am a data scientist with 10 years of experience and need an introductory email to apply for a position at IBM in their research department")
            )

            temperature max_tokens
          )

          (print "Ask the model to select the best option from the candidates")
         )

            temperature max_tokens
    )
}

async fn run_inner_outer_benchmark(
    model: &str,
    n: usize,
    temperature: f32,
    max_tokens: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let query = create_email2_query(model, n, temperature, max_tokens);
    execute(&query, &ExecuteOptions::default()).await?;
    Ok(())
}

fn inner_outer_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("inner_outer");

    // Configure to run fewer samples since these are expensive operations
    group.sample_size(10);

    let model = "ollama/granite3.3:2b";
    let temperature = 0.0;
    let max_tokens = 1000;

    // Benchmark with different numbers of inner generations
    // The original script tests 1-32, but we'll test a subset for reasonable benchmark times
    for n in [1, 2, 4, 8, 16] {
        group.bench_with_input(BenchmarkId::new("email_generation", n), &n, |b, &n| {
            b.to_async(&runtime).iter(|| async move {
                run_inner_outer_benchmark(model, n, temperature, max_tokens)
                    .await
                    .unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, inner_outer_benchmark);
criterion_main!(benches);

// Made with Bob

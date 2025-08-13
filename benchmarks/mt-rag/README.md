# Multi-turn RAG Benchmark

Reference: https://github.com/IBM/mt-rag-benchmark

> A comprehensive and diverse human-generated multi-turn RAG dataset,
> accompanied by four document corpora. To the best of our knowledge,
> MTRAG is the first end-to-end human-generated multi-turn RAG
> benchmark that reflects real-world properties of multi-turn
> conversations.

The benchmark consists of four datasets: clapnq, cloud fiqa, and
govt. A shortened version of fiqa is included in this repo as
`fiqa-first100lines.jsonl.gz`.

## Running with SPNL

> [!IMPORTANT]
> Ensure you have enabled the `rag` feature flag, by adding `-F rag`
> either to your `cargo build` or `cargo run` command lines.  In the
> following, we will assume you have used `cargo build -F rag` to
> build a `spnl` executable (normally placed in
> `./target/release/spnl`).

```shell
spnl \
  --builtin rag \
  --prompt 'can i use my flexible savings account to pay for health insurance premiums?' \
  --document fiqa-first100lines.jsonl
```

Use `-m ollama/granite3.2:2b` to use a specified completion model and
`-e ollama/mxbai-embed-large:335m` to use a specified embedding model
(both settings have reasonable default values).

This command line uses the `rag` builtin to process the given question
(`-w`) with the given augmentation documents (`-u`; you may provide as
many `-u` options as you need, e.g. `-u doc1 -u doc2`). The rag
builtin is simple sugar of a span query:

```lisp
(g model
  (cross
    (system system_prompt)
    (with embedding_model (user question) docs)))
```

### Running the entire benchmark

The file questions.txt.gz contains all of the questions from the
benchmark, flattened in a linear newline-separated list. 

This performs a separate run for each dataset. It uses `--max-tokens
1` to collect TTFT. If you care to compute the accuracy, see
[below](#computing-accuracy).

```shell
for c in fiqa.jsonl clapnq.jsonl cloud.jsonl govt.jsonl
do cat questions.txt | xargs -P1 -n1 spnl -b rag -u $c --max-tokens 1 -w  >& $c.out.txt
done
```

If you wish to use augment each question with fragments (possibly)
from all four datasets:

```shell
gzcat benchmarks/mt-rag/questions.txt.gz | xargs -P1 -n1 spnl -b rag -d fiqa.jsonl -d clapnq.jsonl -d cloud.jsonl -d govt.jsonl -p  >& all.out.txt
```

### Downloading full datasets

- [clapnq.jsonl.gz](https://mtrag.s3.us-east.cloud-object-storage.appdomain.cloud/clapnq.jsonl.gz)
- [cloud.jsonl.gz](https://mtrag.s3.us-east.cloud-object-storage.appdomain.cloud/cloud.jsonl.gz)
- [fiqa.jsonl.gz](https://mtrag.s3.us-east.cloud-object-storage.appdomain.cloud/fiqa.jsonl.gz)
- [govt.jsonl.gz](https://mtrag.s3.us-east.cloud-object-storage.appdomain.cloud/govt.jsonl.gz)

### Post-processing for Timing Information

TODO

### Computing Accuracy

TODO

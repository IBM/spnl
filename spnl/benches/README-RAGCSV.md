# RAGCSV Benchmark for SPNL

Evaluates RAG answer accuracy from a CSV of evaluation queries using SPNL span queries. Each row contains a question, document fragments, and an expected answer. The benchmark executes each query, then uses a second LLM call to grade the response accuracy on a 0-100 scale.

## Overview

1. Load a CSV file with evaluation data (questions, document fragments, expected answers)
2. For each row, construct a span query that provides the document fragments as context and asks the question
3. Grade the model's response against the expected answer using a second LLM query
4. Report quantile statistics for accuracy and response time

## CSV Schema

The CSV has no headers. Columns are positional:

| Index | Content |
|-------|---------|
| 0 | Expected result |
| 1 | Fragments -- Python-style single-quoted JSON array of `{'page_content': '...', 'metadata': {'title': '...'}, ...}` |
| 2 | Hallucination detection proposal (unused) |
| 3 | Timestamp (unused) |
| 4 | Question to pose |

## Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGCSV_FILE` | (required) | Path to the CSV file |
| `RAGCSV_MODEL` | `ollama/granite3.3:2b` | Model for inference |
| `RAGCSV_GRADING_MODEL` | same as `RAGCSV_MODEL` | Model for accuracy grading |
| `RAGCSV_CONCURRENCY` | `1` | Max parallel query executions |
| `RAGCSV_LIMIT` | all rows | Limit number of rows to process |
| `RAGCSV_MAX_TOKENS` | `512` | Max tokens for primary query |
| `RAGCSV_DEBUG` | `false` | Print first row's query/response details |

## Usage Examples

### Basic Usage

```bash
RAGCSV_FILE=~/data/eval.csv cargo bench --bench ragcsv
```

### Debug Mode

```bash
RAGCSV_FILE=~/data/eval.csv RAGCSV_DEBUG=1 RAGCSV_LIMIT=3 \
  cargo bench --bench ragcsv
```

### Concurrent Execution

```bash
RAGCSV_FILE=~/data/eval.csv \
RAGCSV_MODEL="ollama/granite3.3:2b" \
RAGCSV_CONCURRENCY=4 \
RAGCSV_LIMIT=100 \
  cargo bench --bench ragcsv
```

### Separate Grading Model

```bash
RAGCSV_FILE=~/data/eval.csv \
RAGCSV_MODEL="ollama/granite3.3:2b" \
RAGCSV_GRADING_MODEL="ollama/llama3.2:3b" \
  cargo bench --bench ragcsv
```

## Output Format

Progress is shown during execution:

```
[00:01:23] 42/100 | Avg Acc=72.3% | Pass(>=75%)=28/42
```

Final report includes quantile statistics for accuracy and total time:

```
=== RAGCSV Eval Accuracy (n=100) ===
  min:  10.0%
  p25:  55.0%
  p50:  75.0%
  p75:  85.0%
  p90:  95.0%
  p99:  100.0%
  max:  100.0%
  avg:  71.2%
  pass (>=75%): 58/100

=== RAGCSV Eval Total Time (n=100) ===
  min:  234ms
  p25:  456ms
  p50:  623ms
  p75:  891ms
  p90:  1203ms
  p99:  2456ms
  max:  3012ms
  avg:  702ms
```

## Implementation Details

### Query Construction

Each row produces a span query of the form:

```
g model
    (cross
        (system "You are a helpful assistant. Answer the question based only on the provided documents.")
        (plus [user "Document: {title}\n{page_content}", ...])
        (user question)
    )
    temperature
    max_tokens
```

### Accuracy Grading

After getting the model response, a second query grades accuracy:

```
g grading_model
    (cross
        (system "You are an accuracy evaluator...")
        (user "Expected answer: ...\n\nActual answer: ...\n\nAccuracy score (0-100):")
    )
    0.0   // deterministic
    16    // small max_tokens
```

The grading model returns an integer 0-100, which is parsed from the response.

### Concurrency

Rows are dispatched as tokio tasks with a semaphore limiting parallelism to `RAGCSV_CONCURRENCY`. Results are collected via an mpsc channel.

## Future Work

- Capture TTFT (time to first token) and ITL (inter-token latency) metrics programmatically
- Support additional CSV schemas
- Add per-row result CSV export

---

Made with Bob

# Position-Independent Caching (PIC) for spnl Plus

PIC enables the mistral.rs backend to **reuse KV cache entries regardless of where they appear in a sequence**. This dramatically improves cache locality for multi-turn RAG workloads where the same documents are injected at different positions across requests.

For engine internals (deferred RoPE, block-attention masking, cache assembly, optimization decisions), see [`mistral.rs/PIC.md`](mistral.rs/PIC.md).

## Problem

Standard transformer KV caches store positionally-encoded (RoPE'd) key tensors. A cached entry for "document A at position 100" cannot be reused as "document A at position 500" because the rotary position encoding differs. This means every time a RAG document appears at a new position, the entire KV cache for that document must be recomputed from scratch.

spnl's `Plus` operator (`+`) marks input fragments as **position-independent** (commutative) -- the semantic contract is that the model's output should be invariant to the ordering of Plus blocks. PIC exploits this contract to share cached computations.

## Usage in spnl

### Query structure

Use the `Plus` operator to mark position-independent blocks in your spnl query:

```lisp
(seq
  (system "You are a helpful assistant.")
  (plus
    (user "Document 1: The capital of France is Paris...")
    (user "Document 2: Quantum computing uses qubits..."))
  (user "What is the capital of France?"))
```

The `plus` block tells spnl that "Document 1" and "Document 2" are commutative -- their order doesn't matter, and their KV cache can be reused if they appear in a different position in a subsequent request.

### JSON equivalent

```json
{
  "seq": [
    {"system": "You are a helpful assistant."},
    {"plus": [
      {"user": "Document 1: The capital of France is Paris..."},
      {"user": "Document 2: Quantum computing uses qubits..."}
    ]},
    {"user": "What is the capital of France?"}
  ]
}
```

### Multi-turn RAG example

On turn 1:

```lisp
(seq
  (system "You are a helpful assistant.")
  (plus (user "Doc A: ...") (user "Doc B: ..."))
  (user "Question about Doc A?"))
```

On turn 2, the documents appear in a different position but their cached KV entries are reused:

```lisp
(seq
  (system "You are a helpful assistant.")
  (user "Question about Doc A?")
  (assistant "Answer about Doc A...")
  (plus (user "Doc A: ...") (user "Doc C: ..."))
  (user "Follow-up question?"))
```

Doc A's KV cache from turn 1 is reused in turn 2 despite appearing at a different sequence position.

## Benchmarking

The `spnl bench pic` command measures PIC cache reuse. Output is controlled by `-o/--output` (comma-separated, default: `speedup`):

| Mode | Description |
|------|-------------|
| `speedup` | TTFT speedup ratio (no-cache vs PIC reuse) |
| `latency` | Prefix and PIC p50 latency in ms |
| `hitrate` | PIC cache hit rate |
| `iqr` | PIC interquartile range in ms |
| `json` | All TTFT data as JSON |
| `accuracy` | Plus (PIC) vs flat (causal) response comparison (token F1) |

Multiple modes can be combined, e.g. `-o speedup,accuracy`.

```sh
# Single doc length (default: 200 words)
spnl bench pic -m llama3.1:8b

# Sweep across doc sizes (xs=10w, sm=50w, m=200w, lg=500w, xl=1000w, xxl=2000w)
spnl bench pic -m llama3.1:8b -s xs,sm,m,lg,xl,xxl

# Select specific sizes
spnl bench pic -m llama3.1:8b -s sm,m,xl

# Multiple models
spnl bench pic -m llama3.2:1b,llama3.1:8b -s sm,m,xl

# Full sweep: all sizes × default model set
spnl bench pic --full

# Accuracy comparison (Plus vs flat attention)
spnl bench pic -m llama3.1:8b -o accuracy

# Both speedup and accuracy
spnl bench pic -m llama3.1:8b -s sm,m,xl -o speedup,accuracy

# Accuracy with LLM judge
spnl bench pic -m llama3.1:8b -o accuracy --grading-model llama3.2:3b
```

Build with `--features bench,metal` (or `bench,cuda`). Works with `local/` prefix models or pretty names (e.g. `llama3.2:3b`).

### Protocol

For each doc-length, the benchmark runs multiple trials. Each trial:
1. Generates fresh synthetic documents (unique per trial)
2. Sends a **no-cache** request (first time these docs are seen -- full prefill)
3. Sends N **reuse** requests with the same docs shuffled into different orders

With PIC active, reuse requests skip prefill for all Plus blocks (documents) and only compute KV for Cross tokens (system prompt + question).

### Sample results (llama3.1:8b, Metal)

```
 Doc Size     No-cache p50    Reuse p50    Speedup      Saved   Hit Rate
 ──────────── ──────────── ──────────── ────────── ────────── ──────────
 xs 10w             640 ms       272 ms     2.35x     57.4%      100%
 s 50w             1551 ms       278 ms     5.59x     82.1%      100%
 m 200w            5146 ms       292 ms    17.62x     94.3%      100%
 l 500w           13005 ms       367 ms    35.48x     97.2%      100%
 xl 1000w         30030 ms       449 ms    66.89x     98.5%      100%
 xxl 2000w        81731 ms       680 ms   120.24x     99.2%      100%
```

## Comparison with alternatives

### CacheBlend / LMCache

[CacheBlend](https://arxiv.org/pdf/2405.16444) reuses position-encoded KV cache from one context in a different position and applies a small correction. It works as an external layer without model changes but produces **approximate** results.

PIC stores un-rotated K and applies RoPE at attention time using the correct position for each token, so positional encoding is always consistent with the current layout. The tradeoff is that PIC requires model-level changes (deferred RoPE path) and re-applies RoPE to the full cached K sequence on each forward step.

### vLLM block attention

spnl's vLLM backend already supports position-independent caching via block attention at the serving layer. PIC brings the same capability to the mistral.rs backend for local/on-device inference.

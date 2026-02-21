# RAPTOR + leann-rs: Incremental HNSW Insertion Plan

## Current Approach (Two-Phase Batch Rebuild)

The spnl RAPTOR indexer uses a two-phase approach to work with leann-rs's
bulk-only `LeannBuilder`:

1. **Phase 1** -- Build the initial HNSW index from base document fragments
2. **Phase 2** -- For each base fragment, search the index for similar
   passages, generate LLM summaries, collect all summaries
3. **Phase 3** -- Rebuild the entire HNSW index from scratch with all
   original passages + all RAPTOR summaries

## Why This Is a Limitation

In the original RAPTOR implementation (and spnl's previous lancedb-based
approach), summaries were inserted into the vector database **incrementally**.
This meant:

- Summary S1 (generated from fragments A, B, C) gets inserted into the index
- When generating S2 from fragments D, E, F, the search might also find S1 as
  a relevant neighbor
- This creates a richer, more interconnected knowledge graph where summaries
  can reference and build upon other summaries

With the current batch-rebuild approach:

- All summaries are generated against the base fragments only
- No summary can influence the search results used to generate another summary
- The resulting knowledge graph is shallower (one level of summarization only)

## What leann-rs Would Need

### Option A: Append API on LeannBuilder

Add an `append_to_index()` method that takes an existing index path and adds
new passages to it:

```rust
impl LeannBuilder {
    pub fn append_to_index(
        &mut self,
        index_path: &Path,
        provider: &dyn EmbeddingProvider,
    ) -> Result<()> {
        // 1. Load existing HNSW graph
        // 2. Load existing passages
        // 3. Compute embeddings for new chunks
        // 4. Insert new nodes into the graph (incremental HNSW insertion)
        // 5. Re-write the index files
    }
}
```

### Option B: Incremental Insert on HnswGraph

Add a lower-level API for inserting individual vectors into an existing graph:

```rust
impl HnswGraph {
    pub fn insert(&mut self, vector: &[f32], level: i32) -> usize {
        // Standard HNSW insertion algorithm
        // Returns the node ID of the inserted vector
    }
}
```

This is more fundamental and would enable Option A, but also other use cases.

### Considerations

- HNSW supports incremental insertion by design (the original algorithm is
  incremental), so this is architecturally sound
- The current `build_hnsw()` function already does incremental insertion
  internally during construction; the API just doesn't expose single-node
  insertion after the initial build
- CSR (compact) format would need to be re-computed after insertions, or
  insertions would need to work on the standard format first
- The `VectorStorage` would need to support appending new vectors
- The passages JSONL file and offset index would need append support

## Performance Implications

- **Current approach**: O(N) embeddings computed twice (once for initial build,
  once for rebuild with summaries). Total embedding calls = 2N + S where S is
  the number of summaries.
- **Incremental approach**: O(N + S) embedding calls total. Each summary is
  embedded and inserted once.
- For large corpora, the rebuild cost is significant because all original
  embeddings must be recomputed during Phase 3.

### Mitigation: Pre-computed Embeddings

A partial mitigation would be to cache the embeddings from Phase 1 and use
`build_index_from_embeddings()` in Phase 3. This avoids recomputing base
embeddings but still requires a full graph rebuild. leann-rs already supports
this via `LeannBuilder::build_index_from_embeddings()`.

## Priority

Medium -- the two-phase approach is functionally correct and produces useful
RAPTOR summaries. The main loss is the depth of cross-referencing between
summaries, which may not significantly impact retrieval quality for most
document sizes. The performance overhead of double-embedding is more pressing
for large corpora.

# Span Queries - Improving KV Cache Locality

The primary goal of span queries is to provide a mechanism for
offloading critical aspects of generative AI programs from clients to
server components. In this discussion, we cover how this enables
improving KV cache locality for deep research workloads.

## Block Attention

Normally every KV-cached block in a [paged
attention](https://arxiv.org/abs/2309.06180) model serving
architecture is a) positionally encoded to reflect the block's
location in a sequential token stream; and b) "tainted" by what came
before. The cache entry for the block records not only where it is,
but also caches the matrix arithmetic necessary to attend this block's
tokens to prior tokens.  This is desirable for applications that only
need to append to an existing sequential stream of tokens.

A [recent paper](https://arxiv.org/pdf/2409) documents the power of
what they term *block attention*. In a block attention architecture,
the cached blocks can be identified as being *relocatable*. When a KV
cache block is indicated as being relocatable, the model server can
reuse a cached block no matter its position in a given token sequence.

## ABBA Microbenchmark

This microbenchmark helps to identify the potential of [block
attention](https://arxiv.org/pdf/2409), in which the blocks of a
[paged attention](https://arxiv.org/abs/2309.06180) model serving
architecture can be identified as *relocatable* blocks. The KV cache
entries of such blocks can still be used, even if used in a sequence
order that is different than originally seen. We expect either of the
following to exhibit the same cache locality, independent of the order
of presentation of the document fragments.

[<img align="right" src="/benchmarks/abba/abba-chart.svg" width="350">](/benchmarks/abba#readme)

The chart to the right shows good speedup compared to without block
attention, and this is independent of the order in which documents are
sequenced in a prompt (AB vs. BA).

[Details on the ABBA Microbenchmark](/benchmarks/abba#readme)

## Multi-turn RAG 

<img src="/docs/locality/mtrag-locality.svg">
 

# Span Queries - Improving KV Cache Locality

The primary goal of span queries is to provide a mechanism for
offloading critical aspects of generative AI programs from clients to
server components. In this discussion, we cover how this enables
improving KV cache locality for [deep
research](https://openai.com/index/introducing-deep-research/)
workloads.

## Background: Block Attention

Normally every KV cache entry in a [paged
attention](https://arxiv.org/abs/2309.06180) model serving
architecture is a) *positionally encoded* to reflect the block's
location in a sequential token stream; and b) *tainted* by what came
before. The cache entry for the block records where it is and
amortizes the cost of the matrix arithmetic necessary to attend this
block's tokens to prior tokens.

This KV cache behavior is desirable for applications, such as chat,
that only append to a stream of tokens. Here, the positions of prior
tokens do not change as new tokens are appended, which means it is
always beneficial to cache the backwards-looking matrix
computations. Caching solely based on prefix computations is less
effective for workloads with a more random access pattern; say, for
example, that the application desires to replace one interior segment
with another. That is the topic we will explore in this discussion.

A [recent paper](https://arxiv.org/pdf/2409) introduces *block
attention*. A block attention architecture allows blocks to be
identified as being *relocatable*. With a few caveats, the model
server can reuse a relocatable KV cache entry, even if the position of
the second use of the block differs from that of the first use.  For
example, if we expect either of the following to exhibit the same
cache locality, independent of the order of presentation of the
document fragments, normal paged attention does not suffice.

<img src="../../benchmarks/abba/abba-diagram.svg" width=500>

This scenario is typical in
[RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
applications that inject fragments from a corpus of online
(i.e. post-training) information in order to improve the relevancy of
generated output.

## Connection to Span Queries

With [span queries](../about.md), it is possible to express the
distinction that some input sequences are relocatable while others are
not.

## Benchmarks

To measure how well span queries work to leverage block attention, we
explore two benchmarks. First is a microbenchmark aimed to measure an
uppper bound on performance gains possible from exploiting 

### ABBA Microbenchmark

The chart to the right shows good speedup compared to without block
attention, and this is independent of the order in which documents are
sequenced in a prompt (AB vs. BA).

[Details on the ABBA Microbenchmark](/benchmarks/abba#readme)

## Multi-turn RAG 

<img src="/docs/locality/mtrag-locality.svg">
 

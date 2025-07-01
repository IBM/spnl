# Span Queries - Improving KV Cache Locality

The primary goal of span queries is to provide a mechanism for
offloading critical aspects of generative AI programs from clients to
server components. In this discussion, we cover how this enables
improving KV cache locality for deep research workloads.

## Block Attention

Normally every KV cache entry in a [paged
attention](https://arxiv.org/abs/2309.06180) model serving
architecture is a) positionally encoded to reflect the block's
location in a sequential token stream; and b) "tainted" by what came
before. The cache entry for the block records not only where it is,
but also caches the matrix arithmetic necessary to attend this block's
tokens to prior tokens.  This is desirable for applications, such as
chat, that only need to append to an existing sequential stream of
tokens.

A [recent paper](https://arxiv.org/pdf/2409) documents the power of
what they term *block attention*. In a block attention architecture,
the cached blocks can be identified as being *relocatable*. With a few
caveats, the model server can reuse a relocatable KV cache entry, even
if the position of the second use of the block differs from that of
the first use.  For example, we expect either of the following to
exhibit the same cache locality, independent of the order of
presentation of the document fragments.

<img src="../../benchmarks/abba/abba-diagram.svg" width=500>

## Connection to Span Queries

A [span query](../about.md) can encode that input sequences are
independent.

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
 

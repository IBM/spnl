# Span Queries - Improving KV Cache Locality

The primary goal of span queries is to provide a mechanism for
offloading critical aspects of generative AI programs from clients to
server components. In this discussion, we cover how this enables
improving cache locality for deep research workloads[^1]. The
[chart](/benchmarks/abba/abba-chart.svg) shows good speedup compared
to without block attention, and this is independent of the order in
which documents are sequenced in a prompt (AB vs. BA).<br/>

## ABBA Microbenchmark

This microbenchmark helps to identify the potential of [block
attention](https://arxiv.org/pdf/2409), in which the blocks of a
[paged attention](https://arxiv.org/abs/2309.06180) model serving
architecture can be identified as *relocatable* blocks. The KV cache
entries of such blocks can still be used, even if used in a sequence
order that is different than originally seen. We expect either of the
following to exhibit the same cache locality, independent of the order
of presentation of the document fragments.

[<img align="right" src="/benchmarks/abba/abba-chart.svg" width=350>](/benchmarks/abba#readme)

The chart to the right shows good speedup compared to without block
attention, and this is independent of the order in which documents are
sequenced in a prompt (AB vs. BA).

[Details](/benchmarks/abba#readme)

## Multi-run RAG 

<img src="/docs/locality/mtrag-locality.svg">

[^1]: c.f. [block attention](https://arxiv.org/pdf/2409)
  

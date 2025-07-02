# ABBA Microbenchmark

<img align="right" src="/benchmarks/abba/abba-chart.svg" width="350">

This microbenchmark helps to identify the potential of [block
attention](https://arxiv.org/pdf/2409), in which the blocks of a
[paged attention](https://arxiv.org/abs/2309.06180) model serving
architecture can be identified as *relocatable* blocks. The KV cache
entries of such blocks can still be used, even if used in a sequence
order that is different than originally seen. We expect either of the
following to exhibit the same cache locality, independent of the order
of presentation of the document fragments.

<img src="./abba-diagram.svg" width=500>

## Running the ABBA Microbenchmark

> [!WARNING]
> To see the benefits of relocatable blocks currently requires a
> branch of vLLM. Stay tuned!

First, make sure you have a running vLLM endpoint [with vLLM span
support](/docs/vllm.md). More coming soon.


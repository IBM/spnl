# Span Queries

[![arXiv](https://img.shields.io/badge/arXiv-2511.02749-b31b1b.svg?style=flat)](https://arxiv.org/abs/2511.02749)
[![CI - Core](https://github.com/IBM/spnl/actions/workflows/core.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/core.yml)
[![CI - Python](https://github.com/IBM/spnl/actions/workflows/python.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/python.yml)
[![CI - Playground](https://github.com/IBM/spnl/actions/workflows/playground.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/playground.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/spnl)](https://pypi.org/project/spnl/)
![GitHub License](https://img.shields.io/github/license/IBM/spnl)

<img align="right" src="/docs/images/nested-gen.svg" width="150">

Clients are evolving beyond chat completion, and now include a variety
of innovative inference-time scaling and deep reasoning techniques. At
the same time, inference servers remain heavily optimized for chat
completion.  [Prior work](https://arxiv.org/html/2409.15355v5) has
shown that large improvements to KV cache hit rate are possible if
inference servers evolve towards these non-chat use cases. However,
they offer solutions that are also optimized for a single use case,
RAG. We introduce the **Span Query** to generalize the interface to
the inference server.

:rocket: [Playground](https://ibm.github.io/spnl/) **|** [Judge/generator Example](https://ibm.github.io/spnl/?demo=email2&qv=true) **|** [What is a Span Query?](./docs/about.md)


## Getting Started with SPNL

SPNL is a library for manipulating span queries. The library is surfaced for consumption as:

- a [vLLM](https://github.com/vllm-project/vllm) API that can greatly
  improve KV cache locality (by as much as 20x). We have a
  pre-packaged
  [image](https://github.com/IBM/spnl/pkgs/container/spnl-llm-d-cuda)
  that includes vLLM with [llm-d](https://llm-d.ai/) and SPNL support.
- a CLI that can communicate with standard OpenAI-compatible inference
  servers, or with the optimized vLLM API. We have pre-packaged images
  that contain [just the
  CLI](https://github.com/IBM/spnl/pkgs/container/spnl) and [the CLI
  with
  Ollama](https://github.com/IBM/spnl/pkgs/container/spnl-ollama).
- an [online playground](https://ibm.github.io/spnl/?qv=false) that
lets you run queries directly in browsers that support
[WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)

To kick the tires with SPNL running [Ollama](https://ollama.com/):
```shell
podman run --rm -it ghcr.io/ibm/spnl-ollama --verbose
```

This will run a judge/generator email example. You also can point it
to a JSON file containing a [span query](./docs/about).

## Building your own SPNL CLI

First, [configure your
environment](./https://www.rust-lang.org/tools/install) for Rust.  Now
you can build the CLI with `cargo build`, which will produce
`./target/debug/spnl`. Running `cargo build --release` will produce a
build with source code optimizations, and produces
`./target/release/spnl`.

## CLI Usage

```bash
Usage: spnl [OPTIONS] [FILE]

Arguments:
  [FILE]  File to process

Options:
  -b, --builtin <BUILTIN>
          Builtin to run [env: SPNL_BUILTIN=] [possible values: bulkmap, email, email2, email3, sweagent, gsm8k, rag, spans]
  -m, --model <MODEL>
          Generative Model [env: SPNL_MODEL=] [default: ollama/granite3.3:2b]
  -e, --embedding-model <EMBEDDING_MODEL>
          Embedding Model [env: SPNL_EMBEDDING_MODEL=] [default: ollama/mxbai-embed-large:335m]
  -t, --temperature <TEMPERATURE>
          Temperature [default: 0.5]
  -l, --max-tokens <MAX_TOKENS>
          Max Completion/Generated Tokens [default: 100]
  -n, --n <N>
          Number of candidates to consider [default: 5]
  -k, --chunk-size <CHUNK_SIZE>
          Chunk size
      --vecdb-uri <VECDB_URI>
          Vector DB Url [default: data/spnl]
  -r, --reverse
          Reverse order
      --prepare
          Prepare query
  -p, --prompt <PROMPT>
          Question to pose
  -d, --document <DOCUMENT>
          Document(s) that will augment the question
  -x, --max-aug <MAX_AUG>
          Max augmentations to add to the query [env: SPNL_RAG_MAX_MATCHES=]
      --shuffle
          Randomly shuffle order of fragments
  -i, --indexer <INDEXER>
          The RAG indexing scheme [possible values: simple-embed-retrieve, raptor]
  -s, --show-query
          Re-emit the compiled query
      --time <TIME>
          Report query execution time to stderr [possible values: all, gen, gen1]
  -v, --verbose
          Verbose output
      --dry-run
          Dry run (do not execute query)?
  -h, --help
          Print help (see more with '--help')
  -V, --version
          Print version
```

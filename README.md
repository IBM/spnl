# Span Queries

[![arXiv](https://img.shields.io/badge/arXiv-2511.02749-b31b1b.svg?style=flat)](https://arxiv.org/abs/2511.02749)
[![Crates.io - Version](https://img.shields.io/crates/v/spnl)](https://crates.io/crates/spnl)
[![PyPI - Version](https://img.shields.io/pypi/v/spnl)](https://pypi.org/project/spnl)
[![CI - Core](https://github.com/IBM/spnl/actions/workflows/core.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/core.yml)
[![CI - Python](https://github.com/IBM/spnl/actions/workflows/python.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/python.yml)
[![CI - Playground](https://github.com/IBM/spnl/actions/workflows/playground.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/playground.yml)
![GitHub License](https://img.shields.io/github/license/IBM/spnl)

<img align="right" src="/docs/images/nested-gen.svg" width="150">

Use of LLM-based inference is evolving from its origins of chat. These
days, use cases involve the combination of multiple inference calls,
tool calls, and database
lookups. [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation),
[agentic AI](https://en.wikipedia.org/wiki/AI_agent), and [deep
research](https://en.wikipedia.org/wiki/ChatGPT_Deep_Research) are
three examples of these more sophisticated use cases.

The goal of this project to facilitate optimizations that drastically
reduce the cost of inference for RAG, agentics, and deep research (by
10x [^1]) without harming accuracy. Our approach is to
generalize the interface to inference servers via the **Span
Query**.

In a span query, chat is a special case of a more general
form. To the right is a visualization of a [span query for a
"judge/generator"](https://ibm.github.io/spnl/?demo=email2&qv=true) (a.k.a. "LLM-as-a-judge").

Learn more about [span query syntax and semantics](./docs/about.md)

[^1]: https://arxiv.org/html/2409.15355v5


## Getting Started with SPNL

SPNL is a library for creating, optimizing, and tokenizing span
queries. The library is surfaced for consumption as:

[**vLLM image**](https://github.com/IBM/spnl/pkgs/container/spnl-llm-d-cuda) **|** [**vLLM patch**](docker/vllm/llm-d/patches/0.4.0) **|** [**CLI image**](https://github.com/IBM/spnl/pkgs/container/spnl) **|** [**CLI image
  with  Ollama**](https://github.com/IBM/spnl/pkgs/container/spnl-ollama) **|** [**Rust crate**](https://crates.io/crates/spnl) **|** [**Python pip**](https://pypi.org/project/spnl) **|** [**Playground**](https://ibm.github.io/spnl/?qv=false)

To kick the tires with SPNL running [Ollama](https://ollama.com/):
```shell
podman run --rm -it ghcr.io/ibm/spnl-ollama --verbose
```

This will run a judge/generator email example. You also can point it
to a JSON file containing a [span query](./docs/about).

### Building SPNL

First, [configure your
environment](./https://www.rust-lang.org/tools/install) for Rust.  Now
you can build the CLI with `cargo build`, which will produce
`./target/debug/spnl`. Running `cargo build --release` will produce a
build with source code optimizations, and produces
`./target/release/spnl`.

### CLI Usage

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

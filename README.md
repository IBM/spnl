# Span Queries

[![CI - Core](https://github.com/IBM/spnl/actions/workflows/core.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/core.yml)
[![CI - Python](https://github.com/IBM/spnl/actions/workflows/python.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/python.yml)
[![CI - Playground](https://github.com/IBM/spnl/actions/workflows/playground.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/playground.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/spnl)](https://pypi.org/project/spnl/)
![GitHub License](https://img.shields.io/github/license/IBM/spnl)

:rocket: [Playground](https://ibm.github.io/spnl/) **|** [Research Poster](./docs/poster-20250529.pdf)

## What if we had a way to plan and optimize GenAI like we do for [SQL](https://en.wikipedia.org/wiki/SQL)? 

A **Span Query** is a declarative way to specify which portions of a
generative AI (GenAI) program should be **run directly on model
serving components**. As with
[SQL](https://en.wikipedia.org/wiki/SQL), this declarative structure
is safe to run on the backend and provides a clean starting point for
optimization. Also like SQL, some GenAI programs will be entirely
expressible as queries, though most will be expressed as the
programmatic interludes around the declarative queries.

A span query specifies how to *generate* new content from a
combination of *dependent* and *independent* inputs. For example, in a
RAG query, the final output depends on all of the provided input, yet
each fragment from the corpus of documents is independent of the other
fragments. [Details - Span Query](./docs/about.md)

By expressing these data dependencies, and the relationship to a
corpus, the backend can do a better job optimizing query execution.
[Details - KV Cache Locality](/docs/locality/#readme) **|** [Details -
Query Planning](./docs/query-planning.md)

We further argue that by reconsidering GenAI programs as a tree of
such generative expressions, we can achieve a generalized inference
scaling strategy. Independent elements are akin to the *map* of a
[map/reduce](https://en.wikipedia.org/wiki/MapReduce), whereas
depedendent elements are a *reduce*.

**Examples** [Judge/generator](https://pages.github.ibm.com/cloud-computer/spnl/?demo=email&qv=true) **|** [Judge/generator (optimized)](https://pages.github.ibm.com/cloud-computer/spnl/?demo=email2&qv=true) **|** [Policy-driven email generation](https://pages.github.ibm.com/cloud-computer/spnl/?demo=email3&qv=true)

> [!NOTE]
> Plans are underway for integration with
> [vLLM](https://github.com/vllm-project/vllm) and with user-facing
> libraries such as
> [PDL](https://github.com/IBM/prompt-declaration-language).

## Getting Started

To kick the tires, you can use the [online
playground](https://pages.github.ibm.com/cloud-computer/spnl/?qv=false),
or the Span Query CLI. The playground will let you run queries
directly in browsers that support
[WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
(albeit slowly).  The CLI can leverage local
[Ollama](https://ollama.com/) or any OpenAI-compatible model server
(e.g. [vLLM](https://github.com/vllm-project/vllm)). The CLI is
currently geared towards demos. Using it, you can run one of the
built-in demos, or you can point it to a JSON file containing a span
query.

### Setting up the Span Query CLI

The span query system is written in
[Rust](https://www.rust-lang.org/). This choice was made to facilitate
flexible integration with backends, CLIs, and with Python
libraries. Plus, Rust is awesome. Thus, step 1 in getting started with
the CLI is to [configure your
environment](./https://www.rust-lang.org/tools/install) for
Rust.. Step 2 is to clone this repository. Now you can run a quick
demo with:

```shell
cargo run
```

The full usage is provided via `cargo run -- --help`, which also
specifies the available demos.

```bash
Usage: spnl [OPTIONS] [FILE]

Arguments:
  [FILE]  File to process

Options:
  -d, --demo <DEMO>
          Demo to run [possible values: chat, email, email2, email3, sweagent, gsm8k, rag]
  -m, --model <MODEL>
          Generative Model [default: ollama/granite3.2:2b]
  -e, --embedding-model <EMBEDDING_MODEL>
          Embedding Model [default: ollama/mxbai-embed-large:335m]
  -t, --temperature <TEMPERATURE>
          Temperature [default: 0.5]
  -l, --max-tokens <MAX_TOKENS>
          Max Completion/Generated Tokens [default: 100]
  -n, --n <N>
          Number of candidates to consider [default: 5]
  -k, --chunk-size <CHUNK_SIZE>
          Chunk size [default: 1]
      --vecdb-uri <VECDB_URI>
          Vector DB Url [default: data/spnl]
  -s, --show-query
          Re-emit the compiled query
  -v, --verbose
          Verbose output
  -h, --help
          Print help
  -V, --version
          Print version
```

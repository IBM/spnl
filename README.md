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

<img align="right" src="/docs/spnl-cake.svg" width="230">

> [!TIP]
> Stay tuned for integration with
> [vLLM](https://github.com/vllm-project/vllm) and programming
> libraries such as
> [PDL](https://github.com/IBM/prompt-declaration-language).

A span query specifies how to *generate* (**G** in the diagram to the
right) new content from a combination of *dependent* (**x**) and
*independent* (**+**) inputs.For example, in a RAG query the final
output depends on all of the provided input, yet each fragment from
the corpus of documents is independent of the rest.  E.g. `a+b`
signifies that the token sequences `a` and `b` should be considered as
independent of each other. In this sense, a span query can be
considered an expression tree such as those visualized to the right.

By expressing these data dependencies, and the relationship to a
corpus, the backend can do a better job optimizing query execution.
<br>[More on KV Cache Locality](/docs/locality/#readme) **|** [More on Query Planning](./docs/query-planning.md)

By reconsidering GenAI programs as a tree of such generative
expressions, such as the query visualized inside the *Federation
Layer*[^1] of the diagram to the right, we may also achieve a generalized
inference scaling strategy. Independent elements are akin to the *map*
of a [map/reduce](https://en.wikipedia.org/wiki/MapReduce), whereas
depedendent elements are a *reduce*. Map/reduce is a proven way to
code scale-up and scale-out implementations.

[^1]: For example, [llm-d](https://llm-d.ai/) is a system being
    designed to federate model servers such as
    [vLLM](https://github.com/vllm-project/vllm). The llm-d system
    will route model serving requests to a gang of backend servers,
    based on availability, locality, and other constraints.

[More on Span Queries](./docs/about.md)

**Examples** [Judge/generator](https://ibm.github.io/spnl/?demo=email&qv=true) **|** [Judge/generator (optimized)](https://ibm.github.io/spnl/?demo=email2&qv=true) **|** [Policy-driven email generation](https://ibm.github.io/spnl/?demo=email3&qv=true)

## Getting Started

To kick the tires, you can use the [online
playground](https://ibm.github.io/spnl/?qv=false),
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
environment](./https://www.rust-lang.org/tools/install) for Rust. Step
2 is to clone this repository. Now you can build the CLI with:

```shell
cargo build --release
export PATH=$PWD/target/release:$PATH
```

You can now run the `spnl` CLI. If you want the build to complete more
quickly, you can drop the `--release` option. In either case, the full
usage is provided via `spnl --help`, which also specifies the available
demos.

```bash
Usage: spnl [OPTIONS] [FILE]

Arguments:
  [FILE]  File to process

Options:
  -b, --builtin <BUILTIN>
          Builtin to run [possible values: chat, email, email2, email3, sweagent, gsm8k, rag]
  -m, --model <MODEL>
          Generative Model [default: ollama/granite3.3:2b]
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
  -r, --reverse
          Reverse order
      --prepare
          Prepare query
  -p, --prompt <PROMPT>
          Question to pose
  -d, --document <DOCUMENT>
          Document that will augment the question
  -s, --show-query
          Re-emit the compiled query
      --time
          Report query execution time to stderr
  -v, --verbose
          Verbose output
  -h, --help
          Print help
  -V, --version
          Print version
```

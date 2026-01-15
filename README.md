# Span Queries

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

:scroll: [Research Paper](https://arxiv.org/abs/2511.02749) **|** :rocket: [Playground](https://ibm.github.io/spnl/) **|** [Judge/generator Example](https://ibm.github.io/spnl/?demo=email2&qv=true)

[More on Span Queries](./docs/about.md)


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

# Span Queries

[![core tests](https://github.com/IBM/spnl/actions/workflows/core.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/core.yml)
[![python tests](https://github.com/IBM/spnl/actions/workflows/python.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/python.yml)

:rocket: [Playground](https://ibm.github.io/spnl/) **|** [Research Poster](./docs/poster-20250529.pdf) **|** [About Span Queries](./docs/about.md) **|** [Contribute](./docs/dev.md)

What if we had a way to plan and optimize GenAI like we do for
[SQL](https://en.wikipedia.org/wiki/SQL)? A [Span
Query](./docs/about.md) is a declarative description of how to link
together data into a
[map/reduce](https://en.wikipedia.org/wiki/MapReduce) tree of one or
more generation calls. For example, in a RAG scenario, a span query
allows you to express that the relevant document fragments are
independent of each other. The goals of this work are to:

- improve cache locality and reduce inference latency for deep
  research (c.f. [block attention](https://arxiv.org/pdf/2409)).
- provide a generalized inference scaling strategy using the power of map/reduce.
- facilitate [query planning](./docs/query-planning.md) to improve
  generation outcomes as well as to provide a plannable basis for
  routing future requests.
- as with SQL, allow for a clean separation of concerns between
  conventional programming logic and backend interactions.

**Examples** [Judge/generator](https://pages.github.ibm.com/cloud-computer/spnl/?demo=email&qv=true) **|** [Judge/generator (optimized)](https://pages.github.ibm.com/cloud-computer/spnl/?demo=email2&qv=true) **|** [Policy-driven email generation](https://pages.github.ibm.com/cloud-computer/spnl/?demo=email3&qv=true)

> [!NOTE]
> Plans are underway for integration with model serving backends (stay tuned!) and with user-facing libraries such as [PDL](https://github.com/IBM/prompt-declaration-language). Please open an issue documenting your use cases!

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

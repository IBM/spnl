# Span Queries

[![arXiv](https://img.shields.io/badge/arXiv-2511.02749-b31b1b.svg?style=flat)](https://arxiv.org/abs/2511.02749)
[![Crates.io - Version](https://img.shields.io/crates/v/spnl)](https://crates.io/crates/spnl)
[![PyPI - Version](https://img.shields.io/pypi/v/spnl)](https://pypi.org/project/spnl)
[![CI - Core](https://github.com/IBM/spnl/actions/workflows/core.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/core.yml)
[![CI - Python](https://github.com/IBM/spnl/actions/workflows/python.yml/badge.svg)](https://github.com/IBM/spnl/actions/workflows/python.yml)
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
form. To the right is a visualization of a span query for a
"judge/generator" (a.k.a. "LLM-as-a-judge").

Learn more about [span query syntax and semantics](./docs/about.md)

[^1]: https://arxiv.org/html/2409.15355v5


## Getting Started with SPNL

SPNL is a library for creating, optimizing, and tokenizing span
queries. The library is surfaced for consumption as:

[**vLLM image**](https://github.com/IBM/spnl/pkgs/container/spnl-llm-d-cuda) **|** [**vLLM patch**](docker/vllm/llm-d/patches/0.4.0) **|** [**CLI image**](https://github.com/IBM/spnl/pkgs/container/spnl) **|** [**CLI image
  with  Ollama**](https://github.com/IBM/spnl/pkgs/container/spnl-ollama) **|** [**Rust crate**](https://crates.io/crates/spnl) **|** [**Python pip**](https://pypi.org/project/spnl)

## Using the `spnl` CLI

The `spnl` CLI provides commands for running span queries and managing vLLM deployments. For macOS users, you can install via Homebrew:

```bash
# Add the tap
brew tap IBM/spnl https://github.com/IBM/spnl

# Install the spnl CLI
brew install spnl
```

For other platforms, you can download the latest `spnl` CLI from the [SPNL releases page](https://github.com/IBM/spnl/releases/latest).

### Managing vLLM Deployments

The `spnl` CLI provides commands to easily deploy and manage vLLM inference servers on Kubernetes or Google Compute Engine. See the [vLLM documentation](./docs/vllm.md) for detailed instructions.

Quick example:
```shell
# Bring up a vLLM server on Kubernetes (requires HuggingFace token)
spnl vllm up my-deployment --target k8s --hf-token YOUR_HF_TOKEN

# Bring down the vLLM server
spnl vllm down my-deployment --target k8s
```

### Quick Start with Docker

To kick the tires with the `spnl` CLI running [Ollama](https://ollama.com/):
```shell
podman run --rm -it ghcr.io/ibm/spnl-ollama --verbose
```

This will run a judge/generator email example. You also can point it
to a JSON file containing a [span query](./docs/about).

### CLI Usage

For comprehensive CLI documentation including all commands, options, and examples, see [docs/cli.md](./docs/cli.md).

Quick reference:
```bash
# Run a query
spnl run [OPTIONS]

# Run with timing metrics (reports TTFT and ITL to stderr)
spnl run --time [OPTIONS]

# List available local models (requires 'local' feature)
spnl list

# Run with a local model using pretty names
spnl run --builtin email2 --model llama3.2:1b

# Manage vLLM deployments
spnl vllm <up|down> [OPTIONS]

# Get help
spnl --help
spnl run --help
spnl vllm --help
```

## Building SPNL

First, [configure your
environment](./https://www.rust-lang.org/tools/install) for Rust.  Now
you can build the CLI with `cargo build -p spnl-cli`, which will
produce `./target/debug/spnl`. Adding `--release` will produce a build
with source code optimizations in `./target/release/spnl`.

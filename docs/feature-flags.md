# Feature Flags of the `spnl` library for Span Queries

> [!NOTE]
> Rust supports conditional compilation via
> ["features"](https://doc.rust-lang.org/cargo/reference/features.html). The
> `spnl` library uses this to allow you to selectively compile in more
> complex, expensive, or exploratory capabilities.

On top of these core concepts is a set of feature flags that extend
what can be expressed in a span query. For example, you may wish to
have client-side support for extended features, such as reading in
messages from a filesystem or from stdin. Or you may wish to have your
server side also support fetching message content from a
filesystem. The choice is yours.

## Core Features

- **run**: Enables execution of span queries. Without this flag enabled, the compiled code will only be able to parse queries.

- **print**: Enables printing/display functionality for span queries.

- **cli_support**: Enables CLI-specific support features. Depends on `print` and includes pretty-printing with `ptree`.

## Model Backend Features

- **ollama**: Enables support for directing `g` (generate) operations to a local Ollama model server. Depends on `openai` for API compatibility.

- **openai**: Enables support for directing `g` (generate) operations to an OpenAI-compatible model server. By default, this will talk to `http://localhost:8000`, but this can be changed via the `OPENAI_BASE_URL` environment variable.

- **gemini**: Enables support for Google's Gemini API. Depends on `openai` for API compatibility.

- **local**: Enables local model inference using mistral.rs. Supports running models directly on your machine without external API calls. Depends on `run` and includes mistralrs, tokio, and related dependencies.

- **metal**: Enables Metal GPU acceleration for local inference on macOS. Depends on `local` and enables mistralrs Metal backend.

- **cuda**: Enables CUDA GPU acceleration for local inference on NVIDIA GPUs. Depends on `local` and enables mistralrs CUDA backend.

- **cuda-flash-attn**: Enables Flash Attention optimization for CUDA. Depends on `cuda`.

- **cuda-flash-attn-v3**: Enables Flash Attention v3 optimization for CUDA. Depends on `cuda` and uses mistralrs-core directly.

## RAG (Retrieval-Augmented Generation) Features

- **rag**: Enables RAG capabilities, allowing span queries to augment messages with fragments from a given set of documents. The query process handles fragmentation, indexing, embedding, and retrieval. Depends on `run` and includes LanceDB, PDF extraction, and vector operations.

- **rag-deep-debug**: Enables deep debugging output for RAG operations.

## Language & Format Features

- **lisp**: A highly experimental effort to allow for [static compilation](./lisp) of a query into a shrinkwrapped executable.

- **yaml**: Enables YAML parsing and serialization support.

## Tokenization & Python Features

- **tok**: Adds an API for both parsing and then tokenizing the messages in a query.

- **ffi**: Enables Foreign Function Interface support for calling spnl from other languages.

- **pypi**: Enables Python bindings for spnl. Depends on `ffi` and `tok`. Includes PyO3 for Python interop.

- **run_py**: Enables running span queries from Python with async support. Depends on `run`, `pypi`, and model backends.

## Cloud & Infrastructure Features

- **spnl-api**: Enables the spnl API client for communicating with spnl services.

- **vllm**: Enables support for vLLM model serving. Depends on `yaml`.

- **k8s**: Enables Kubernetes integration for deploying and managing vLLM instances. Depends on `vllm` and includes kube client libraries.

- **gce**: Enables Google Compute Engine integration for deploying and managing vLLM instances. Depends on `vllm` and includes GCP client libraries.

## Utility Features

- **openssl-vendored**: Uses a vendored (statically linked) version of OpenSSL instead of the system version. Useful for portable builds.

## Default Features

The following features are enabled by default:
- `cli_support`
- `lisp`
- `run`
- `ollama`
- `openai`
- `gemini`
- `yaml`
- `local`

This provides a full-featured experience with CLI support, multiple model backends (local and remote), and common format support.

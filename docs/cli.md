# SPNL CLI Usage

This document provides comprehensive documentation for the SPNL command-line interface.

## Main Command

```bash
spnl <COMMAND>
```

### Commands

- `run` - Run a query
- `vllm` - Bring up vLLM in a Kubernetes cluster or Google Compute Engine
- `help` - Print help message or help for a specific subcommand

### Global Options

- `-h, --help` - Print help
- `-V, --version` - Print version

---

## `spnl run` - Run a Query

Execute a span query with various configuration options.

```bash
spnl run [OPTIONS]
```

### Options

#### Input Options

- `-f, --file <FILE>` - File to process
- `-b, --builtin <BUILTIN>` - Builtin query to run (env: `SPNL_BUILTIN`)
  - Possible values: `bulkmap`, `email`, `email2`, `email3`, `sweagent`, `gsm8k`, `rag`, `spans`

#### Model Configuration

- `-m, --model <MODEL>` - Generative model to use (env: `SPNL_MODEL`)
  - Default: `ollama/granite3.3:2b`
- `-e, --embedding-model <EMBEDDING_MODEL>` - Embedding model to use (env: `SPNL_EMBEDDING_MODEL`)
  - Default: `ollama/mxbai-embed-large:335m`
- `-t, --temperature <TEMPERATURE>` - Temperature for generation
  - Default: `0.5`
- `-l, --max-tokens <MAX_TOKENS>` - Maximum completion/generated tokens
  - Default: `100`
- `-n, --n <N>` - Number of candidates to consider
  - Default: `5`

#### RAG (Retrieval-Augmented Generation) Options

- `-p, --prompt <PROMPT>` - Question to pose
- `-d, --document <DOCUMENT>` - Document(s) that will augment the question
- `-x, --max-aug <MAX_AUG>` - Maximum augmentations to add to the query (env: `SPNL_RAG_MAX_MATCHES`)
- `-i, --indexer <INDEXER>` - The RAG indexing scheme
  - `simple-embed-retrieve` - Only perform the initial embedding without any further knowledge graph formation
  - `raptor` - Use the RAPTOR algorithm (https://github.com/parthsarthi03/raptor)
- `-k, --chunk-size <CHUNK_SIZE>` - Chunk size for document processing
- `--vecdb-uri <VECDB_URI>` - Vector database URL
  - Default: `data/spnl`

#### Query Execution Options

- `-r, --reverse` - Reverse order
- `--shuffle` - Randomly shuffle order of fragments
- `--prepare` - Prepare query without executing
- `--dry-run` - Dry run (do not execute query)

#### Output Options

- `-s, --show-query` - Re-emit the compiled query
- `--time <TIME>` - Report query execution time to stderr
  - Possible values: `all`, `gen`, `gen1`
- `-v, --verbose` - Verbose output

### Examples

```bash
# Run a builtin example
spnl run --builtin email2 --verbose

# Run a query from a file
spnl run --file query.json --model ollama/granite3.3:2b

# RAG query with custom settings
spnl run --prompt "What is the main topic?" --document paper.pdf --max-aug 5 --indexer raptor
```

---

## `spnl vllm` - Manage vLLM Deployments

Deploy and manage vLLM inference servers on Kubernetes or Google Compute Engine.

```bash
spnl vllm <COMMAND>
```

### Commands

- `up` - Bring up a vLLM server
- `down` - Tear down a vLLM server
- `help` - Print help message

---

## `spnl vllm up` - Deploy vLLM Server

Deploy a vLLM inference server with a model from HuggingFace.

```bash
spnl vllm up [OPTIONS] --hf-token <HF_TOKEN> <NAME>
```

### Arguments

- `<NAME>` - Name of the deployment resource (required)

### Options

#### Platform Configuration

- `--target <TARGET>` - Target platform
  - Possible values: `k8s` (Kubernetes), `gce` (Google Compute Engine)
  - Default: `k8s`

#### Kubernetes Options

- `-n, --namespace <NAMESPACE>` - Namespace for the Kubernetes deployment

#### Model Configuration

- `-m, --model <MODEL>` - Model to serve from HuggingFace (env: `SPNL_MODEL`)
  - Default: `ibm-granite/granite-3.3-8b-instruct`
- `-t, --hf-token <HF_TOKEN>` - HuggingFace token for pulling model weights (env: `HF_TOKEN`) (required)

#### Resource Configuration

- `--gpus <GPUS>` - Number of GPUs to request
  - Default: `1`

#### Port Forwarding

- `-p, --local-port <LOCAL_PORT>` - Local port for port forwarding
  - Default: `8000`
- `-r, --remote-port <REMOTE_PORT>` - Remote port for port forwarding
  - Default: `8000`

### Google Compute Engine Requirements

When using `--target gce`, the following environment variables must be set:

- `GCP_PROJECT` or `GOOGLE_CLOUD_PROJECT` - Your GCP project ID (required)
- `GCP_SERVICE_ACCOUNT` - Service account name for the instance (required)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to your service account key file (optional, only needed if not already logged in via `gcloud auth login`)

See [GCP authentication docs](https://docs.cloud.google.com/docs/authentication/application-default-credentials#GAC) for more information.

### Examples

```bash
# Deploy on Kubernetes with default model
spnl vllm up my-deployment --target k8s --hf-token YOUR_HF_TOKEN

# Deploy with custom model and multiple GPUs
spnl vllm up my-deployment --target k8s --model meta-llama/Llama-3.1-8B-Instruct --hf-token YOUR_HF_TOKEN --gpus 2

# Deploy on Google Compute Engine
export GCP_PROJECT=my-project
export GCP_SERVICE_ACCOUNT=my-service-account
spnl vllm up my-deployment --target gce --hf-token YOUR_HF_TOKEN

# Deploy with custom ports
spnl vllm up my-deployment --target k8s --hf-token YOUR_HF_TOKEN --local-port 8080 --remote-port 8000
```

---

## `spnl vllm down` - Tear Down vLLM Server

Remove a vLLM deployment and clean up resources.

```bash
spnl vllm down [OPTIONS] <NAME>
```

### Arguments

- `<NAME>` - Name of the deployment resource to tear down (required)

### Options

- `--target <TARGET>` - Target platform
  - Possible values: `k8s` (Kubernetes), `gce` (Google Compute Engine)
  - Default: `k8s`
- `-n, --namespace <NAMESPACE>` - Namespace of the Kubernetes deployment

### Examples

```bash
# Tear down Kubernetes deployment
spnl vllm down my-deployment --target k8s

# Tear down GCE deployment
spnl vllm down my-deployment --target gce

# Tear down with specific namespace
spnl vllm down my-deployment --target k8s --namespace my-namespace
```

---

## Environment Variables

The following environment variables can be used to configure SPNL:

### Query Execution

- `SPNL_BUILTIN` - Default builtin query to run
- `SPNL_MODEL` - Default generative model
- `SPNL_EMBEDDING_MODEL` - Default embedding model
- `SPNL_RAG_MAX_MATCHES` - Default maximum RAG augmentations

### vLLM Deployment

- `HF_TOKEN` - HuggingFace token for model access
- `GCP_PROJECT` or `GOOGLE_CLOUD_PROJECT` - GCP project ID (for GCE deployments)
- `GCP_SERVICE_ACCOUNT` - GCP service account name (for GCE deployments)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account key file (for GCE deployments)

---

## Feature Gates

Some CLI options may require specific features to be enabled at compile time. To build with all features:

```bash
cargo build --all-features
```

Refer to the project's `Cargo.toml` for a complete list of available features.
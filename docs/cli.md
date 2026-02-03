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
- `--shuffle` - Randomly shuffle order of fragments

#### Query Execution Options

- `-r, --reverse` - Reverse order
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
- `image` - Manage custom images with vLLM pre-installed (GCE only)
- `patchfile` - Emit vLLM patchfile to stdout
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

#### Google Compute Engine Configuration

When using `--target gce`, the following options are available:

- `--project <PROJECT>` - GCP project ID (env: `GCP_PROJECT` or `GOOGLE_CLOUD_PROJECT`) (required)
- `--service-account <SERVICE_ACCOUNT>` - GCP service account email without @PROJECT.iam.gserviceaccount.com (env: `GCP_SERVICE_ACCOUNT`) (required)
- `--region <REGION>` - GCE region (env: `GCE_REGION`)
  - Default: `us-west1`
- `--zone <ZONE>` - GCE zone (env: `GCE_ZONE`)
  - Default: `us-west1-a`
- `--machine-type <MACHINE_TYPE>` - GCE machine type (env: `GCE_MACHINE_TYPE`)
  - Default: `g2-standard-4`
- `--gcs-bucket <GCS_BUCKET>` - GCS bucket for storing artifacts (env: `GCS_BUCKET`)
  - Default: `spnl-test`
- `--spnl-github <SPNL_GITHUB>` - SPNL GitHub repository URL for dev mode (env: `SPNL_GITHUB`)
- `--github-sha <GITHUB_SHA>` - SPNL GitHub commit SHA (env: `GITHUB_SHA`)
- `--github-ref <GITHUB_REF>` - SPNL GitHub ref (branch/tag) (env: `GITHUB_REF`)
- `--vllm-org <VLLM_ORG>` - vLLM organization on GitHub (env: `VLLM_ORG`)
  - Default: `neuralmagic`
- `--vllm-repo <VLLM_REPO>` - vLLM repository name (env: `VLLM_REPO`)
  - Default: `vllm`
- `--vllm-branch <VLLM_BRANCH>` - vLLM branch to use (env: `VLLM_BRANCH`)
  - Default: `llm-d-release-0.4`

### Google Compute Engine Requirements

When using `--target gce`, you must set:

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

# Deploy on GCE with custom configuration
spnl vllm up my-deployment --target gce \
  --hf-token YOUR_HF_TOKEN \
  --project my-gcp-project \
  --service-account my-sa \
  --region us-central1 \
  --zone us-central1-a \
  --machine-type g2-standard-8 \
  --gpus 2
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

#### Google Compute Engine Configuration

When using `--target gce`, the same GCE configuration options from `vllm up` are available.

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

## `spnl vllm image` - Manage Custom vLLM Images (GCE Only)

Create and manage custom GCE images with vLLM pre-installed.

```bash
spnl vllm image <COMMAND>
```

### Commands

- `create` - Create a custom image with vLLM pre-installed

---

## `spnl vllm image create` - Create Custom vLLM Image

Create a custom GCE image with vLLM pre-installed for faster instance startup.

```bash
spnl vllm image create [OPTIONS]
```

### Options

- `--target <TARGET>` - Target platform (only `gce` is supported)
  - Default: `gce`
- `-f, --force` - Force overwrite of existing image with the same name
- `--image-name <IMAGE_NAME>` - Custom image name (defaults to auto-generated from hash)
- `--image-family <IMAGE_FAMILY>` - Image family
  - Default: `vllm-spnl`
- `--llmd-version <LLMD_VERSION>` - LLM-D version for patch file
  - Default: `0.4.0`

#### Google Compute Engine Configuration

The same GCE configuration options from `vllm up` are available.

### Examples

```bash
# Create a custom image with default settings
spnl vllm image create --project my-project --service-account my-sa

# Create with custom image name and force overwrite
spnl vllm image create --project my-project --service-account my-sa \
  --image-name my-vllm-image --force

# Create with custom vLLM version
spnl vllm image create --project my-project --service-account my-sa \
  --vllm-branch main --llmd-version 0.5.0
```

---

## `spnl vllm patchfile` - Emit vLLM Patchfile

Output the vLLM patchfile to stdout.

```bash
spnl vllm patchfile
```

This command outputs the patchfile used to modify vLLM for SPNL integration.

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

### Google Compute Engine

- `GCP_PROJECT` or `GOOGLE_CLOUD_PROJECT` - GCP project ID (required for GCE)
- `GCP_SERVICE_ACCOUNT` - GCP service account name (required for GCE)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account key file
- `GCE_REGION` - GCE region (default: `us-west1`)
- `GCE_ZONE` - GCE zone (default: `us-west1-a`)
- `GCE_MACHINE_TYPE` - GCE machine type (default: `g2-standard-4`)
- `GCS_BUCKET` - GCS bucket for artifacts (default: `spnl-test`)
- `SPNL_GITHUB` - SPNL GitHub repository URL (for dev mode)
- `GITHUB_SHA` - SPNL GitHub commit SHA
- `GITHUB_REF` - SPNL GitHub ref (branch/tag)
- `VLLM_ORG` - vLLM organization on GitHub (default: `neuralmagic`)
- `VLLM_REPO` - vLLM repository name (default: `vllm`)
- `VLLM_BRANCH` - vLLM branch to use (default: `llm-d-release-0.4`)

---

## Feature Gates

Some CLI options may require specific features to be enabled at compile time. To build with all features:

```bash
cargo build --all-features
```

Refer to the project's `Cargo.toml` for a complete list of available features.
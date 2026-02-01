# Managing vLLM Deployments with the SPNL CLI

The SPNL CLI provides commands to easily deploy and manage vLLM inference servers on Kubernetes or Google Compute Engine.

## Quick Start

```shell
# Bring up a vLLM server on Kubernetes (requires HuggingFace token)
spnl vllm up my-deployment --target k8s --hf-token YOUR_HF_TOKEN

# Optionally specify a different model from HuggingFace (default: ibm-granite/granite-3.3-8b-instruct)
spnl vllm up my-deployment --target k8s --model meta-llama/Llama-3.1-8B-Instruct --hf-token YOUR_HF_TOKEN

# Bring down the vLLM server
spnl vllm down my-deployment --target k8s
```

The `up` command deploys a vLLM server with a model from [HuggingFace](https://huggingface.co/models) and automatically sets up port forwarding to `localhost:8000`. You can customize the number of GPUs with `--gpus` and ports with `--local-port` and `--remote-port`. The `down` command tears down the deployment.

## Google Compute Engine Deployment

For Google Compute Engine (`--target gce`), you must set the following environment variables:
- `GCP_PROJECT` or `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `GCP_SERVICE_ACCOUNT`: Service account name for the instance
- `GOOGLE_APPLICATION_CREDENTIALS` (optional): Path to your service account key file, only needed if not already logged in via `gcloud auth login` (see [GCP authentication docs](https://docs.cloud.google.com/docs/authentication/application-default-credentials#GAC))

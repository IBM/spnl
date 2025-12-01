variable "gcp_project" {
  description = "Google Cloud project name"
  type        = string
}

variable "gcp_service_account" {
  description = "Google Cloud service account name (base name, not fully qualified)"
  type        = string
}

variable "gcs_bucket" {
  description = "Google Cloud Storage bucket in which to store state"
  type        = string
  default     = "spnl-test"
}

variable "run_id" {
  description = "Run ID"
  type        = string
  default     = ""
}

variable "hf_token" {
  description = "HuggingFace token"
  type        = string
  sensitive   = true
}

variable "machine_type" {
  description = "GCE machine type"
  type        = string
  default     = "g2-standard-4"
}

variable "gce_region" {
  description = "GCE region"
  type        = string
  default     = "us-west1"
}

variable "gce_zone" {
  description = "GCE zone"
  type        = string
  default     = "us-west1-a"
}

variable "spnl_github" {
  description = "GitHub location for spnl source"
  type        = string
  default     = "https://github.com/IBM/spnl.git"
}

variable "spnl_github_sha" {
  description = "Git SHA to test"
  type        = string
  default     = ""
}

variable "spnl_github_ref" {
  description = "Git ref to test"
  type        = string
  default     = ""
}

variable "vllm_org" {
  description = "GitHub organization from which to pull vLLM source"
  type        = string
}

variable "vllm_repo" {
  description = "GitHub repository from which to pull vLLM source"
  type        = string
}

variable "vllm_branch" {
  description = "GitHub branch from which to pull vLLM source"
  type        = string
}

variable "model" {
  description = "Model to serve"
  type        = string
  default     = "ibm-granite/granite-3.3-2b-instruct"
}

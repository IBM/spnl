use clap::Args;
use serde::Serialize;

/// GCE configuration that can be set via environment variables or command-line arguments
#[derive(Debug, Clone, Args, Serialize)]
pub struct GceConfig {
    /// GCP project ID
    #[arg(
        long,
        env = "GCP_PROJECT",
        help = "GCP project ID (can also use GOOGLE_CLOUD_PROJECT)"
    )]
    pub project: Option<String>,

    /// GCP service account email (without @PROJECT.iam.gserviceaccount.com)
    #[arg(long, env = "GCP_SERVICE_ACCOUNT")]
    pub service_account: Option<String>,

    /// GCE region
    #[arg(long, env = "GCE_REGION", default_value = "us-west1")]
    pub region: String,

    /// GCE zone
    #[arg(long, env = "GCE_ZONE", default_value = "us-west1-a")]
    pub zone: String,

    /// GCE machine type
    #[arg(long, env = "GCE_MACHINE_TYPE", default_value = "g2-standard-4")]
    pub machine_type: String,

    /// GCS bucket for storing artifacts
    #[arg(long, env = "GCS_BUCKET", default_value = "spnl-test")]
    pub gcs_bucket: String,

    /// SPNL GitHub repository URL
    #[arg(
        long,
        env = "SPNL_GITHUB",
        default_value = "https://github.com/IBM/spnl.git"
    )]
    pub spnl_github: String,

    /// SPNL GitHub commit SHA
    #[arg(long, env = "GITHUB_SHA")]
    pub github_sha: Option<String>,

    /// SPNL GitHub ref (branch/tag)
    #[arg(long, env = "GITHUB_REF")]
    pub github_ref: Option<String>,

    /// vLLM organization on GitHub
    #[arg(long, env = "VLLM_ORG", default_value = "neuralmagic")]
    pub vllm_org: String,

    /// vLLM repository name
    #[arg(long, env = "VLLM_REPO", default_value = "vllm")]
    pub vllm_repo: String,

    /// vLLM branch to use
    #[arg(long, env = "VLLM_BRANCH", default_value = "llm-d-release-0.4")]
    pub vllm_branch: String,
}

impl GceConfig {
    /// Create a new GceConfig with default values
    pub fn new() -> Self {
        Self {
            project: None,
            service_account: None,
            region: "us-west1".to_string(),
            zone: "us-west1-a".to_string(),
            machine_type: "g2-standard-4".to_string(),
            gcs_bucket: "spnl-test".to_string(),
            spnl_github: "https://github.com/IBM/spnl.git".to_string(),
            github_sha: None,
            github_ref: None,
            vllm_org: "neuralmagic".to_string(),
            vllm_repo: "vllm".to_string(),
            vllm_branch: "llm-d-release-0.4".to_string(),
        }
    }

    /// Get the project ID, checking both GCP_PROJECT and GOOGLE_CLOUD_PROJECT
    pub fn get_project(&self) -> anyhow::Result<String> {
        self.project
            .clone()
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GCP_PROJECT or GOOGLE_CLOUD_PROJECT environment variable must be set"
                )
            })
    }

    /// Get the service account, returning an error if not set
    pub fn get_service_account(&self) -> anyhow::Result<String> {
        self.service_account
            .clone()
            .ok_or_else(|| anyhow::anyhow!("GCP_SERVICE_ACCOUNT environment variable must be set"))
    }

    /// Get the GitHub SHA, returning empty string if not set
    pub fn get_github_sha(&self) -> String {
        self.github_sha.clone().unwrap_or_default()
    }

    /// Get the GitHub ref, returning empty string if not set
    pub fn get_github_ref(&self) -> String {
        self.github_ref.clone().unwrap_or_default()
    }
}

impl Default for GceConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gce_config_defaults() {
        let config = GceConfig::new();
        assert_eq!(config.region, "us-west1");
        assert_eq!(config.zone, "us-west1-a");
        assert_eq!(config.machine_type, "g2-standard-4");
        assert_eq!(config.gcs_bucket, "spnl-test");
        assert_eq!(config.spnl_github, "https://github.com/IBM/spnl.git");
        assert_eq!(config.vllm_org, "neuralmagic");
        assert_eq!(config.vllm_repo, "vllm");
        assert_eq!(config.vllm_branch, "llm-d-release-0.4");
    }

    #[test]
    fn test_get_project_returns_error_when_not_set() {
        let config = GceConfig::new();
        assert!(config.get_project().is_err());
    }

    #[test]
    fn test_get_service_account_returns_error_when_not_set() {
        let config = GceConfig::new();
        assert!(config.get_service_account().is_err());
    }

    #[test]
    fn test_get_github_sha_returns_empty_when_not_set() {
        let config = GceConfig::new();
        assert_eq!(config.get_github_sha(), "");
    }

    #[test]
    fn test_get_github_ref_returns_empty_when_not_set() {
        let config = GceConfig::new();
        assert_eq!(config.get_github_ref(), "");
    }

    #[test]
    fn test_gce_config_with_values() {
        let mut config = GceConfig::new();
        config.project = Some("test-project".to_string());
        config.service_account = Some("test-sa".to_string());
        config.github_sha = Some("abc123".to_string());
        config.github_ref = Some("refs/heads/main".to_string());

        assert_eq!(config.get_project().unwrap(), "test-project");
        assert_eq!(config.get_service_account().unwrap(), "test-sa");
        assert_eq!(config.get_github_sha(), "abc123");
        assert_eq!(config.get_github_ref(), "refs/heads/main");
    }
}

// Made with Bob

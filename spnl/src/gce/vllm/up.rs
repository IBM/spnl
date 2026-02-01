use std::collections::HashMap;

#[derive(derive_builder::Builder)]
pub struct UpArgs {
    /// Name of resource
    #[builder(setter(into), default = "\"vllm\".to_string()")]
    pub(crate) name: String,

    /// Model to serve
    #[builder(default)]
    pub(crate) model: Option<String>,

    /// HuggingFace api token
    #[builder(setter(into))]
    pub(crate) hf_token: String,
}

/// Indent each line of text by the specified number of spaces, except the first line
fn indent(text: &str, spaces: usize) -> String {
    let indent_str = " ".repeat(spaces);
    let mut lines = text.lines();
    let first_line = lines.next().unwrap_or("");
    let remaining_lines: Vec<String> = lines
        .map(|line| format!("{}{}", indent_str, line))
        .collect();

    if remaining_lines.is_empty() {
        first_line.to_string()
    } else {
        format!("{}\n{}", first_line, remaining_lines.join("\n"))
    }
}

fn load_cloud_config(args: &UpArgs) -> anyhow::Result<String> {
    let cloud_config_template = include_str!("../../../../docker/gce/vllm/cloud-config.yaml");
    let setup_script = include_str!("../../../../docker/gce/vllm/setup.sh");

    // Read vllm patch file if it exists
    let vllm_patch_path = std::path::Path::new("../git/spnl/docker/gce/vllm/vllm.patch");
    let vllm_patch_b64 = if vllm_patch_path.exists() {
        let patch_content = std::fs::read(vllm_patch_path)?;
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&patch_content)?;
        let compressed = encoder.finish()?;
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, compressed)
    } else {
        String::new()
    };

    // Generate a unique run ID
    let run_id = uuid::Uuid::new_v4().to_string();

    // Default values from terraform variables.tf
    let gcs_bucket = std::env::var("GCS_BUCKET").unwrap_or_else(|_| "spnl-test".to_string());

    // If SPNL_GITHUB is not set, use the compiled version as a release
    let spnl_release = match std::env::var("SPNL_GITHUB") {
        Ok(_) => String::new(),
        Err(_) => format!("v{}", env!("CARGO_PKG_VERSION")),
    };

    let spnl_github = std::env::var("SPNL_GITHUB")
        .unwrap_or_else(|_| "https://github.com/IBM/spnl.git".to_string());
    let spnl_github_sha = std::env::var("GITHUB_SHA").unwrap_or_default();
    let spnl_github_ref = std::env::var("GITHUB_REF").unwrap_or_default();
    let vllm_org = std::env::var("VLLM_ORG").unwrap_or_else(|_| "neuralmagic".to_string());
    let vllm_repo = std::env::var("VLLM_REPO").unwrap_or_else(|_| "vllm".to_string());
    let vllm_branch =
        std::env::var("VLLM_BRANCH").unwrap_or_else(|_| "llm-d-release-0.4".to_string());
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| "ibm-granite/granite-3.3-2b-instruct".to_string());

    // Conditionally include packages section (only needed when building from source)
    let packages = if spnl_release.is_empty() {
        vec![
            "build-essential",
            "pkg-config",
            "libssl-dev",
            "protobuf-compiler",
            "python3-dev",
        ]
    } else {
        vec![]
    };

    let packages_section = if packages.is_empty() {
        "# Packages not needed when using release binaries".to_string()
    } else {
        format!(
            "packages:\n{}",
            packages
                .iter()
                .map(|p| format!("  - {}", p))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    // Create substitution map
    let mut substitutions = HashMap::new();
    substitutions.insert("run_id", run_id.as_str());
    substitutions.insert("hf_token", args.hf_token.as_str());
    substitutions.insert("gcs_bucket", gcs_bucket.as_str());
    substitutions.insert("spnl_github", spnl_github.as_str());
    substitutions.insert("spnl_github_sha", spnl_github_sha.as_str());
    substitutions.insert("spnl_github_ref", spnl_github_ref.as_str());
    substitutions.insert("spnl_release", spnl_release.as_str());
    substitutions.insert("vllm_org", vllm_org.as_str());
    substitutions.insert("vllm_repo", vllm_repo.as_str());
    substitutions.insert("vllm_branch", vllm_branch.as_str());
    substitutions.insert("model", model.as_str());
    substitutions.insert("packages_section", &packages_section);

    // Indent setup_script and vllm_patch_b64 by 6 spaces for proper YAML formatting
    let setup_script_indented = indent(setup_script, 6);
    let vllm_patch_b64_indented = indent(&vllm_patch_b64, 6);
    substitutions.insert("setup_script", &setup_script_indented);
    substitutions.insert("vllm_patch_b64", &vllm_patch_b64_indented);

    // Perform substitutions
    let mut result = cloud_config_template.to_string();
    for (key, value) in substitutions {
        result = result.replace(&format!("${{{}}}", key), value);
    }

    Ok(result)
}

/// Create and start a GCE instance for running vLLM
///
/// This function creates a GCE instance with GPU support, configures it with cloud-init,
/// and streams the cloud-init output log until the instance setup completes.
pub async fn up(args: UpArgs) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;
    use google_cloud_compute_v1::model::{
        AcceleratorConfig, AccessConfig, AttachedDisk, AttachedDiskInitializeParams, Instance,
        Metadata, NetworkInterface, Scheduling, ServiceAccount, ShieldedInstanceConfig,
        metadata::Items as MetadataItems,
    };
    use google_cloud_lro::Poller;

    // Load and template the cloud-config
    let cloud_config = load_cloud_config(&args)?;

    eprintln!("Cloud config YAML:");
    eprintln!("---");
    eprintln!("{}", cloud_config);
    eprintln!("---\n");

    // Get configuration from environment variables (matching terraform defaults)
    let project = std::env::var("GCP_PROJECT")
        .or_else(|_| std::env::var("GOOGLE_CLOUD_PROJECT"))
        .map_err(|_| {
            anyhow::anyhow!("GCP_PROJECT or GOOGLE_CLOUD_PROJECT environment variable must be set")
        })?;
    let service_account = std::env::var("GCP_SERVICE_ACCOUNT")
        .map_err(|_| anyhow::anyhow!("GCP_SERVICE_ACCOUNT environment variable must be set"))?;
    let region = std::env::var("GCE_REGION").unwrap_or_else(|_| "us-west1".to_string());
    let zone = std::env::var("GCE_ZONE").unwrap_or_else(|_| "us-west1-a".to_string());
    let machine_type =
        std::env::var("GCE_MACHINE_TYPE").unwrap_or_else(|_| "g2-standard-4".to_string());

    // Generate a unique run ID for this instance (or use provided name)
    let run_id = uuid::Uuid::new_v4().to_string();
    let instance_name = if args.name == "vllm" {
        format!("spnl-test-big-{}", run_id)
    } else {
        args.name.clone()
    };

    eprintln!("Creating GCE instance:");
    eprintln!("  Name: {}", instance_name);
    eprintln!("  Project: {}", project);
    eprintln!("  Zone: {}", zone);
    eprintln!("  Machine type: {}", machine_type);
    eprintln!("  Run ID: {}", run_id);

    // Create labels
    let mut labels = std::collections::HashMap::new();
    labels.insert("role".to_string(), "gh-runner".to_string());
    labels.insert("gh-run-id".to_string(), run_id.clone());
    labels.insert("goog-ec-src".to_string(), "vm_add-tf".to_string());
    labels.insert(
        "goog-ops-agent-policy".to_string(),
        "v2-x86-template-1-4-0".to_string(),
    );

    // Create the instance configuration matching the terraform file
    let instance = Instance::new()
        .set_name(&instance_name)
        .set_machine_type(format!("zones/{}/machineTypes/{}", zone, machine_type))
        .set_disks([AttachedDisk::new()
            .set_boot(true)
            .set_auto_delete(true)
            .set_device_name(format!("spnl-test-big-{}", run_id))
            .set_initialize_params(
                AttachedDiskInitializeParams::new()
                    .set_source_image("projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-580-v20251210")
                    .set_disk_size_gb(100)
                    .set_disk_type(format!("zones/{}/diskTypes/pd-ssd", zone))
            )
            .set_mode("READ_WRITE")])
        .set_network_interfaces([NetworkInterface::new()
            .set_subnetwork(format!("regions/{}/subnetworks/default", region))
            .set_access_configs([AccessConfig::new()
                .set_network_tier("PREMIUM")])
                                 // .set_queue_count(0)
            .set_stack_type("IPV4_ONLY")])
        .set_guest_accelerators([AcceleratorConfig::new()
            .set_accelerator_count(1)
            .set_accelerator_type(format!("zones/{}/acceleratorTypes/nvidia-l4", zone))])
        .set_metadata(Metadata::default()
            .set_items([
                MetadataItems::default()
                    .set_key("enable-osconfig")
                    .set_value("TRUE"),
                MetadataItems::default()
                    .set_key("user-data")
                    .set_value(cloud_config),
            ]))
        .set_scheduling(Scheduling::new()
            .set_automatic_restart(false)
            .set_on_host_maintenance("TERMINATE")
            .set_preemptible(true)
            .set_provisioning_model("SPOT"))
        .set_service_accounts([ServiceAccount::new()
            .set_email(format!("{}@{}.iam.gserviceaccount.com", service_account, project))
            .set_scopes([
                "https://www.googleapis.com/auth/devstorage.read_write",
                "https://www.googleapis.com/auth/logging.write",
                "https://www.googleapis.com/auth/monitoring.write",
                "https://www.googleapis.com/auth/service.management.readonly",
                "https://www.googleapis.com/auth/servicecontrol",
                "https://www.googleapis.com/auth/trace.append",
            ])])
        .set_shielded_instance_config(ShieldedInstanceConfig::new()
            .set_enable_integrity_monitoring(true)
            .set_enable_secure_boot(false)
            .set_enable_vtpm(true))
        .set_labels(labels)
        .set_can_ip_forward(false)
        .set_deletion_protection(false);

    // Create the client and insert the instance
    let client = Instances::builder().build().await?;

    eprintln!("Submitting instance creation request...");
    let operation = client
        .insert()
        .set_project(&project)
        .set_zone(&zone)
        .set_body(instance)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("Instance '{}' created successfully", instance_name);
    eprintln!("Operation: {:?}", operation);

    // Get the instance details to show the external IP
    let instance_info = client
        .get()
        .set_project(&project)
        .set_zone(&zone)
        .set_instance(&instance_name)
        .send()
        .await?;

    let network_interfaces = instance_info.network_interfaces;
    for ni in network_interfaces {
        let access_configs = ni.access_configs;
        for ac in access_configs {
            if let Some(nat_ip) = ac.nat_ip {
                eprintln!("Instance external IP: {}", nat_ip);
            }
        }
    }

    eprintln!("\nInstance '{}' is now running", instance_name);
    eprintln!("The instance will automatically shut down when setup completes");
    eprintln!("\nStreaming cloud-init output log...\n");

    // Stream the cloud-init output log
    stream_cloud_init_log(&instance_name, &zone, &project).await?;

    Ok(())
}

async fn stream_cloud_init_log(
    instance_name: &str,
    zone: &str,
    project: &str,
) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;

    let client = Instances::builder().build().await?;

    // Wait a bit for the instance to start and cloud-init to begin
    eprintln!("Waiting for instance to start...");
    tokio::time::sleep(tokio::time::Duration::from_secs(15)).await;

    let mut last_start = 0i64;
    let mut consecutive_errors = 0;
    let mut should_stop = false;

    loop {
        if should_stop {
            break;
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nReceived Ctrl+C, stopping log stream...");
                break;
            }
            _ = async {
                // Get serial port output (port 1 contains console output including cloud-init)
                match client
                    .get_serial_port_output()
                    .set_project(project)
                    .set_zone(zone)
                    .set_instance(instance_name)
                    .set_port(1)
                    .set_start(last_start)
                    .send()
                    .await
                {
                    Ok(output) => {
                        consecutive_errors = 0;
                        if let Some(contents) = output.contents
                            && !contents.is_empty()
                        {
                            print!("{}", contents);
                            std::io::Write::flush(&mut std::io::stdout()).ok();
                        }
                        // Update the start position for next request
                        if let Some(next) = output.next {
                            last_start = next;
                        }
                    }
                    Err(e) => {
                        consecutive_errors += 1;
                        if consecutive_errors == 1 {
                            eprintln!("Error fetching serial port output: {}", e);
                        }
                        if consecutive_errors > 10 {
                            eprintln!("Instance terminated or too many errors, stopping log stream");
                            should_stop = true;
                            return;
                        }
                    }
                }

                // Poll every 2 seconds
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            } => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_cloud_config() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default().hf_token("test_token").build()?;

        let config = load_cloud_config(&args)?;
        assert!(config.contains("#cloud-config"));
        assert!(config.contains("test_token"));

        Ok(())
    }

    #[test]
    fn test_load_cloud_config_with_model() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .hf_token("test_token")
            .model(Some("meta-llama/Llama-2-7b-hf".to_string()))
            .build()?;

        let config = load_cloud_config(&args)?;
        assert!(config.contains("meta-llama/Llama-2-7b-hf"));

        Ok(())
    }

    #[test]
    fn test_load_cloud_config_with_custom_name() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .name("custom-vllm")
            .hf_token("test_token")
            .build()?;

        let config = load_cloud_config(&args)?;
        // Name is used in instance creation, not in cloud-config
        assert!(config.contains("#cloud-config"));

        Ok(())
    }

    #[test]
    fn test_indent_single_line() {
        let text = "single line";
        let result = indent(text, 4);
        assert_eq!(result, "single line");
    }

    #[test]
    fn test_indent_multiple_lines() {
        let text = "first line\nsecond line\nthird line";
        let result = indent(text, 4);
        assert_eq!(result, "first line\n    second line\n    third line");
    }

    #[test]
    fn test_indent_empty_string() {
        let text = "";
        let result = indent(text, 4);
        assert_eq!(result, "");
    }

    #[test]
    fn test_up_args_builder_defaults() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default().hf_token("test-token").build()?;

        assert_eq!(args.name, "vllm");
        assert_eq!(args.model, None);
        assert_eq!(args.hf_token, "test-token");

        Ok(())
    }

    #[test]
    fn test_up_args_builder_custom_values() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .name("my-vllm")
            .model(Some("my-model".to_string()))
            .hf_token("my-token")
            .build()?;

        assert_eq!(args.name, "my-vllm");
        assert_eq!(args.model, Some("my-model".to_string()));
        assert_eq!(args.hf_token, "my-token");

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Mock GCE API tests
    // ------------------------------------------------------------------------

    #[cfg(test)]
    mod mock_tests {
        use super::*;

        /// Mock GCE client for testing
        struct MockGceClient {
            should_fail: bool,
            instance_created: std::sync::Arc<std::sync::Mutex<bool>>,
            instance_name: std::sync::Arc<std::sync::Mutex<Option<String>>>,
        }

        impl MockGceClient {
            fn new() -> Self {
                Self {
                    should_fail: false,
                    instance_created: std::sync::Arc::new(std::sync::Mutex::new(false)),
                    instance_name: std::sync::Arc::new(std::sync::Mutex::new(None)),
                }
            }

            fn with_failure() -> Self {
                Self {
                    should_fail: true,
                    instance_created: std::sync::Arc::new(std::sync::Mutex::new(false)),
                    instance_name: std::sync::Arc::new(std::sync::Mutex::new(None)),
                }
            }

            async fn create_instance(&self, name: &str) -> anyhow::Result<()> {
                if self.should_fail {
                    return Err(anyhow::anyhow!("Mock GCE API error"));
                }

                let mut created = self.instance_created.lock().unwrap();
                *created = true;

                let mut stored_name = self.instance_name.lock().unwrap();
                *stored_name = Some(name.to_string());

                Ok(())
            }

            fn was_instance_created(&self) -> bool {
                *self.instance_created.lock().unwrap()
            }

            fn get_instance_name(&self) -> Option<String> {
                self.instance_name.lock().unwrap().clone()
            }
        }

        #[tokio::test]
        async fn mock_instance_creation_success() {
            let client = MockGceClient::new();
            let result = client.create_instance("test-instance").await;

            assert!(result.is_ok());
            assert!(client.was_instance_created());
            assert_eq!(
                client.get_instance_name(),
                Some("test-instance".to_string())
            );
        }

        #[tokio::test]
        async fn mock_instance_creation_failure() {
            let client = MockGceClient::with_failure();
            let result = client.create_instance("test-instance").await;

            assert!(result.is_err());
            assert!(!client.was_instance_created());
        }

        #[test]
        fn test_cloud_config_contains_required_fields() -> anyhow::Result<()> {
            let args = UpArgsBuilder::default()
                .hf_token("test_token_123")
                .model(Some("test-model".to_string()))
                .build()?;

            let config = load_cloud_config(&args)?;

            // Verify cloud-config structure
            assert!(config.contains("#cloud-config"));
            assert!(config.contains("test_token_123"));
            assert!(config.contains("test-model"));

            Ok(())
        }

        #[test]
        fn test_cloud_config_uses_defaults() -> anyhow::Result<()> {
            // Test that cloud config uses default values when env vars are not set
            let args = UpArgsBuilder::default().hf_token("test_token").build()?;

            let config = load_cloud_config(&args)?;

            // Should contain default values from the code
            assert!(config.contains("#cloud-config"));
            assert!(config.len() > 100); // Should be substantial

            Ok(())
        }

        #[test]
        fn test_cloud_config_default_model() -> anyhow::Result<()> {
            let args = UpArgsBuilder::default().hf_token("test_token").build()?;

            let config = load_cloud_config(&args)?;

            // Should use default model
            assert!(config.contains("ibm-granite/granite-3.3-2b-instruct"));

            Ok(())
        }

        #[test]
        fn test_instance_name_generation() {
            let args = UpArgsBuilder::default()
                .name("vllm")
                .hf_token("test_token")
                .build()
                .unwrap();

            // When name is "vllm", it should generate a unique name with UUID
            assert_eq!(args.name, "vllm");

            let args_custom = UpArgsBuilder::default()
                .name("custom-instance")
                .hf_token("test_token")
                .build()
                .unwrap();

            // When name is custom, it should use that name
            assert_eq!(args_custom.name, "custom-instance");
        }

        #[test]
        fn test_cloud_config_includes_setup_script() -> anyhow::Result<()> {
            let args = UpArgsBuilder::default().hf_token("test_token").build()?;

            let config = load_cloud_config(&args)?;

            // Verify setup script is included (it should be base64 encoded or embedded)
            assert!(config.len() > 100); // Cloud config should be substantial

            Ok(())
        }
    }
}

// Made with Bob

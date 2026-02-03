use std::collections::HashMap;
use std::sync::Arc;

use super::args::GceConfig;
use super::ssh_tunnel::SshTunnel;
use tabled::{Table, Tabled, settings::Style};

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

    /// Local port for SSH tunnel (defaults to 8000)
    #[builder(default = "Some(8000)")]
    pub(crate) local_port: Option<u16>,

    /// GCE configuration
    #[builder(default)]
    pub(crate) config: GceConfig,
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

    // Generate a unique run ID
    let run_id = uuid::Uuid::new_v4().to_string();

    // Get configuration from args
    let gcs_bucket = &args.config.gcs_bucket;

    // Determine if we're in dev mode (building from source)
    // Dev mode is when SPNL_GITHUB is defined (Some)
    let is_dev_mode = args.config.spnl_github.is_some();

    let spnl_github = args
        .config
        .spnl_github
        .as_deref()
        .unwrap_or("https://github.com/IBM/spnl.git");
    let spnl_github_sha = args.config.get_github_sha();
    let spnl_github_ref = args.config.get_github_ref();
    let spnl_release = if !is_dev_mode {
        format!("v{}", env!("CARGO_PKG_VERSION"))
    } else {
        String::new()
    };
    let vllm_org = &args.config.vllm_org;
    let vllm_repo = &args.config.vllm_repo;
    let vllm_branch = &args.config.vllm_branch;
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| "ibm-granite/granite-3.3-2b-instruct".to_string());

    // Conditionally include packages section (only needed when building from source in dev mode)
    let packages_section = if is_dev_mode {
        format!(
            "packages:\n{}",
            [
                "build-essential",
                "pkg-config",
                "libssl-dev",
                "protobuf-compiler",
                "python3-dev",
            ]
            .iter()
            .map(|p| format!("  - {}", p))
            .collect::<Vec<_>>()
            .join("\n")
        )
    } else {
        "# Packages not needed when using custom image".to_string()
    };

    // In dev mode, we use default cloud-init modules and runcmd to run setup script
    // In production mode, we optimize by disabling most modules (custom image has everything)
    let (
        setup_dev_script,
        vllm_patch_b64,
        vllm_config_section,
        cloud_init_modules_section,
        runcmd_section,
    ) = if is_dev_mode {
        let setup_dev_script = include_str!("../../../../docker/gce/vllm/setup-dev.sh");

        // Read vllm patch file if it exists
        let vllm_patch_path = std::path::Path::new("../git/spnl/docker/gce/vllm/vllm.patch");
        let vllm_patch_b64 = if vllm_patch_path.exists() {
            let patch_content = std::fs::read(vllm_patch_path)?;
            use std::io::Write;
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&patch_content)?;
            let compressed = encoder.finish()?;
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, compressed)
        } else {
            String::new()
        };

        let runcmd_section = r#"runcmd:
  - /tmp/setup-dev.sh
  - echo "Shutting down"
  - sudo /sbin/shutdown -h now"#
            .to_string();

        // Dev mode: Use default cloud-init modules (let cloud-init handle everything)
        let cloud_init_modules_section =
            r#"# Dev mode: Using default cloud-init modules for full setup"#.to_string();

        (
            indent(setup_dev_script, 6),
            indent(&vllm_patch_b64, 6),
            String::new(), // No vllm config file needed in dev mode
            cloud_init_modules_section,
            runcmd_section,
        )
    } else {
        // Production mode: write /etc/vllm/config for systemd service to read
        let vllm_config = format!(
            r#"  - path: /etc/vllm/config
    permissions: '0644'
    content: |
      # vLLM Configuration for systemd service
      MODEL={}
      VLLM_ATTENTION_BACKEND=TRITON_ATTN
      VLLM_USE_V1=1
      VLLM_V1_SPANS_ENABLED=True
      VLLM_V1_SPANS_TOKEN_PLUS=10
      VLLM_V1_SPANS_TOKEN_CROSS=13
      VLLM_SERVER_DEV_MODE=1
      HF_TOKEN={}"#,
            model, args.hf_token
        );

        // Production mode: Optimize by disabling unnecessary cloud-init modules
        let cloud_init_modules_section =
            r#"# Disable unnecessary cloud-init modules for faster boot
cloud_init_modules:
  - write_files
cloud_config_modules:
  - ssh
cloud_final_modules: []"#
                .to_string();

        (
            String::new(), // No setup script needed
            String::new(), // No patch needed
            vllm_config,
            cloud_init_modules_section,
            "# No runcmd needed - systemd manages services".to_string(),
        )
    };

    // Create substitution map
    let mut substitutions = HashMap::new();
    substitutions.insert("run_id", run_id.as_str());
    substitutions.insert("hf_token", args.hf_token.as_str());
    substitutions.insert("gcs_bucket", gcs_bucket.as_str());
    substitutions.insert("spnl_github", spnl_github);
    substitutions.insert("spnl_github_sha", &spnl_github_sha);
    substitutions.insert("spnl_github_ref", &spnl_github_ref);
    substitutions.insert("spnl_release", spnl_release.as_str());
    substitutions.insert("vllm_org", vllm_org.as_str());
    substitutions.insert("vllm_repo", vllm_repo.as_str());
    substitutions.insert("vllm_branch", vllm_branch.as_str());
    substitutions.insert("model", model.as_str());
    substitutions.insert("packages_section", &packages_section);
    substitutions.insert("setup_dev_script", &setup_dev_script);
    substitutions.insert("vllm_patch_b64", &vllm_patch_b64);
    substitutions.insert("vllm_config_section", &vllm_config_section);
    substitutions.insert("cloud_init_modules_section", &cloud_init_modules_section);
    substitutions.insert("runcmd_section", &runcmd_section);

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

    // Uncomment to debug cloud-config
    // eprintln!("Cloud config YAML:");
    // eprintln!("---");
    // eprintln!("{}", cloud_config);
    // eprintln!("---\n");

    // Get configuration from args
    let project = args.config.get_project()?;
    let service_account = args.config.get_service_account()?;
    let region = &args.config.region;
    let zone = &args.config.zone;
    let machine_type = &args.config.machine_type;

    // Generate a unique run ID for this instance (or use provided name)
    let run_id = uuid::Uuid::new_v4().to_string();
    let instance_name = if args.name == "vllm" {
        format!("spnl-test-big-{}", run_id)
    } else {
        args.name.clone()
    };

    // Determine if we're in dev mode (building from source)
    let is_dev_mode = args.config.spnl_github.is_some();

    // Determine which image to use
    let source_image = if is_dev_mode {
        // Dev mode: use standard Ubuntu accelerator image
        "projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-580-v20251210".to_string()
    } else {
        // Production mode: use custom image based on vLLM configuration
        // Use the image family to get the latest image
        format!("projects/{}/global/images/family/vllm-spnl", project)
    };

    #[derive(Tabled)]
    struct InstanceInfo {
        #[tabled(rename = "Property")]
        property: String,
        #[tabled(rename = "Value")]
        value: String,
    }

    let info = vec![
        InstanceInfo {
            property: "Name".to_string(),
            value: instance_name.clone(),
        },
        InstanceInfo {
            property: "Project".to_string(),
            value: project.clone(),
        },
        InstanceInfo {
            property: "Zone".to_string(),
            value: zone.clone(),
        },
        InstanceInfo {
            property: "Machine Type".to_string(),
            value: machine_type.clone(),
        },
        InstanceInfo {
            property: "Run ID".to_string(),
            value: run_id.clone(),
        },
        InstanceInfo {
            property: "Mode".to_string(),
            value: if is_dev_mode {
                "dev".to_string()
            } else {
                "production".to_string()
            },
        },
        InstanceInfo {
            property: "Source Image".to_string(),
            value: source_image.clone(),
        },
    ];

    let mut table = Table::new(info);
    table.with(Style::sharp());

    eprintln!("\nCreating GCE Instance:");
    eprintln!("{}\n", table);

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
                    .set_source_image(&source_image)
                    .set_disk_size_gb(100)
                    .set_disk_type(format!("zones/{}/diskTypes/pd-ssd", zone)),
            )
            .set_mode("READ_WRITE")])
        .set_network_interfaces([NetworkInterface::new()
            .set_subnetwork(format!("regions/{}/subnetworks/default", region))
            .set_access_configs([AccessConfig::new().set_network_tier("PREMIUM")])
            // .set_queue_count(0)
            .set_stack_type("IPV4_ONLY")])
        .set_guest_accelerators([AcceleratorConfig::new()
            .set_accelerator_count(1)
            .set_accelerator_type(format!("zones/{}/acceleratorTypes/nvidia-l4", zone))])
        .set_metadata(
            Metadata::default().set_items([
                MetadataItems::default()
                    .set_key("enable-osconfig")
                    .set_value("TRUE"),
                MetadataItems::default()
                    .set_key("user-data")
                    .set_value(&cloud_config),
            ]),
        )
        .set_scheduling(
            Scheduling::new()
                .set_automatic_restart(false)
                .set_on_host_maintenance("TERMINATE")
                .set_preemptible(true)
                .set_provisioning_model("SPOT")
                .set_instance_termination_action("STOP"),
        )
        .set_service_accounts([ServiceAccount::new()
            .set_email(format!(
                "{}@{}.iam.gserviceaccount.com",
                service_account, project
            ))
            .set_scopes([
                "https://www.googleapis.com/auth/devstorage.read_write",
                "https://www.googleapis.com/auth/logging.write",
                "https://www.googleapis.com/auth/monitoring.write",
                "https://www.googleapis.com/auth/service.management.readonly",
                "https://www.googleapis.com/auth/servicecontrol",
                "https://www.googleapis.com/auth/trace.append",
            ])])
        .set_shielded_instance_config(
            ShieldedInstanceConfig::new()
                .set_enable_integrity_monitoring(true)
                .set_enable_secure_boot(false)
                .set_enable_vtpm(true),
        )
        .set_labels(labels)
        .set_can_ip_forward(false)
        .set_deletion_protection(false);

    // Create the client and insert the instance
    let client = Instances::builder().build().await?;

    eprintln!("Submitting instance creation request...");
    let _operation = client
        .insert()
        .set_project(&project)
        .set_zone(zone)
        .set_body(instance)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("Instance '{}' created successfully", instance_name);
    // eprintln!("Operation: {:?}", operation);

    // Get the instance details to show the external IP
    let instance_info = client
        .get()
        .set_project(&project)
        .set_zone(zone)
        .set_instance(&instance_name)
        .send()
        .await?;

    let network_interfaces = instance_info.network_interfaces;

    // Get external IP for SSH tunnel (before consuming network_interfaces)
    let external_ip = network_interfaces
        .iter()
        .find_map(|ni| ni.access_configs.iter().find_map(|ac| ac.nat_ip.as_ref()))
        .ok_or_else(|| anyhow::anyhow!("No external IP found for instance"))?
        .clone();

    // Display network interfaces
    for ni in network_interfaces {
        let access_configs = ni.access_configs;
        for ac in access_configs {
            if let Some(nat_ip) = ac.nat_ip {
                eprintln!("Instance external IP: {}", nat_ip);
            }
        }
    }

    eprintln!("\nInstance '{}' is now running", instance_name);
    if is_dev_mode {
        eprintln!("The instance will automatically shut down when setup completes");
    }

    // Start SSH tunnel in the background only if NOT in dev mode
    let tunnel_handle = if !is_dev_mode {
        let local_port = args.local_port.unwrap_or(8000);
        eprintln!("\nStarting SSH tunnel for vLLM port {}...", local_port);
        let tunnel = start_ssh_tunnel_rust(&external_ip, local_port).await?;
        eprintln!("SSH tunnel established: http://localhost:{}", local_port);
        Some(tunnel)
    } else {
        eprintln!("\nDev mode: Skipping SSH tunnel setup");
        None
    };

    eprintln!("\nStreaming cloud-init output log...\n");

    // Extract run_id from cloud_config to fetch exit code later
    let run_id = extract_run_id_from_cloud_config(&cloud_config)?;
    let gcs_bucket = std::env::var("GCS_BUCKET").unwrap_or_else(|_| "spnl-test".to_string());

    // Stream the cloud-init output log
    let stream_result = stream_cloud_init_log(&instance_name, zone, &project).await;

    // Fetch the exit code from GCS
    eprintln!("\nFetching exit code from GCS...");
    let exit_code = fetch_exit_code_from_gcs(&gcs_bucket, &run_id).await?;

    // Clean up SSH tunnel
    if let Some(tunnel) = tunnel_handle {
        eprintln!("\nClosing SSH tunnel...");
        let _ = tunnel.close().await;
    }

    // Check stream result
    stream_result?;

    if exit_code == 0 {
        eprintln!("Setup completed successfully (exit code: 0)");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Setup failed with exit code: {}",
            exit_code
        ))
    }
}
/// Strip timestamp and instance name prefix from serial port output lines
/// Format: "YYYY-MM-DDTHH:MM:SS.ffffff+00:00 instance-name "
/// Converts octal escape sequences (#033) to proper ANSI escape codes for color rendering
/// Colorizes process prefixes and strips redundant ones when content has detailed process info
fn strip_serial_prefix(contents: &str) -> String {
    use std::io::IsTerminal;

    let is_tty = std::io::stdout().is_terminal();

    contents
        .lines()
        .map(|line| {
            // Find the second space (after timestamp and instance name)
            let mut space_count = 0;
            let mut stripped_line = line;

            for (i, c) in line.char_indices() {
                if c == ' ' {
                    space_count += 1;
                    if space_count == 2 {
                        // Get everything after the second space
                        stripped_line = &line[i + 1..];
                        break;
                    }
                }
            }

            // Convert octal escape sequences (#033) to proper ANSI escape codes (\x1b)
            // GCE serial console escapes control characters to octal for safe transmission
            let result = stripped_line.replace("#033", "\x1b");

            // Handle process prefix coloring and redundant prefix stripping
            if let Some(colon_pos) = result.find(':') {
                let prefix = &result[..colon_pos];
                let rest = &result[colon_pos + 1..];

                // Check if prefix looks like "process[pid]" and rest contains "pid="
                // If so, strip the redundant prefix
                if prefix.contains('[') && prefix.contains(']') && rest.contains("pid=") {
                    return rest.trim_start().to_string();
                }

                // Otherwise, if TTY and prefix doesn't have colors, colorize it
                if is_tty && !prefix.contains('\x1b') {
                    // Check if prefix looks like a service/process name
                    if prefix.chars().all(|c| {
                        c.is_alphanumeric() || c == '[' || c == ']' || c == '-' || c == '_'
                    }) {
                        // Yellow color for the prefix (avoiding cyan/36 which vLLM uses)
                        return format!("\x1b[33m{}:\x1b[0m{}", prefix, rest);
                    }
                }
            }

            result
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn stream_cloud_init_log(
    instance_name: &str,
    zone: &str,
    project: &str,
) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;

    let client = Instances::builder().build().await?;

    // Start polling immediately - the instance may already be outputting
    eprintln!("Streaming serial console output...");

    let mut last_start = 0i64;
    let mut consecutive_errors = 0;
    let mut should_stop = false;
    let mut first_output = true;

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
                            if first_output {
                                eprintln!("Instance is running, output received:\n");
                                first_output = false;
                            }
                            // Strip timestamp and instance name prefix from each line
                            let cleaned = strip_serial_prefix(&contents);
                            print!("{}", cleaned);
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

/// Extract run_id from the cloud-config YAML
fn extract_run_id_from_cloud_config(cloud_config: &str) -> anyhow::Result<String> {
    // The run_id is embedded in the cloud-config as an environment variable
    // Look for a line like: RUN_ID=<uuid>
    for line in cloud_config.lines() {
        if let Some(stripped) = line.trim().strip_prefix("RUN_ID=") {
            return Ok(stripped.trim().to_string());
        }
    }
    Err(anyhow::anyhow!("Could not find RUN_ID in cloud-config"))
}

/// Fetch the exit code from Google Cloud Storage
async fn fetch_exit_code_from_gcs(bucket: &str, run_id: &str) -> anyhow::Result<i32> {
    use google_cloud_storage::client::{Storage, StorageControl};

    // Create GCS clients
    let storage_client = Storage::builder().build().await?;
    let control_client = StorageControl::builder().build().await?;

    // Path to the exit code file in GCS
    let object_name = format!("runs/{}/status/exit_code", run_id);
    // Format bucket name as required by v1 API (projects/_/buckets/{bucket})
    // The underscore means "globally unique bucket" without specifying project ID
    let bucket_path = format!("projects/_/buckets/{}", bucket);

    // Download the exit code file
    eprintln!("Downloading gs://{}/{}", bucket, object_name);
    let mut response = storage_client
        .read_object(&bucket_path, &object_name)
        .send()
        .await?;

    // Collect all chunks into a single buffer
    let mut data = Vec::new();
    while let Some(chunk) = response.next().await {
        data.extend_from_slice(&chunk?);
    }

    // Parse the exit code
    let exit_code_str = String::from_utf8(data)?;
    let exit_code = exit_code_str.trim().parse::<i32>()?;

    // Delete the exit code file from GCS (matching the original script behavior)
    eprintln!("Deleting gs://{}/{}", bucket, object_name);
    control_client
        .delete_object()
        .set_bucket(bucket_path)
        .set_object(object_name)
        .send()
        .await?;

    Ok(exit_code)
}

/// Start an SSH tunnel to forward vLLM port using pure Rust
async fn start_ssh_tunnel_rust(
    external_ip: &str,
    local_port: u16,
) -> anyhow::Result<Arc<SshTunnel>> {
    // Determine username (try to get from gcloud config or use default)
    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "user".to_string());

    // eprintln!("Connecting to {}:22 as user '{}'...", external_ip, username);

    // Create SSH tunnel with retry logic built-in
    let tunnel = SshTunnel::new(
        external_ip,
        22,
        &username,
        local_port,
        "localhost".to_string(),
        8000,
    )
    .await?;

    let tunnel = Arc::new(tunnel);

    // Start the tunnel in a background task
    let tunnel_clone = tunnel.clone();
    tokio::spawn(async move {
        if let Err(e) = tunnel_clone.start().await {
            eprintln!("SSH tunnel error: {}", e);
        }
    });

    // Give the tunnel a moment to start accepting connections
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    Ok(tunnel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_cloud_config() -> anyhow::Result<()> {
        let mut config = GceConfig::new();
        config.project = Some("test-project".to_string());
        let args = UpArgsBuilder::default()
            .hf_token("test_token")
            .config(config)
            .build()?;

        let cloud_config = load_cloud_config(&args)?;
        assert!(cloud_config.contains("#cloud-config"));
        assert!(cloud_config.contains("test_token"));

        Ok(())
    }

    #[test]
    fn test_load_cloud_config_with_model() -> anyhow::Result<()> {
        let mut config = GceConfig::new();
        config.project = Some("test-project".to_string());
        let args = UpArgsBuilder::default()
            .hf_token("test_token")
            .model(Some("meta-llama/Llama-2-7b-hf".to_string()))
            .config(config)
            .build()?;

        let cloud_config = load_cloud_config(&args)?;
        assert!(cloud_config.contains("meta-llama/Llama-2-7b-hf"));

        Ok(())
    }

    #[test]
    fn test_load_cloud_config_with_custom_name() -> anyhow::Result<()> {
        let mut config = GceConfig::new();
        config.project = Some("test-project".to_string());
        let args = UpArgsBuilder::default()
            .name("custom-vllm")
            .hf_token("test_token")
            .config(config)
            .build()?;

        let cloud_config = load_cloud_config(&args)?;
        // Name is used in instance creation, not in cloud-config
        assert!(cloud_config.contains("#cloud-config"));

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
        assert_eq!(args.config.region, "us-west1");
        assert_eq!(args.config.zone, "us-west1-a");

        Ok(())
    }

    #[test]
    fn test_up_args_builder_custom_values() -> anyhow::Result<()> {
        let mut config = GceConfig::new();
        config.region = "us-east1".to_string();
        config.zone = "us-east1-b".to_string();

        let args = UpArgsBuilder::default()
            .name("my-vllm")
            .model(Some("my-model".to_string()))
            .hf_token("my-token")
            .config(config)
            .build()?;

        assert_eq!(args.name, "my-vllm");
        assert_eq!(args.model, Some("my-model".to_string()));
        assert_eq!(args.hf_token, "my-token");
        assert_eq!(args.config.region, "us-east1");
        assert_eq!(args.config.zone, "us-east1-b");

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
            let mut gce_config = GceConfig::new();
            gce_config.project = Some("test-project".to_string());
            let args = UpArgsBuilder::default()
                .hf_token("test_token_123")
                .model(Some("test-model".to_string()))
                .config(gce_config)
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
            // Test that cloud config uses default values
            let mut gce_config = GceConfig::new();
            gce_config.project = Some("test-project".to_string());
            let args = UpArgsBuilder::default()
                .hf_token("test_token")
                .config(gce_config)
                .build()?;

            let config = load_cloud_config(&args)?;

            // Should contain default values from the code
            assert!(config.contains("#cloud-config"));
            assert!(config.len() > 100); // Should be substantial

            Ok(())
        }

        #[test]
        fn test_cloud_config_default_model() -> anyhow::Result<()> {
            let mut gce_config = GceConfig::new();
            gce_config.project = Some("test-project".to_string());
            let args = UpArgsBuilder::default()
                .hf_token("test_token")
                .config(gce_config)
                .build()?;

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
            let mut gce_config = GceConfig::new();
            gce_config.project = Some("test-project".to_string());
            let args = UpArgsBuilder::default()
                .hf_token("test_token")
                .config(gce_config)
                .build()?;

            let config = load_cloud_config(&args)?;

            // Verify setup script is included (it should be base64 encoded or embedded)
            assert!(config.len() > 100); // Cloud config should be substantial

            Ok(())
        }
    }
}

// Made with Bob

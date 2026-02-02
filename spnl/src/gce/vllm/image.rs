use super::args::GceConfig;

/// Generate image name from patch content hash and vLLM source identifier
fn generate_image_name(
    patch_content: &[u8],
    vllm_org: &str,
    vllm_repo: &str,
    vllm_branch: &str,
) -> String {
    use sha2::{Digest, Sha256};

    // Compute patch content hash
    let patch_hash = format!("{:x}", Sha256::digest(patch_content));

    // Create vLLM source identifier
    let vllm_source_id = format!("{}/{}@{}", vllm_org, vllm_repo, vllm_branch);

    // Combine and hash (GCE image names have 63 char limit, format is "vllm-spnl-{hash}")
    let combined = format!("{}{}", patch_hash, vllm_source_id);
    let combined_hash = Sha256::digest(combined.as_bytes());
    let hash_str = format!("{:x}", combined_hash);

    // Take first 52 characters (63 - 11 for "vllm-spnl-")
    let truncated_hash = &hash_str[..52.min(hash_str.len())];

    format!("vllm-spnl-{}", truncated_hash)
}

/// Arguments for creating a vLLM GCE image
#[derive(derive_builder::Builder)]
pub struct ImageCreateArgs {
    /// Force overwrite of existing image with the same name
    #[builder(default = "false")]
    pub(crate) force_overwrite: bool,

    /// vLLM organization on GitHub
    #[builder(setter(into), default = "neuralmagic".to_string())]
    pub(crate) vllm_org: String,

    /// vLLM repository name
    #[builder(setter(into), default = "vllm".to_string())]
    pub(crate) vllm_repo: String,

    /// vLLM branch to use
    #[builder(setter(into), default = "llm-d-release-0.4".to_string())]
    pub(crate) vllm_branch: String,

    /// LLM-D version for patch file
    #[builder(setter(into), default = "0.4.0".to_string())]
    pub(crate) llmd_version: String,

    /// Custom image name (defaults to auto-generated from hash)
    #[builder(default)]
    pub(crate) image_name: Option<String>,

    /// Image family
    #[builder(setter(into), default = "vllm-spnl".to_string())]
    pub(crate) image_family: String,

    /// GCE configuration
    #[builder(default)]
    pub(crate) config: GceConfig,
}

/// Generate the startup script for image preparation
fn generate_startup_script(
    vllm_org: &str,
    vllm_repo: &str,
    vllm_branch: &str,
    patch_content_b64: &str,
) -> String {
    format!(
        r#"#!/usr/bin/env bash
set -euo pipefail

export HOME=/root
cd $HOME

# Load environment
if [[ -f /etc/environment ]]; then
    . /etc/environment
fi

echo "=== Disabling unnecessary services ==="
# Disable services not needed for vLLM/ollama
sudo systemctl disable snapd.service || true
sudo systemctl disable snapd.socket || true
sudo systemctl disable unattended-upgrades.service || true
sudo systemctl disable apt-daily.timer || true
sudo systemctl disable apt-daily-upgrade.timer || true

echo "=== Resizing root filesystem ==="
# Ensure the root filesystem uses the full disk size
sudo growpart /dev/sda 1 2>/dev/null || true
sudo resize2fs /dev/sda1 2>/dev/null || true

echo "=== Installing vLLM ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/{}/{}.git vllm -b {}
cd vllm

echo "=== Applying vLLM patch ==="
# Decode the embedded patch file
cat << 'PATCH_EOF' | base64 -d > /tmp/vllm-patch.gz
{}
PATCH_EOF

# Apply the patch
gunzip -c /tmp/vllm-patch.gz | git apply
rm /tmp/vllm-patch.gz

echo "=== Installing vLLM with dependencies ==="
uv venv --seed
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .

echo "=== Installing ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Creating systemd service for vLLM ==="
# Create directory for vLLM configuration
sudo mkdir -p /etc/vllm

# Create default configuration file (can be overridden at instance startup)
sudo tee /etc/vllm/config > /dev/null << 'VLLM_CONFIG_EOF'
# vLLM Configuration
# These values can be overridden by setting them in this file at instance startup
MODEL=meta-llama/Llama-3.2-1B-Instruct
VLLM_ATTENTION_BACKEND=TRITON_ATTN
VLLM_USE_V1=1
VLLM_V1_SPANS_ENABLED=True
VLLM_V1_SPANS_TOKEN_PLUS=10
VLLM_V1_SPANS_TOKEN_CROSS=13
VLLM_SERVER_DEV_MODE=1
VLLM_CONFIG_EOF

# Create vLLM systemd service that reads from config file
sudo tee /etc/systemd/system/vllm.service > /dev/null << 'VLLM_SERVICE_EOF'
[Unit]
Description=vLLM Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/vllm
Environment="HOME=/root"
Environment="PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile=/etc/vllm/config
ExecStart=/bin/bash -c 'source /root/vllm/.venv/bin/activate && vllm serve ${{MODEL}} --enforce-eager'
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
VLLM_SERVICE_EOF

echo "=== Creating systemd service for Ollama ==="
# Create Ollama systemd service (ollama install.sh already creates one, but we ensure it's enabled)
sudo systemctl enable ollama.service

echo "=== Enabling services to start at boot ==="
sudo systemctl enable vllm.service

echo "=== Cleaning up ==="
# Clean up build artifacts to reduce image size
cd $HOME
if [[ -d vllm ]]; then
    # Keep vllm directory but clean cache
    find vllm -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null || true
fi

# Clean package manager caches
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "=== Image preparation complete ==="
"#,
        vllm_org, vllm_repo, vllm_branch, patch_content_b64
    )
}

/// Find the most recent Ubuntu accelerator image
async fn find_base_image() -> anyhow::Result<String> {
    use google_cloud_compute_v1::client::Images;

    let client = Images::builder().build().await?;

    eprintln!("Finding most recent Ubuntu accelerator image...");

    // List images from ubuntu-os-accelerator-images project
    // Note: GCE API doesn't support both filter and orderBy, so we filter and sort in code
    let response = client
        .list()
        .set_project("ubuntu-os-accelerator-images")
        .set_filter("family eq ubuntu-accelerator-2404-amd64-with-nvidia-580")
        .send()
        .await?;

    let mut images = response.items;
    if images.is_empty() {
        return Err(anyhow::anyhow!("Could not find base image"));
    }

    // Sort by creation timestamp descending (most recent first)
    images.sort_by(|a, b| {
        let a_time = a.creation_timestamp.as_deref().unwrap_or("");
        let b_time = b.creation_timestamp.as_deref().unwrap_or("");
        b_time.cmp(a_time) // Reverse order for descending
    });

    let image_name = images[0]
        .name
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Image has no name"))?;

    eprintln!("Using base image: {}", image_name);
    Ok(image_name)
}

/// Create a GCE instance for image building
async fn create_builder_vm(
    instance_name: &str,
    project: &str,
    zone: &str,
    base_image: &str,
    startup_script: &str,
) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;
    use google_cloud_compute_v1::model::{
        AcceleratorConfig, AccessConfig, AttachedDisk, AttachedDiskInitializeParams, Instance,
        Metadata, NetworkInterface, Scheduling, metadata::Items as MetadataItems,
    };
    use google_cloud_lro::Poller;

    let client = Instances::builder().build().await?;

    let instance = Instance::new()
        .set_name(instance_name)
        .set_machine_type(format!("zones/{}/machineTypes/g2-standard-4", zone))
        .set_disks([AttachedDisk::new()
            .set_boot(true)
            .set_auto_delete(true)
            .set_device_name(instance_name)
            .set_initialize_params(
                AttachedDiskInitializeParams::new()
                    .set_source_image(format!(
                        "projects/ubuntu-os-accelerator-images/global/images/{}",
                        base_image
                    ))
                    .set_disk_size_gb(100)
                    .set_disk_type(format!("zones/{}/diskTypes/pd-ssd", zone)),
            )
            .set_mode("READ_WRITE")])
        .set_network_interfaces([NetworkInterface::new()
            .set_access_configs([AccessConfig::new().set_network_tier("PREMIUM")])])
        .set_guest_accelerators([AcceleratorConfig::new()
            .set_accelerator_count(1)
            .set_accelerator_type(format!("zones/{}/acceleratorTypes/nvidia-l4", zone))])
        .set_metadata(
            Metadata::default().set_items([MetadataItems::default()
                .set_key("startup-script")
                .set_value(startup_script)]),
        )
        .set_scheduling(
            Scheduling::new()
                .set_automatic_restart(false)
                .set_on_host_maintenance("TERMINATE")
                .set_preemptible(true)
                .set_provisioning_model("SPOT")
                .set_instance_termination_action("STOP"),
        );

    eprintln!("Creating VM instance: {}", instance_name);
    client
        .insert()
        .set_project(project)
        .set_zone(zone)
        .set_body(instance)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("VM instance created successfully");
    Ok(())
}

/// Try to create VM in multiple regions/zones
async fn create_vm_with_retry(
    instance_name: &str,
    project: &str,
    base_image: &str,
    startup_script: &str,
) -> anyhow::Result<String> {
    let regions = ["us-west1", "us-central1", "us-east1"];
    let zones = ["a", "b", "c"];
    let max_retries = 3;

    for region in &regions {
        for zone_suffix in &zones {
            let zone = format!("{}-{}", region, zone_suffix);

            for attempt in 1..=max_retries {
                eprintln!(
                    "Attempt {}/{}: Trying to create VM in zone {}...",
                    attempt, max_retries, zone
                );

                match create_builder_vm(instance_name, project, &zone, base_image, startup_script)
                    .await
                {
                    Ok(_) => {
                        eprintln!("Successfully created VM in zone {}", zone);
                        return Ok(zone);
                    }
                    Err(e) => {
                        eprintln!(
                            "Failed to create VM in zone {} (attempt {}/{}): {}",
                            zone, attempt, max_retries, e
                        );
                        if attempt < max_retries {
                            eprintln!("Waiting 5 seconds before retry...");
                            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                        }
                    }
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "Failed to create VM in any available zone after trying all regions"
    ))
}

/// Monitor serial console output for completion
async fn monitor_startup_completion(
    instance_name: &str,
    zone: &str,
    project: &str,
) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;

    let client = Instances::builder().build().await?;

    eprintln!("Monitoring startup script progress...");
    eprintln!("This may take 15-30 minutes depending on build options...");
    eprintln!("Streaming serial console output:\n---");

    let mut last_start = 0i64;

    loop {
        let output = client
            .get_serial_port_output()
            .set_project(project)
            .set_zone(zone)
            .set_instance(instance_name)
            .set_port(1)
            .set_start(last_start)
            .send()
            .await?;

        if let Some(contents) = output.contents.filter(|c| !c.is_empty()) {
            print!("{}", contents);
            std::io::Write::flush(&mut std::io::stdout()).ok();

            // Check for completion marker
            if contents.contains("Image preparation complete") {
                eprintln!("\n---\nStartup script completed successfully!");
                return Ok(());
            }

            // Check for failure
            if let Some(status_line) = contents
                .lines()
                .find(|line| line.contains("startup-script exit status"))
                && !status_line.contains("status 0")
            {
                return Err(anyhow::anyhow!("Startup script failed: {}", status_line));
            }
        }

        if let Some(next) = output.next {
            last_start = next;
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}

/// Stop a GCE instance
async fn stop_instance(instance_name: &str, zone: &str, project: &str) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;
    use google_cloud_lro::Poller;

    let client = Instances::builder().build().await?;

    eprintln!("Stopping VM before creating image...");
    client
        .stop()
        .set_project(project)
        .set_zone(zone)
        .set_instance(instance_name)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("VM stopped successfully");
    Ok(())
}

/// Parameters for creating a custom image from a VM disk
struct ImageCreationParams<'a> {
    image_name: &'a str,
    image_family: &'a str,
    instance_name: &'a str,
    zone: &'a str,
    project: &'a str,
    vllm_org: &'a str,
    vllm_repo: &'a str,
    vllm_branch: &'a str,
    llmd_version: &'a str,
}

/// Create a custom image from a VM disk
async fn create_image_from_disk(params: ImageCreationParams<'_>) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Images;
    use google_cloud_lro::Poller;

    let client = Images::builder().build().await?;

    let description = format!(
        "vLLM custom image with VLLM_ORG={}, VLLM_REPO={}, VLLM_BRANCH={}, LLMD_VERSION={}",
        params.vllm_org, params.vllm_repo, params.vllm_branch, params.llmd_version
    );

    eprintln!("Creating custom image: {}", params.image_name);

    let image = google_cloud_compute_v1::model::Image::new()
        .set_name(params.image_name)
        .set_source_disk(format!(
            "projects/{}/zones/{}/disks/{}",
            params.project, params.zone, params.instance_name
        ))
        .set_family(params.image_family)
        .set_description(&description);

    client
        .insert()
        .set_project(params.project)
        .set_body(image)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("Image created successfully");
    Ok(())
}

/// Make image publicly accessible
async fn make_image_public(image_name: &str, project: &str) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Images;

    let client = Images::builder().build().await?;

    eprintln!("Making image publicly accessible...");

    let binding = google_cloud_compute_v1::model::Binding::new()
        .set_role("roles/compute.imageUser")
        .set_members(["allAuthenticatedUsers"]);

    let policy_request = google_cloud_compute_v1::model::GlobalSetPolicyRequest::new()
        .set_policy(google_cloud_compute_v1::model::Policy::new().set_bindings([binding]));

    client
        .set_iam_policy()
        .set_project(project)
        .set_resource(image_name)
        .set_body(policy_request)
        .send()
        .await?;

    eprintln!("Image is now publicly accessible");
    Ok(())
}

/// Delete a GCE instance
async fn delete_instance(instance_name: &str, zone: &str, project: &str) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;
    use google_cloud_lro::Poller;

    let client = Instances::builder().build().await?;

    eprintln!("Deleting temporary VM...");
    client
        .delete()
        .set_project(project)
        .set_zone(zone)
        .set_instance(instance_name)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("VM deleted successfully");
    Ok(())
}

/// Delete an existing image
async fn delete_image(image_name: &str, project: &str) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Images;
    use google_cloud_lro::Poller;

    let client = Images::builder().build().await?;

    eprintln!("Deleting existing image: {}", image_name);
    client
        .delete()
        .set_project(project)
        .set_image(image_name)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("Existing image deleted");
    Ok(())
}

/// Check if an image exists
async fn image_exists(image_name: &str, project: &str) -> anyhow::Result<bool> {
    use google_cloud_compute_v1::client::Images;

    let client = Images::builder().build().await?;

    match client
        .get()
        .set_project(project)
        .set_image(image_name)
        .send()
        .await
    {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Create a custom GCE image with vLLM pre-installed
pub async fn create_image(args: ImageCreateArgs) -> anyhow::Result<String> {
    // Get configuration
    let project = args.config.get_project()?;

    eprintln!("=== GCE vLLM Image Builder ===");
    eprintln!("Configuration:");
    eprintln!("  VLLM_ORG: {}", args.vllm_org);
    eprintln!("  VLLM_REPO: {}", args.vllm_repo);
    eprintln!("  VLLM_BRANCH: {}", args.vllm_branch);
    eprintln!("  LLMD_VERSION: {}", args.llmd_version);
    eprintln!("  IMAGE_FAMILY: {}", args.image_family);
    eprintln!("  IMAGE_PROJECT: {}", project);
    eprintln!();

    // Embed patch file at compile time based on LLMD version
    let patch_content = match args.llmd_version.as_str() {
        "0.4.0" => include_bytes!(
            "../../../../docker/vllm/llm-d/patches/0.4.0/01-spans-llmd-vllm.patch.gz"
        ),
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported LLMD version: {}. Only 0.4.0 is currently supported.",
                args.llmd_version
            ));
        }
    };

    eprintln!(
        "Using embedded patch file for LLMD version {}",
        args.llmd_version
    );

    // Generate or use provided image name
    let image_name = if let Some(name) = args.image_name {
        name
    } else {
        generate_image_name(
            patch_content,
            &args.vllm_org,
            &args.vllm_repo,
            &args.vllm_branch,
        )
    };

    eprintln!("  IMAGE_NAME: {}", image_name);
    eprintln!();

    // Check if image already exists
    if image_exists(&image_name, &project).await? {
        if args.force_overwrite {
            eprintln!(
                "Image {} already exists. Deleting due to --force flag...",
                image_name
            );
            delete_image(&image_name, &project).await?;
        } else {
            return Err(anyhow::anyhow!(
                "Image {} already exists. Use --force to overwrite",
                image_name
            ));
        }
    }

    // Find base image
    let base_image = find_base_image().await?;

    // Encode patch file to base64
    let patch_content_b64 =
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, patch_content);

    // Generate startup script
    let startup_script = generate_startup_script(
        &args.vllm_org,
        &args.vllm_repo,
        &args.vllm_branch,
        &patch_content_b64,
    );

    // Create temporary VM name
    let temp_vm_name = format!("vllm-image-builder-{}", uuid::Uuid::new_v4());

    // Create VM with retry logic
    let zone = create_vm_with_retry(&temp_vm_name, &project, &base_image, &startup_script).await?;

    // Monitor startup completion
    let monitor_result = monitor_startup_completion(&temp_vm_name, &zone, &project).await;

    // Stop the VM
    stop_instance(&temp_vm_name, &zone, &project).await?;

    // Check monitor result
    monitor_result?;

    // Create image from disk
    create_image_from_disk(ImageCreationParams {
        image_name: &image_name,
        image_family: &args.image_family,
        instance_name: &temp_vm_name,
        zone: &zone,
        project: &project,
        vllm_org: &args.vllm_org,
        vllm_repo: &args.vllm_repo,
        vllm_branch: &args.vllm_branch,
        llmd_version: &args.llmd_version,
    })
    .await?;

    // Make image public
    make_image_public(&image_name, &project).await?;

    // Delete temporary VM
    delete_instance(&temp_vm_name, &zone, &project).await?;

    eprintln!();
    eprintln!("=== Image creation complete! ===");
    eprintln!("Image name: {}", image_name);
    eprintln!("Image family: {}", args.image_family);
    eprintln!("Project: {}", project);

    Ok(image_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_image_name() {
        let test_content = b"test content";
        let name = generate_image_name(test_content, "neuralmagic", "vllm", "main");

        assert!(name.starts_with("vllm-spnl-"));
        assert!(name.len() <= 63); // GCE limit

        // Test determinism - same input should produce same output
        let name2 = generate_image_name(test_content, "neuralmagic", "vllm", "main");
        assert_eq!(name, name2);

        // Different content should produce different name
        let name3 = generate_image_name(b"different content", "neuralmagic", "vllm", "main");
        assert_ne!(name, name3);
    }

    #[test]
    fn test_image_create_args_builder() {
        let args = ImageCreateArgsBuilder::default()
            .force_overwrite(true)
            .vllm_org("test-org")
            .vllm_repo("test-repo")
            .vllm_branch("test-branch")
            .llmd_version("0.4.0")
            .image_family("test-family")
            .build()
            .unwrap();

        assert!(args.force_overwrite);
        assert_eq!(args.vllm_org, "test-org");
        assert_eq!(args.vllm_repo, "test-repo");
        assert_eq!(args.vllm_branch, "test-branch");
        assert_eq!(args.llmd_version, "0.4.0");
        assert_eq!(args.image_family, "test-family");
    }
}

// Made with Bob

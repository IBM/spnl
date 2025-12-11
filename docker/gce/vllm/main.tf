# This code is compatible with Terraform 4.25.0 and versions that are backwards compatible to 4.25.0.
# For information about validating this Terraform code, see https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

provider "google" {
  project     = var.gcp_project
  region      = var.gce_region
}

# Automatically assigned run_id if not provided as a variable
resource "random_uuid" "auto_run_id" {
}

locals {
  run_id = var.run_id=="" ? random_uuid.auto_run_id.result : var.run_id
}

resource "google_compute_instance" "spnl-test-big" {
  boot_disk {
    auto_delete = true
    device_name = "spnl-test-big-${local.run_id}"

    initialize_params {
      image = "projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-580-v20251210"
      size  = 100
      type  = "pd-ssd"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  guest_accelerator {
    count = 1
    type  = "projects/${var.gcp_project}/zones/${var.gce_zone}/acceleratorTypes/nvidia-l4"
  }

  labels = {
    role                  = "gh-runner"
    gh-run-id             = local.run_id
    goog-ec-src           = "vm_add-tf"
    goog-ops-agent-policy = "v2-x86-template-1-4-0"
  }

  machine_type = var.machine_type

  metadata = {
    enable-osconfig = "TRUE",
    user-data = templatefile("cloud-config.yaml", {run_id=local.run_id,hf_token=var.hf_token,model=var.model,gcs_bucket=var.gcs_bucket,spnl_github=var.spnl_github,spnl_github_sha=var.spnl_github_sha,spnl_github_ref=var.spnl_github_ref,vllm_org=var.vllm_org,vllm_repo=var.vllm_repo,vllm_branch=var.vllm_branch,setup_script=indent(6, file("setup.sh"))})
  }

  name = "spnl-test-big-${local.run_id}"

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/${var.gcp_project}/regions/${var.gce_region}/subnetworks/default"
  }

  scheduling {
    automatic_restart   = false # true with STANDARD
    on_host_maintenance = "TERMINATE"
    preemptible         = true # false with STANDARD
    provisioning_model  = "SPOT" # versus STANDARD
  }

  service_account {
    email  = "${var.gcp_service_account}@${var.gcp_project}.iam.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/devstorage.read_write", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/trace.append"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  zone = var.gce_zone
}

/*module "ops_agent_policy" {
  source          = "github.com/terraform-google-modules/terraform-google-cloud-operations/modules/ops-agent-policy"
  project         = "${var.gcp_project}"
  zone            = var.gce_zone
  assignment_id   = "goog-ops-agent-v2-x86-template-1-4-0-${var.gce_zone}"
  agents_rule = {
    package_state = "installed"
    version = "latest"
  }
  instance_filter = {
    all = false
    inclusion_labels = [{
      labels = {
        goog-ops-agent-policy = "v2-x86-template-1-4-0"
      }
    }]
  }
}
*/

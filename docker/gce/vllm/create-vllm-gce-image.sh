#!/usr/bin/env bash

#
# Create a custom GCE image with vLLM pre-installed
# This script creates a reusable image based on the setup.sh logic
#

set -euo pipefail

# Parse command line arguments
FORCE_OVERWRITE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_OVERWRITE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-f|--force]"
            echo ""
            echo "Options:"
            echo "  -f, --force    Force overwrite of existing image with the same name"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Environment variables (optional):"
            echo "  VLLM_ORG           vLLM organization (default: neuralmagic)"
            echo "  VLLM_REPO          vLLM repository (default: vllm)"
            echo "  VLLM_BRANCH        vLLM branch (default: llm-d-release-0.4)"
            echo "  LLMD_VERSION       LLM-D version (default: 0.4.0)"
            echo "  IMAGE_NAME         Custom image name (default: auto-generated from hash)"
            echo "  IMAGE_FAMILY       Image family (default: vllm-spnl)"
            echo "  IMAGE_PROJECT      GCP project (default: current project)"
            echo "  ZONE               GCP zone (default: us-west1-a)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located (must be early in the script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration parameters (can be overridden via environment variables)
# Defaults match spnl/src/gce/vllm/up.rs
: "${VLLM_ORG:=neuralmagic}"
: "${VLLM_REPO:=vllm}"
: "${VLLM_BRANCH:=llm-d-release-0.4}"
: "${LLMD_VERSION:=0.4.0}"
: "${SPNL_VERSION:=}"
: "${IMAGE_FAMILY:=vllm-spnl}"
: "${IMAGE_PROJECT:=$(gcloud config get-value project)}"
: "${ZONE:=us-west1-a}"
: "${MACHINE_TYPE:=g2-standard-4}"
: "${ACCELERATOR_TYPE:=nvidia-l4}"
: "${ACCELERATOR_COUNT:=1}"
: "${DISK_SIZE:=100}"
: "${DISK_TYPE:=pd-ssd}"

# Compute shasum of patch file for image naming
PATCH_FILE_PATH="$SCRIPT_DIR/../../vllm/llm-d/patches/$LLMD_VERSION/01-spans-llmd-vllm.patch.gz"
if [[ ! -f "$PATCH_FILE_PATH" ]]; then
    echo "Error: Patch file not found at $PATCH_FILE_PATH"
    echo "Cannot compute shasum for image name"
    exit 1
fi

# Create a combined hash of the patch file and vLLM source identifier
# GCE image names have a 63 character limit. Format is "vllm-spnl-{hash}" (11 chars + hash)
# So we can use up to 52 characters for the hash (63 - 11 = 52)
PATCH_SHASUM=$(shasum -a 256 "$PATCH_FILE_PATH" | cut -d' ' -f1)
VLLM_SOURCE_ID="${VLLM_ORG}/${VLLM_REPO}@${VLLM_BRANCH}"
COMBINED_HASH=$(echo -n "${PATCH_SHASUM}${VLLM_SOURCE_ID}" | shasum -a 256 | cut -d' ' -f1 | cut -c1-52)

# Set IMAGE_NAME based on combined hash
: "${IMAGE_NAME:=vllm-spnl-${COMBINED_HASH}}"

# Temporary VM name
TEMP_VM_NAME="vllm-image-builder-$(date +%s)"

echo "=== GCE vLLM Image Builder ==="
echo "Configuration:"
echo "  VLLM_ORG: $VLLM_ORG"
echo "  VLLM_REPO: $VLLM_REPO"
echo "  VLLM_BRANCH: $VLLM_BRANCH"
echo "  LLMD_VERSION: $LLMD_VERSION"
echo "  IMAGE_NAME: $IMAGE_NAME"
echo "  IMAGE_FAMILY: $IMAGE_FAMILY"
echo "  IMAGE_PROJECT: $IMAGE_PROJECT"
echo "  ZONE: $ZONE"
echo ""

# Find the most recent Ubuntu accelerator image
echo "Finding most recent Ubuntu accelerator image..."
BASE_IMAGE=$(gcloud compute images list \
    --project=ubuntu-os-accelerator-images \
    --filter="family:ubuntu-accelerator-2404-amd64-with-nvidia-580" \
    --format="value(name)" \
    --sort-by="~creationTimestamp" \
    --limit=1)

if [[ -z "$BASE_IMAGE" ]]; then
    echo "Error: Could not find base image"
    exit 1
fi

echo "Using base image: $BASE_IMAGE"
echo ""

# Locate the patch file (relative to script directory)
PATCH_FILE="$SCRIPT_DIR/../../vllm/llm-d/patches/$LLMD_VERSION/01-spans-llmd-vllm.patch.gz"

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "Error: Patch file not found at $PATCH_FILE"
    echo "Please ensure you're running this script from the correct location"
    exit 1
fi

echo "Using patch file: $PATCH_FILE"
echo ""

# Extract the patch content and base64 encode it for embedding in the startup script
PATCH_CONTENT_B64=$(base64 < "$PATCH_FILE")

# Create temporary file for startup script
STARTUP_SCRIPT=$(mktemp)
trap "rm -f $STARTUP_SCRIPT" EXIT

# Create startup script that will prepare the image
cat > "$STARTUP_SCRIPT" << SETUP_SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail

export HOME=/root
cd \$HOME

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
source \$HOME/.local/bin/env
git clone https://github.com/$VLLM_ORG/$VLLM_REPO.git vllm -b $VLLM_BRANCH
cd vllm

echo "=== Applying vLLM patch ==="
# Decode the embedded patch file
cat << 'PATCH_EOF' | base64 -d > /tmp/vllm-patch.gz
$PATCH_CONTENT_B64
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
ExecStart=/bin/bash -c 'source /root/vllm/.venv/bin/activate && vllm serve \${MODEL} --enforce-eager'
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
cd \$HOME
if [[ -d vllm ]]; then
    # Keep vllm directory but clean cache
    find vllm -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
fi

# Clean package manager caches
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "=== Image preparation complete ==="
SETUP_SCRIPT_EOF

# Try multiple regions and zones to find available capacity
REGIONS=("us-west1" "us-central1" "us-east1")
ZONES=("a" "b" "c")
MAX_RETRIES=3

VM_CREATED=false
ACTUAL_ZONE=""
ACTUAL_REGION=""

for region in "${REGIONS[@]}"; do
    if [[ "$VM_CREATED" == "true" ]]; then
        break
    fi
    
    for zone_suffix in "${ZONES[@]}"; do
        if [[ "$VM_CREATED" == "true" ]]; then
            break
        fi
        
        zone="${region}-${zone_suffix}"
        
        for attempt in $(seq 1 $MAX_RETRIES); do
            echo "Attempt $attempt/$MAX_RETRIES: Trying to create VM in zone $zone..."
            
            if gcloud compute instances create "$TEMP_VM_NAME" \
                --project="$IMAGE_PROJECT" \
                --zone="$zone" \
                --machine-type="$MACHINE_TYPE" \
                --accelerator="type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT" \
                --maintenance-policy=TERMINATE \
                --preemptible \
                --image="$BASE_IMAGE" \
                --image-project=ubuntu-os-accelerator-images \
                --boot-disk-size="${DISK_SIZE}Gi" \
                --boot-disk-type="$DISK_TYPE" \
                --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
                --scopes=cloud-platform 2>&1; then
                
                echo "Successfully created VM in zone $zone"
                VM_CREATED=true
                ACTUAL_ZONE="$zone"
                ACTUAL_REGION="$region"
                break
            else
                echo "Failed to create VM in zone $zone (attempt $attempt/$MAX_RETRIES)"
                if [[ $attempt -lt $MAX_RETRIES ]]; then
                    echo "Waiting 5 seconds before retry..."
                    sleep 5
                fi
            fi
        done
    done
done

if [[ "$VM_CREATED" != "true" ]]; then
    echo "Error: Failed to create VM in any available zone after trying all regions"
    exit 1
fi

# Update ZONE variable for subsequent commands
ZONE="$ACTUAL_ZONE"
echo "Using zone: $ZONE"

echo ""
echo "Waiting for VM to be ready..."
gcloud compute instances describe "$TEMP_VM_NAME" \
    --project="$IMAGE_PROJECT" \
    --zone="$ZONE" \
    --format="value(status)" | grep -q RUNNING

echo "VM is running. Monitoring startup script progress..."
echo "This may take 15-30 minutes depending on build options..."
echo ""
echo "Streaming serial console output (Ctrl+C to stop monitoring, image build will continue):"
echo "---"

# Stream serial console output in real-time
LAST_LINE_COUNT=0
STARTUP_COMPLETE=false

while true; do
    OUTPUT=$(gcloud compute instances get-serial-port-output "$TEMP_VM_NAME" \
        --project="$IMAGE_PROJECT" \
        --zone="$ZONE" 2>/dev/null || echo "")
    
    # Count lines and show only new ones
    CURRENT_LINE_COUNT=$(echo "$OUTPUT" | wc -l)
    if [[ $CURRENT_LINE_COUNT -gt $LAST_LINE_COUNT ]]; then
        echo "$OUTPUT" | tail -n +$((LAST_LINE_COUNT + 1))
        LAST_LINE_COUNT=$CURRENT_LINE_COUNT
    fi
    
    # Check for completion
    if echo "$OUTPUT" | grep -q "Image preparation complete"; then
        echo "---"
        echo "Startup script completed successfully!"
        STARTUP_COMPLETE=true
        break
    fi
    
    # Check for failure
    if echo "$OUTPUT" | grep -q "startup-script exit status"; then
        EXIT_STATUS=$(echo "$OUTPUT" | grep "startup-script exit status" | tail -1 | grep -oP 'status \K\d+' || echo "unknown")
        if [[ "$EXIT_STATUS" != "0" ]] && [[ "$EXIT_STATUS" != "unknown" ]]; then
            echo "---"
            echo "Error: Startup script failed with exit status $EXIT_STATUS"
            echo "Full logs available with: gcloud compute instances get-serial-port-output $TEMP_VM_NAME --project=$IMAGE_PROJECT --zone=$ZONE"
            exit 1
        fi
    fi
    
    sleep 5
done

if [[ "$STARTUP_COMPLETE" != "true" ]]; then
    echo "Warning: Could not confirm startup script completion"
    echo "Check logs with: gcloud compute instances get-serial-port-output $TEMP_VM_NAME --project=$IMAGE_PROJECT --zone=$ZONE"
fi

echo ""
echo "Stopping VM before creating image..."
gcloud compute instances stop "$TEMP_VM_NAME" \
    --project="$IMAGE_PROJECT" \
    --zone="$ZONE"

echo "Waiting for VM to stop..."
while true; do
    STATUS=$(gcloud compute instances describe "$TEMP_VM_NAME" \
        --project="$IMAGE_PROJECT" \
        --zone="$ZONE" \
        --format="value(status)")
    
    if [[ "$STATUS" == "TERMINATED" ]]; then
        break
    fi
    sleep 5
done

echo ""
echo "Checking if image already exists..."
if gcloud compute images describe "$IMAGE_NAME" --project="$IMAGE_PROJECT" &>/dev/null; then
    if [[ "$FORCE_OVERWRITE" == "true" ]]; then
        echo "Image $IMAGE_NAME already exists. Deleting due to --force flag..."
        gcloud compute images delete "$IMAGE_NAME" \
            --project="$IMAGE_PROJECT" \
            --quiet
        echo "Existing image deleted."
    else
        echo "Error: Image $IMAGE_NAME already exists in project $IMAGE_PROJECT"
        echo "Use -f or --force to overwrite the existing image, or delete it manually with:"
        echo "  gcloud compute images delete $IMAGE_NAME --project=$IMAGE_PROJECT"
        echo ""
        echo "Cleaning up temporary VM..."
        gcloud compute instances delete "$TEMP_VM_NAME" \
            --project="$IMAGE_PROJECT" \
            --zone="$ZONE" \
            --quiet
        exit 1
    fi
else
    echo "Image name is available."
fi

echo ""
echo "Creating custom image: $IMAGE_NAME"
gcloud compute images create "$IMAGE_NAME" \
    --project="$IMAGE_PROJECT" \
    --source-disk="$TEMP_VM_NAME" \
    --source-disk-zone="$ZONE" \
    --family="$IMAGE_FAMILY" \
    --description="vLLM custom image with VLLM_ORG=$VLLM_ORG, VLLM_REPO=$VLLM_REPO, VLLM_BRANCH=$VLLM_BRANCH, LLMD_VERSION=$LLMD_VERSION"

echo ""
echo "Making image publicly accessible..."
gcloud compute images add-iam-policy-binding "$IMAGE_NAME" \
    --project="$IMAGE_PROJECT" \
    --member='allAuthenticatedUsers' \
    --role='roles/compute.imageUser'

echo ""
echo "Deleting temporary VM..."
gcloud compute instances delete "$TEMP_VM_NAME" \
    --project="$IMAGE_PROJECT" \
    --zone="$ZONE" \
    --quiet

echo ""
echo "=== Image creation complete! ==="
echo "Image name: $IMAGE_NAME"
echo "Image family: $IMAGE_FAMILY"
echo "Project: $IMAGE_PROJECT"
echo ""
echo "To use this image, create a VM with:"
echo "  gcloud compute instances create my-vllm-instance \\"
echo "    --zone=$ZONE \\"
echo "    --image=$IMAGE_NAME \\"
echo "    --image-project=$IMAGE_PROJECT \\"
echo "    --machine-type=n1-standard-8 \\"
echo "    --accelerator=type=nvidia-tesla-t4,count=1 \\"
echo "    --maintenance-policy=TERMINATE \\"
echo "    --preemptible"
echo ""
echo "To list all images in this family:"
echo "  gcloud compute images list --project=$IMAGE_PROJECT --filter=\"family:$IMAGE_FAMILY\""

# Made with Bob

#!/usr/bin/env bash

#
# Development setup script for building spnl from source
# Note: this script is executed inside the VM, via a cloud-init runcmd. See ./cloud-config.yaml
#

set -eo pipefail

# DEBUG
#set -x

cleanup() {
    rc=$?
    echo "Exiting with exit_code=$rc"
    gsutil cp <(echo $rc) gs://$GCS_BUCKET/runs/$RUN_ID/status/exit_code
}
trap "cleanup" EXIT

export HOME=/root
cd $HOME

# TODO: i was expecting this to be loaded automatically. Apparently not if this is run via a cloud-init runcmd.
. /etc/environment

export SCCACHE_GCS_BUCKET=$GCS_BUCKET
SCCACHE_VERSION=$(curl -s "https://api.github.com/repos/mozilla/sccache/releases/latest" | grep -Po '"tag_name": "v\K[0-9.]+')
wget -qO sccache.tar.gz https://github.com/mozilla/sccache/releases/latest/download/sccache-v$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz
mkdir sccache-temp
tar xf sccache.tar.gz --strip-components=1 -C sccache-temp
sudo mv sccache-temp/sccache /usr/local/bin
sudo chmod a+x /usr/local/bin/sccache
rm -rf sccache.tar.gz sccache-temp
export RUSTC_WRAPPER=/usr/local/bin/sccache
export SCCACHE_GCS_RW_MODE=READ_WRITE
export SCCACHE_GCS_KEY_PREFIX=sccache

# Install and build spnl from source
export CARGO_INCREMENTAL=0 # Disable incremental compilation for faster from-scratch builds
export CARGO_PROFILE_TEST_DEBUG=0

if [[ -n "$GITHUB_SHA" ]] && [[ -n "$GITHUB_REF" ]]
then
    echo "Cloning spnl from GITHUB_SHA=$GITHUB_SHA GITHUB_REF=$GITHUB_REF"
    (
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
            source $HOME/.cargo/env && \
            mkdir spnl && \
            cd spnl && \
            git init && \
            git remote add origin $SPNL_GITHUB && \
            git fetch --prune --no-recurse-submodules --depth=1 origin +$GITHUB_SHA:$GITHUB_REF && \
            git checkout --progress --force $GITHUB_REF && \
            cargo build -F rag,spnl-api,vllm && sudo cp target/debug/spnl /usr/local/bin && sudo chmod a+rX /usr/local/bin/spnl \
            ) &
    spnl_pid=$!
else
    echo "Cloning spnl from repo=$SPNL_GITHUB"
    (
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
            source $HOME/.cargo/env && \
            git clone $SPNL_GITHUB spnl && \
            cd spnl && \
            cargo build -F rag,spnl-api,vllm && sudo cp target/debug/spnl /usr/local/bin && sudo chmod a+rX /usr/local/bin/spnl \
            ) &
    spnl_pid=$!
fi

# Install vLLM
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/$VLLM_ORG/$VLLM_REPO.git vllm
cd vllm
git fetch origin $VLLM_SHA
git checkout $VLLM_SHA
uv venv --seed
source .venv/bin/activate

# Default VLLM_PRECOMPILED_WHEEL_COMMIT to VLLM_SHA if not set
# This allows using precompiled binaries from a different commit (e.g., main) while checking out a specific source commit
VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-${VLLM_SHA}}"

# Detect if precompiled wheel exists (following llm-d approach)
MACHINE=$(uname -m)
case "${MACHINE}" in
  x86_64|amd64) PLATFORM_TAG="manylinux_2_31_x86_64" ;;
  aarch64|arm64) PLATFORM_TAG="manylinux_2_31_aarch64" ;;
  *) echo "Unsupported architecture: ${MACHINE}"; PLATFORM_TAG="" ;;
esac

WHEEL_URL=""
if [ -n "${PLATFORM_TAG}" ]; then
  echo "Looking for precompiled wheel at: https://wheels.vllm.ai/${VLLM_PRECOMPILED_WHEEL_COMMIT}/vllm/"
  WHEEL_INDEX_HTML=$(curl -sf "https://wheels.vllm.ai/${VLLM_PRECOMPILED_WHEEL_COMMIT}/vllm/" 2>/dev/null || echo "")
  if [ -n "${WHEEL_INDEX_HTML}" ]; then
    WHEEL_FILENAME=$(echo "${WHEEL_INDEX_HTML}" | grep -oE "vllm-[^\"]+${PLATFORM_TAG}\.whl" | head -1)
    if [ -n "${WHEEL_FILENAME}" ]; then
      # URL-encode the + sign in the wheel filename
      WHEEL_URL="https://wheels.vllm.ai/${VLLM_PRECOMPILED_WHEEL_COMMIT}/${WHEEL_FILENAME}"
      WHEEL_URL=$(echo "${WHEEL_URL}" | sed -E 's/\+/%2B/g')
      echo "Found precompiled wheel: ${WHEEL_URL}"
    fi
  fi
fi

# Install vLLM with or without precompiled binaries
if [ -n "${WHEEL_URL}" ]; then
  echo "Using precompiled binaries from commit: ${VLLM_PRECOMPILED_WHEEL_COMMIT} (source: ${VLLM_SHA})"
  export VLLM_USE_PRECOMPILED=1
  export VLLM_PRECOMPILED_WHEEL_LOCATION="${WHEEL_URL}"
  uv pip install --editable .
else
  echo "Compiling vLLM from source (no precompiled wheel found or unsupported platform)"
  unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION 2>/dev/null || true
  uv pip install --editable .
fi

# Wait for spnl build to complete
echo "Waiting for spnl build to complete..."
wait $spnl_pid
echo "spnl build completed"

# Patch the vllm code and install spnl Python package
spnl vllm patchfile | git apply

# Build the cloned version of spnl into vLLM, via maturin
uv pip install maturin[patchelf]
source $HOME/.cargo/env # to get rustc on path
(cd $HOME/spnl && maturin develop -F tok,run_py -m spnl/Cargo.toml)

# Start vLLM
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    VLLM_USE_V1=1 \
    VLLM_V1_SPANS_ENABLED=True \
    VLLM_V1_SPANS_TOKEN_PLUS=10 \
    VLLM_V1_SPANS_TOKEN_CROSS=13 \
    VLLM_SERVER_DEV_MODE=1 \
    nohup vllm serve $MODEL --enforce-eager &

# Install ollama (for embedding)
(curl -fsSL https://ollama.com/install.sh | sh && ollama serve) &

# Wait till vllm is ready
timeout 5m bash -c 'until curl --output /dev/null --silent --fail http://localhost:8000/health; do sleep 3; done'
echo "vllm is ready"

# Wait till ollama is ready
timeout 5m bash -c 'until ollama ps; do sleep 3; done'
echo "ollama is ready"

# Run tests
# Here are the variables we will allow to be used in the test.d/* scripts
declare -x GCS_BUCKET
declare -x RUN_ID
declare -x MODEL
declare -x OPENAI_API_BASE=http://localhost:8000/v1

cd $HOME
TESTS_DIR=$HOME/spnl/docker/gce/vllm/test.d
if [ -d "$TESTS_DIR" ]
then
    n_tests=$(ls "$TESTS_DIR" | wc -l | xargs)
    echo "Starting $n_tests tests"
    find "$TESTS_DIR" -type f -name '*.sh' -print0 | xargs -0L1 -I{} bash -c 'echo "Executing {} at $(date -u)"; "{}"'
else echo "No tests found in $TESTS_DIR"
fi

# Made with Bob

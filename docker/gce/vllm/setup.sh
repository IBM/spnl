#!/usr/bin/env bash

#
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

# Install and build spnl
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
            cargo build --release -F rag,spnl-api && sudo cp target/release/spnl /usr/local/bin && sudo chmod a+rX /usr/local/bin/spnl \
            ) &
    spnl_pid=$!
else
    echo "Cloning spnl from repo=$SPNL_GITHUB"
    (
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
            source $HOME/.cargo/env && \
            git clone $SPNL_GITHUB spnl && \
            cd spnl && \
            cargo build --release -F rag,spnl-api && sudo cp target/release/spnl /usr/local/bin && sudo chmod a+rX /usr/local/bin/spnl \
            ) &
    spnl_pid=$!
fi

# Install vLLM
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/$VLLM_ORG/$VLLM_REPO.git vllm -b $VLLM_BRANCH
cd vllm
uv venv --seed
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .

# Patch the vllm code. We could do this prior to the pip install just
# above, but... for now at least this would mean downloading the spnl
# pip, which is unnecessary due to the `maturin develop` just below,
# which installs the specific version of spnl that was cloned above
git apply $VLLM_PATCHFILE

# Build the cloned version of spnl into vLLM, via maturin
uv pip install maturin[patchelf]
source $HOME/.cargo/env # to get rustc on path
(cd $HOME/spnl && maturin develop --release -F tok,run_py -m spnl/Cargo.toml)

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
#timeout 5m bash -c 'until curl --output /dev/null --silent --fail http://localhost:11434; do sleep 3; done'
timeout 5m bash -c 'until ollama ps; do sleep 3; done'
echo "ollama is ready"

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

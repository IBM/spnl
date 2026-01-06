#!/bin/sh

set -e

SCRIPTDIR=$(cd $(dirname "$0") && pwd)

SPANS_VLLM_FORK=https://github.com/starpit/vllm-ibm.git
SPANS_VLLM_BRANCH=spnl-ibm

LLMD_VERSION=0.4.0
BASE_VLLM_FORK=https://github.com/neuralmagic/vllm.git
BASE_VLLM_BRANCH=llm-d-release-0.4

T=$(mktemp -d)
trap "rm -rf $T" EXIT

git clone $BASE_VLLM_FORK $T/vllm-llmd -b $BASE_VLLM_BRANCH
cd $T/vllm-llmd
BASE_VLLM_REVISION=$(git rev-parse --verify HEAD)

git remote add spans $SPANS_VLLM_FORK
git fetch spans $SPANS_VLLM_BRANCH
git checkout $SPANS_VLLM_BRANCH
SPANS_VLLM_REVISION=$(git rev-parse --verify HEAD)

git checkout $BASE_VLLM_BRANCH
git rebase spans/$SPANS_VLLM_BRANCH -C0

# Notes: gzip --no-name ensures deterministic output (gzip won't save mtime in the file); this helps with git sanity
mkdir -p "$SCRIPTDIR"/patches
git diff $BASE_VLLM_REVISION | gzip --no-name -c > "$SCRIPTDIR"/patches/llm-d/$LLMD_VERSION/01-spans-llmd-vllm.patch.gz

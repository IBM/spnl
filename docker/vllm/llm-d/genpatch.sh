#!/bin/sh

set -e

SCRIPTDIR=$(cd $(dirname "$0") && pwd)

SPANS_VLLM_FORK=https://github.com/starpit/vllm-ibm.git
SPANS_VLLM_BRANCH=spnl-ibm-3
SPANS_VLLM_BASE=1892993bc1

. "$SCRIPTDIR/defaults.env"
#BASE_VLLM_FORK=https://github.com/$VLLM_ORG/$VLLM_REPO.git
#BASE_VLLM_COMMIT_SHA=$VLLM_COMMIT_SHA

T=$(mktemp -d)
trap "rm -rf $T" EXIT

git clone $SPANS_VLLM_FORK -b $SPANS_VLLM_BRANCH $T/vllm-ibm
cd $T/vllm-ibm

#git clone $BASE_VLLM_FORK $T/vllm-llmd
#cd $T/vllm-llmd
#git fetch origin $BASE_VLLM_COMMIT_SHA
#git checkout -q $BASE_VLLM_COMMIT_SHA
#BASE_VLLM_REVISION=$BASE_VLLM_COMMIT_SHA
#git apply $T/spans.patch

# Notes: gzip --no-name ensures deterministic output (gzip won't save mtime in the file); this helps with git sanity
mkdir -p "$SCRIPTDIR"/patches/$LLMD_VERSION
(cd $T/vllm-ibm && git diff $SPANS_VLLM_BASE..HEAD | gzip --no-name -c > "$SCRIPTDIR"/patches/$LLMD_VERSION/01-spans-llmd-vllm.patch.gz)

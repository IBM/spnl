#!/bin/sh

set -e

SCRIPTDIR=$(cd $(dirname "$0") && pwd)

SPANS_VLLM_FORK=https://github.com/starpit/vllm-ibm.git
SPANS_VLLM_BRANCH=spnl-ibm

. "$SCRIPTDIR/defaults.env"
BASE_VLLM_FORK=https://github.com/$VLLM_ORG/$VLLM_REPO.git
BASE_VLLM_COMMIT_SHA=$VLLM_COMMIT_SHA

T=vllm
#trap "rm -rf $T" EXIT

git clone $BASE_VLLM_FORK $T/vllm-llmd
cd $T/vllm-llmd
git fetch origin $BASE_VLLM_COMMIT_SHA
git checkout -q $BASE_VLLM_COMMIT_SHA
BASE_VLLM_REVISION=$BASE_VLLM_COMMIT_SHA

git remote add spans $SPANS_VLLM_FORK
git fetch spans $SPANS_VLLM_BRANCH

git rebase spans/$SPANS_VLLM_BRANCH -C0 

# Notes: gzip --no-name ensures deterministic output (gzip won't save mtime in the file); this helps with git sanity
mkdir -p "$SCRIPTDIR"/patches/$LLMD_VERSION
git diff $BASE_VLLM_REVISION | gzip --no-name -c > "$SCRIPTDIR"/patches/$LLMD_VERSION/01-spans-llmd-vllm.patch.gz

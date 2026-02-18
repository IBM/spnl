#!/usr/bin/env bash

set -e

SCRIPTDIR=$(cd $(dirname "$0") && pwd)
. "$SCRIPTDIR/defaults.env"
BASE_VLLM_FORK=https://github.com/$VLLM_ORG/$VLLM_REPO.git
BASE_VLLM_COMMIT_SHA=$VLLM_COMMIT_SHA

git clone $BASE_VLLM_FORK vllm
cd vllm
git fetch --depth=1 origin $BASE_VLLM_COMMIT_SHA
git checkout -q $BASE_VLLM_COMMIT_SHA

for patchfile in ../patches/$LLMD_VERSION/*.patch.gz
do git apply <(gunzip -c $patchfile) --reject
done

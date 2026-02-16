#!/usr/bin/env bash

set -e

LLMD_VERSION=0.5.0
BASE_VLLM_FORK=https://github.com/vllm-project/vllm.git
BASE_VLLM_COMMIT_SHA=d7de043d55d1dd629554467e23874097e1c48993

git clone $BASE_VLLM_FORK vllm
cd vllm
git fetch --depth=1 origin $BASE_VLLM_COMMIT_SHA
git checkout -q $BASE_VLLM_COMMIT_SHA

for patchfile in ../patches/$LLMD_VERSION/*.patch.gz
do git apply <(gunzip -c $patchfile) --reject
done

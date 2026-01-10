#!/usr/bin/env bash

set -e

LLMD_VERSION=0.4.0
BASE_VLLM_FORK=https://github.com/neuralmagic/vllm.git
BASE_VLLM_BRANCH=llm-d-release-0.4

git clone $BASE_VLLM_FORK -b $BASE_VLLM_BRANCH --depth 1
cd vllm

for patchfile in ../patches/$LLMD_VERSION/*.patch.gz
do git apply <(gunzip -c $patchfile) --reject
done

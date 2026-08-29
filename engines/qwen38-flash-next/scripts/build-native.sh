#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
export SPARKSERVE_CUDA_IMAGE="${SPARK_ENGINE_CUDA_IMAGE:-${SPARKSERVE_CUDA_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}}"
export SPARKSERVE_RUST_IMAGE="${SPARK_ENGINE_RUST_IMAGE:-${SPARKSERVE_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}}"
exec "${REPO_ROOT}/scripts/build-qwen-native.sh"

#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
CHECKOUT="${SPARKSERVE_DEPS_DIR:-${REPO_ROOT}/third_party/_deps}/qwen38-flash-blazux"

export IMAGE="${SPARK_ENGINE_IMAGE:-qwen38-flash-dgx}"
export MODEL="${SPARK_ENGINE_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
export HF_CACHE="${SPARK_ENGINE_CACHE:-${HOME}/.cache/huggingface}"
export PORT="${SPARK_ENGINE_PORT:-18300}"
export NAME="sparkserve-qwen38-flash-oracle"
export CTX="${QWEN_FLASH_ORACLE_CONTEXT:-262144}"
export SEQS="${QWEN_FLASH_ORACLE_SEQS:-8}"
export GPU_MEM="${QWEN_FLASH_ORACLE_GPU_MEM:-0.85}"
export MTP="${QWEN_FLASH_ORACLE_MTP:-2}"

cd "${CHECKOUT}"
exec ./scripts/serve.sh

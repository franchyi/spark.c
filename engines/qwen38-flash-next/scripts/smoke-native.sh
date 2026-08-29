#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
port="${SPARK_ENGINE_PORT:-8020}"
bind="${SPARK_ENGINE_BIND:-127.0.0.1}"
export SPARKSERVE_QWEN_MODEL_ID="${SPARK_ENGINE_MODEL_ID:-${SPARKSERVE_QWEN_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}}"
exec "${REPO_ROOT}/scripts/smoke-qwen-native-api.sh" "http://${bind}:${port}"

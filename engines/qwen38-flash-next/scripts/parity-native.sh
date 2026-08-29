#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
export SPARKSERVE_QWEN_MODEL="${SPARK_ENGINE_MODEL:-${SPARKSERVE_QWEN_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}}"
exec "${REPO_ROOT}/scripts/check-qwen-oracle-parity.sh"

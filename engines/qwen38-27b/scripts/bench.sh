#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
CHECKOUT="${SPARKSERVE_DEPS_DIR:-${REPO_ROOT}/third_party/_deps}/qwen38-27b-miaai"
REVISION="751e29eb6a3057ccfd8f992f87dfc260787e05a1"

if [[ ! -x "${CHECKOUT}/bench/bench.sh" ]]; then
  echo "run 'make build' before bench" >&2
  exit 1
fi

echo "engine=qwen38-27b revision=${REVISION} checkpoint=RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead concurrency=1"
cd "${CHECKOUT}"
exec ./bench/bench.sh "${QWEN27_BENCH_LABEL:-sparkserve-oracle}"

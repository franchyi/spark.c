#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
CHECKOUT="${SPARKSERVE_DEPS_DIR:-${REPO_ROOT}/third_party/_deps}/qwen38-flash-blazux"
export PORT="${SPARK_ENGINE_PORT:-18300}"

cd "${CHECKOUT}"
exec ./scripts/smoke-test.sh "127.0.0.1:${PORT}"

#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
bind="${SPARK_ENGINE_BIND:-127.0.0.1}"
port="${SPARK_ENGINE_PORT:-8010}"
exec "${REPO_ROOT}/scripts/smoke-glm53-q2-api.sh" "http://${bind}:${port}"

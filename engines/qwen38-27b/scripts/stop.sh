#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
CHECKOUT="${SPARKSERVE_DEPS_DIR:-${REPO_ROOT}/third_party/_deps}/qwen38-27b-miaai"

if [[ ! -x "${CHECKOUT}/stop.sh" ]]; then
  echo "qwen38-27b oracle has not been fetched; nothing to stop"
  exit 0
fi

cd "${CHECKOUT}"
exec ./stop.sh

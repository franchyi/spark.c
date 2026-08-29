#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
source_root="${DS4_GLM53_Q2_ROOT:-${REPO_ROOT}/third_party/_deps/ds4-glm53-q2}"
exec "${REPO_ROOT}/scripts/build-glm53-q2.sh" "${source_root}"

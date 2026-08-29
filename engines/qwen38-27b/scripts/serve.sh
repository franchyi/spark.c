#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
CHECKOUT="${SPARKSERVE_DEPS_DIR:-${REPO_ROOT}/third_party/_deps}/qwen38-27b-miaai"

"${ENGINE_DIR}/scripts/fetch-oracle.sh"

port="${SPARK_ENGINE_PORT:-8888}"
if [[ "${port}" != "8888" ]]; then
  echo "the pinned first-version launcher requires SPARK_ENGINE_PORT=8888" >&2
  exit 64
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export QUANT="${QWEN27_QUANT:-${QUANT:-nvfp4}}"
export CPUSET="${QWEN27_CPUSET:-${CPUSET:-5-9,15-19}}"
export CONTEXT_LENGTH="${QWEN27_CONTEXT_LENGTH:-${CONTEXT_LENGTH:-262144}}"
export MAX_CONCURRENT_REQUESTS="${QWEN27_MAX_CONCURRENCY:-${MAX_CONCURRENT_REQUESTS:-10}}"
mirror_env="HF_ENDPOINT=${HF_ENDPOINT} HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
export DOCKER_ENV="${DOCKER_ENV:+${DOCKER_ENV} }${mirror_env}"

cd "${CHECKOUT}"
profile="${QWEN27_PROFILE:-resident-mtp}"
case "${profile}" in
  resident|resident-mtp|mtp)
    export IMAGE="${SPARK_ENGINE_IMAGE:-${IMAGE:-docker.1ms.run/lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6}}"
    exec ./start.sh
    ;;
  dflash2)
    export IMAGE="${SPARK_ENGINE_IMAGE:-${IMAGE:-lmsysorg/sglang:qwen38-27b-dflash2}}"
    exec ./start-dflash.sh
    ;;
  *)
    echo "QWEN27_PROFILE must be resident-mtp or dflash2" >&2
    exit 64
    ;;
esac

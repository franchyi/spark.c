#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
CHECKOUT="${FLASH_DEPS_DIR:-${REPO_ROOT}/vendor/_deps}/qwen38-flash-blazux"
IMAGE="${SPARK_ENGINE_IMAGE:-qwen38-flash-dgx}"
BASE_DIGEST="sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
BASE_NAME="vllm/vllm-openai:qwen38-flash-next"
MIRROR="${DOCKER_MIRROR:-docker.1ms.run}"
MIRROR_REF="${MIRROR}/vllm/vllm-openai:qwen38-flash-next@${BASE_DIGEST}"

"${ENGINE_DIR}/scripts/fetch-oracle.sh"
docker pull "${MIRROR_REF}"
base_id="$(docker image inspect --format '{{.Id}}' "${MIRROR_REF}")"
docker tag "${base_id}" "${BASE_NAME}"
docker build --pull=false -t "${IMAGE}" "${CHECKOUT}"

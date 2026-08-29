#!/usr/bin/env bash
set -euo pipefail

HANDOFF_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPOS_DIR="${HANDOFF_DIR}/repos"
GH_PROXY="${GH_PROXY:-https://ghfast.top/}"

clone_locked() {
  local name="$1"
  local source_url="$2"
  local revision="$3"
  local checkout="${REPOS_DIR}/${name}"
  local clone_url="${GH_PROXY%/}/${source_url}"

  if [[ -e "${checkout}" && ! -d "${checkout}/.git" ]]; then
    echo "refusing non-Git handoff path: ${checkout}" >&2
    return 1
  fi
  if [[ ! -d "${checkout}/.git" ]]; then
    git clone --filter=blob:none "${clone_url}" "${checkout}"
  fi

  local origin
  origin="$(git -C "${checkout}" remote get-url origin)"
  if [[ "${origin}" != "${clone_url}" && "${origin}" != "${source_url}" ]]; then
    echo "refusing unexpected origin for ${name}: ${origin}" >&2
    return 1
  fi
  if [[ -n "$(git -C "${checkout}" status --porcelain --untracked-files=no)" ]]; then
    echo "refusing modified reference checkout: ${checkout}" >&2
    return 1
  fi

  git -C "${checkout}" fetch --depth 1 origin "${revision}"
  git -C "${checkout}" checkout --detach --quiet FETCH_HEAD

  local actual
  actual="$(git -C "${checkout}" rev-parse HEAD)"
  if [[ "${actual}" != "${revision}" ]]; then
    echo "revision mismatch for ${name}: ${actual} != ${revision}" >&2
    return 1
  fi
  echo "${name} ${actual}"
}

mkdir -p "${REPOS_DIR}"
clone_locked \
  ds4-glm53 \
  https://github.com/antirez/ds4.git \
  a60a2a0d25137a849a101e04e86ea830a346073a
clone_locked \
  qwen38-flash-blazux \
  https://github.com/blazux/qwen3.8-Flash-DGX.git \
  d2854bfff0a0b6f46984b0941ed1db6010031295
clone_locked \
  qwen38-27b-miaai \
  https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.git \
  751e29eb6a3057ccfd8f992f87dfc260787e05a1

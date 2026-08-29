#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
DEPS_DIR="${FLASH_DEPS_DIR:-${REPO_ROOT}/vendor/_deps}"
CHECKOUT="${DEPS_DIR}/qwen38-flash-blazux"
SOURCE_URL="https://github.com/blazux/qwen3.8-Flash-DGX.git"
REVISION="d2854bfff0a0b6f46984b0941ed1db6010031295"
GH_PROXY="${GH_PROXY:-https://ghfast.top/}"
CLONE_URL="${GH_PROXY%/}/${SOURCE_URL}"

mkdir -p "${DEPS_DIR}"
if [[ ! -d "${CHECKOUT}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${CLONE_URL}" "${CHECKOUT}"
fi

origin="$(git -C "${CHECKOUT}" remote get-url origin)"
if [[ "${origin}" != "${CLONE_URL}" && "${origin}" != "${SOURCE_URL}" ]]; then
  echo "refusing unexpected origin in ${CHECKOUT}: ${origin}" >&2
  exit 1
fi

git -C "${CHECKOUT}" fetch --depth 1 origin "${REVISION}"
git -C "${CHECKOUT}" checkout --detach "${REVISION}"
actual="$(git -C "${CHECKOUT}" rev-parse HEAD)"
if [[ "${actual}" != "${REVISION}" ]]; then
  echo "oracle revision mismatch: expected ${REVISION}, got ${actual}" >&2
  exit 1
fi

echo "qwen38-flash oracle=${CHECKOUT}"
echo "revision=${actual}"

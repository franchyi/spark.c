#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
PID_FILE="${ENGINE_DIR}/.native.pid"
port="${SPARK_ENGINE_PORT:-8020}"
bind="${SPARK_ENGINE_BIND:-127.0.0.1}"

export SPARKSERVE_QWEN_MODEL="${SPARK_ENGINE_MODEL:-${SPARKSERVE_QWEN_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}}"
export SPARKSERVE_QWEN_MODEL_ID="${SPARK_ENGINE_MODEL_ID:-${SPARKSERVE_QWEN_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}}"
export SPARKSERVE_QWEN_BIND="${bind}:${port}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "native Qwen server already running as pid ${old_pid}" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

child_pid=""
cleanup() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
}
trap cleanup EXIT INT TERM

"${REPO_ROOT}/scripts/run-qwen-native.sh" &
child_pid=$!
printf '%s\n' "${child_pid}" > "${PID_FILE}"
wait "${child_pid}"

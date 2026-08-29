#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
PID_FILE="${ENGINE_DIR}/.server.pid"
model="${SPARK_ENGINE_MODEL:-/home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf}"

export GLM53_Q2_MODEL="${model}"
export GLM53_Q2_HOST="${SPARK_ENGINE_BIND:-${GLM53_Q2_HOST:-127.0.0.1}}"
export GLM53_Q2_PORT="${SPARK_ENGINE_PORT:-${GLM53_Q2_PORT:-8010}}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "GLM-5.3 Q2 server already running as pid ${old_pid}" >&2
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

"${REPO_ROOT}/scripts/run-glm53-q2.sh" "${model}" &
child_pid=$!
printf '%s\n' "${child_pid}" > "${PID_FILE}"
wait "${child_pid}"

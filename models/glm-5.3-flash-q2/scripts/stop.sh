#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PID_FILE="${ENGINE_DIR}/.server.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "GLM-5.3 Q2 pid file absent; nothing to stop"
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "removed stale GLM-5.3 Q2 pid file"
  exit 0
fi

command_line="$(ps -p "${pid}" -o command=)"
if [[ "${command_line}" != *"ds4-server"* ]]; then
  echo "refusing to stop pid ${pid}; command is not ds4-server: ${command_line}" >&2
  exit 1
fi

kill "${pid}"
rm -f "${PID_FILE}"
echo "stopped GLM-5.3 Q2 server pid ${pid}"

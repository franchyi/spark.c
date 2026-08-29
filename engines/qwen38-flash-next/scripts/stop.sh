#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PID_FILE="${ENGINE_DIR}/.native.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "native Qwen pid file absent; nothing to stop"
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "removed stale native Qwen pid file"
  exit 0
fi

command_line="$(ps -p "${pid}" -o command=)"
if [[ "${command_line}" != *"qwen_serve"* ]]; then
  echo "refusing to stop pid ${pid}; command is not qwen_serve: ${command_line}" >&2
  exit 1
fi

kill "${pid}"
rm -f "${PID_FILE}"
echo "stopped native Qwen server pid ${pid}"

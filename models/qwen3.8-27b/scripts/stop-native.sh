#!/usr/bin/env bash
set -euo pipefail

model_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
pid_file="$model_dir/.native.pid"

if [[ ! -f "$pid_file" ]]; then
  echo "native Qwen3.8-27B pid file absent; nothing to stop"
  exit 0
fi

pid=$(cat "$pid_file")
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file"
  echo "removed stale native Qwen3.8-27B pid file"
  exit 0
fi

command_line=$(ps -p "$pid" -o command=)
if [[ "$command_line" != *"q27-serve"* ]]; then
  echo "refusing to stop pid $pid; command is not q27-serve: $command_line" >&2
  exit 1
fi

kill "$pid"
rm -f "$pid_file"
echo "stopped native Qwen3.8-27B server pid $pid"

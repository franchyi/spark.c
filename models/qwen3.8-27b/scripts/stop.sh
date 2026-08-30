#!/usr/bin/env bash
set -euo pipefail

model_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd "$model_dir/../.." && pwd)
pid_file="$model_dir/.server.pid"
server="$repo_root/build/bin/q27-serve-dflash2"

if [[ ! -f "$pid_file" ]]; then
  echo "Qwen3.8-27B server is not running"
  exit 0
fi

pid=$(<"$pid_file")
if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file"
  echo "removed stale Qwen3.8-27B pid file"
  exit 0
fi
if [[ ! -e "/proc/$pid/exe" ]] ||
   [[ "$(readlink -f "/proc/$pid/exe")" != "$(readlink -f "$server")" ]]; then
  echo "refusing to stop pid $pid because it is not the Qwen3.8-27B server" >&2
  exit 1
fi

kill "$pid"
rm -f "$pid_file"
echo "stopped Qwen3.8-27B server pid $pid"

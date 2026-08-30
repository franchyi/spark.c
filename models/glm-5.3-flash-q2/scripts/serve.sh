#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
engine_dir=$(cd "$script_dir/.." && pwd)
pid_file="$engine_dir/.server.pid"
model=${1:-${HOME}/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf}
host=${2:-127.0.0.1}
port=${3:-8010}

server="$repo_root/build/glm-5.3-flash-q2/glm-server"
if [[ ! -x "$server" ]]; then
  echo "missing $server; run './spark setup glm'" >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  echo "missing GLM checkpoint: $model" >&2
  echo "run './spark setup glm'" >&2
  exit 1
fi

expected_bytes=96505816384
actual_bytes=$(stat -c %s "$model")
if [[ "$actual_bytes" != "$expected_bytes" ]]; then
  echo "GLM-5.3 Q2 size mismatch: $actual_bytes != $expected_bytes" >&2
  exit 1
fi

minimum_available_kib=$((110 * 1024 * 1024))
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < minimum_available_kib )); then
  available_gib=$(awk -v kib="$available_kib" 'BEGIN {printf "%.1f", kib / 1048576}')
  echo "resident Q2 requires about 110 GiB MemAvailable before startup; found ${available_gib} GiB" >&2
  echo "stop nonessential services before starting the resident Q2 model" >&2
  exit 1
fi

context=2048
max_tokens=128

if [[ -f "$pid_file" ]]; then
  old_pid=$(cat "$pid_file")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "GLM-5.3 Q2 server already running as pid $old_pid" >&2
    exit 1
  fi
  rm -f "$pid_file"
fi

child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}
trap cleanup EXIT INT TERM

"$server" \
  --cuda \
  -c "$context" \
  -n "$max_tokens" \
  -m "$model" \
  --host "$host" \
  --port "$port" &
child_pid=$!
printf '%s\n' "$child_pid" >"$pid_file"
wait "$child_pid"

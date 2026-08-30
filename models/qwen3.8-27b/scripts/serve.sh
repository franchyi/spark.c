#!/usr/bin/env bash
set -euo pipefail

# Foreground, single-slot launcher for the model-specific DFlash2 engine.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
model_dir=$(cd "$script_dir/.." && pwd)
server="$repo_root/build/bin/q27-serve-dflash2"
pid_file="$model_dir/.server.pid"

target=${1:-${HOME}/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}
draft=${2:-${HOME}/models/z-lab/Qwen3.8-27B-DFlash2}
host=${3:-127.0.0.1}
port=${4:-30000}
sidecar="$target/.spark.c/q27-scales-v1.bin"
bind="$host:$port"
model_id=spark/Qwen3.8-27B-DFlash2
capacity=16384

for path in "$target" "$sidecar" "$draft"; do
  if [[ ! -e "$path" ]]; then
    echo "required model path does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -x "$server" ]]; then
  echo "Q27 DFlash2 server is not built; run './spark setup qwen27'" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$repo_root/build/q27:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export Q27_GDN_C427_AOT=1
export Q27_GDN_C427_AOT_DIR=${HOME}/.cache/spark-c/aot/q27/c427
export Q27_PREFILL_M8192=1
export Q27_DFLASH2_T8_GDN=1

server_pid() {
  local pid=$1
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -e "/proc/$pid/exe" ]] || return 1
  [[ "$(readlink -f "/proc/$pid/exe")" == "$(readlink -f "$server")" ]]
}

if [[ -f "$pid_file" ]]; then
  old_pid=$(<"$pid_file")
  if kill -0 "$old_pid" 2>/dev/null; then
    if server_pid "$old_pid"; then
      echo "Q27 DFlash2 server already running as pid $old_pid" >&2
    else
      echo "refusing to reuse active pid $old_pid from $pid_file" >&2
    fi
    exit 1
  fi
  rm -f "$pid_file"
fi

child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    if server_pid "$child_pid"; then
      kill "$child_pid" 2>/dev/null || true
      wait "$child_pid" 2>/dev/null || true
    else
      echo "refusing to stop reused pid $child_pid" >&2
    fi
  fi
  rm -f "$pid_file"
}
trap cleanup EXIT INT TERM

"$server" "$target" "$sidecar" "$draft" \
  "$bind" "$model_id" "$capacity" &
child_pid=$!
printf '%s\n' "$child_pid" >"$pid_file"
wait "$child_pid"

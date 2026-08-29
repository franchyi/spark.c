#!/usr/bin/env bash
set -euo pipefail

# Foreground, single-slot launcher for the native model-specific DFlash2 server.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
model_dir=$(cd "$script_dir/.." && pwd)
server="$repo_root/build/bin/q27-serve-dflash2"
pid_file="$model_dir/.dflash2-native.pid"

target=${SPARK_ENGINE_MODEL:?set SPARK_ENGINE_MODEL to the target Qwen3.8-27B snapshot}
sidecar=${SPARK_ENGINE_SIDECAR:?set SPARK_ENGINE_SIDECAR to q27-scales-v1.bin}
draft=${SPARK_DFLASH2_MODEL:?set SPARK_DFLASH2_MODEL to the pinned draft snapshot}
bind=${SPARK_ENGINE_BIND:-0.0.0.0:${SPARK_ENGINE_PORT:-30000}}
model_id=${SPARK_ENGINE_MODEL_ID:-spark/Qwen3.8-27B-DFlash2}
capacity=${SPARK_ENGINE_CONTEXT_CAPACITY:-16384}

for path in "$target" "$sidecar" "$draft"; do
  if [[ ! -e "$path" ]]; then
    echo "required model path does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -x "$server" ]]; then
  echo "native DFlash2 server is not built: $server" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$repo_root/build/q27:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

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
      echo "native DFlash2 server already running as pid $old_pid" >&2
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

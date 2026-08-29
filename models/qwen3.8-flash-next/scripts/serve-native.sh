#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
engine_dir=$(cd "$script_dir/.." && pwd)
pid_file="$engine_dir/.native.pid"
model_root=${FLASH_QWEN_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}
bind=${FLASH_QWEN_BIND:-127.0.0.1:8020}
model_id=${FLASH_QWEN_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
binary="$repo_root/build/bin/qwen_serve"

for required in \
  "$binary" \
  "$repo_root/build/flash-next/libflash-fabric.so" \
  "$repo_root/build/flash-next/libflash-qwen-runtime.so" \
  "$repo_root/build/flash-next/libtvm_ffi.so" \
  "$model_root/config.json" \
  "$model_root/tokenizer.json" \
  "$model_root/.spark.c/ple.ssple"; do
  if [[ ! -e "$required" ]]; then
    echo "missing native Qwen artifact: $required" >&2
    exit 1
  fi
done

if [[ -f "$pid_file" ]]; then
  old_pid=$(cat "$pid_file")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "native Flash-Next server already running as pid $old_pid" >&2
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

export LD_LIBRARY_PATH="$repo_root/build/flash-next:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$binary" "$model_root" "$bind" "$model_id" &
child_pid=$!
printf '%s\n' "$child_pid" >"$pid_file"
wait "$child_pid"

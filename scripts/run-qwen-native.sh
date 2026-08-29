#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
model_root=${SPARKSERVE_QWEN_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}
bind=${SPARKSERVE_QWEN_BIND:-127.0.0.1:8020}
model_id=${SPARKSERVE_QWEN_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
binary="$repo_root/build/bin/qwen_serve"

for required in \
  "$binary" \
  "$repo_root/build/libsparkserve-fabric.so" \
  "$repo_root/build/libsparkserve-qwen-runtime.so" \
  "$repo_root/build/libsparkserve-qsa.so" \
  "$repo_root/build/libtvm_ffi.so" \
  "$model_root/config.json" \
  "$model_root/tokenizer.json" \
  "$model_root/.sparkserve/ple.ssple"; do
  if [[ ! -e "$required" ]]; then
    echo "missing native Qwen artifact: $required" >&2
    exit 1
  fi
done

export LD_LIBRARY_PATH="$repo_root/build:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$binary" "$model_root" "$bind" "$model_id"

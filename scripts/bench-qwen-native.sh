#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
model_root=${SPARKSERVE_QWEN_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}
input_token=${SPARKSERVE_QWEN_INPUT_TOKEN:-9707}
steps=${SPARKSERVE_QWEN_BENCH_STEPS:-2}

for binary in qwen_first_token qwen_decode; do
  if [[ ! -x "$repo_root/build/bin/$binary" ]]; then
    echo "missing native Qwen benchmark binary: $repo_root/build/bin/$binary" >&2
    exit 1
  fi
done

export LD_LIBRARY_PATH="$repo_root/build:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "=== native Qwen one-token replay ==="
/usr/bin/time -v "$repo_root/build/bin/qwen_first_token" "$model_root" "$input_token" 1
echo "=== native Qwen persistent continuation ==="
/usr/bin/time -v "$repo_root/build/bin/qwen_decode" "$model_root" "$input_token" "$steps"

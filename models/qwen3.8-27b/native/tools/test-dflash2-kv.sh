#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
build_dir="${repo_root}/build/q27"
kv_library="${build_dir}/libq27-dflash2-kv.so"
attention_library="${build_dir}/libq27-dflash2-attention.so"
binary="${build_dir}/q27-dflash2-kv-smoke"

"${script_dir}/build-dflash2-attention.sh" "${attention_library}"
"${script_dir}/build-dflash2-kv.sh" "${kv_library}"
"${nvcc}" -O2 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${script_dir}/q27_dflash2_kv_smoke.cu" \
  -L"${build_dir}" -lq27-dflash2-kv -lq27-dflash2-attention \
  -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -o "${binary}"

"${binary}"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
flashinfer_dir="${repo_root}/vendor/_deps/flashinfer"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
build_dir="${repo_root}/build/q27"
library="${build_dir}/libq27-prefill-attention.so"
binary="${build_dir}/q27-prefill-attention-smoke"

"${script_dir}/build-q27-prefill-attention.sh" "${library}"
"${nvcc}" -O2 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  -I"${flashinfer_dir}/include" \
  -I"${flashinfer_dir}/3rdparty/cutlass/include" \
  "${script_dir}/q27_prefill_attention_smoke.cu" \
  -L"${build_dir}" -lq27-prefill-attention -lcudart \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -o "${binary}"

"${binary}"

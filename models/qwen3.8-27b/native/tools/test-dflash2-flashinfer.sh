#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
flashinfer_dir="${repo_root}/vendor/_deps/flashinfer"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
build_dir="${repo_root}/build/q27"
library="${build_dir}/libq27-dflash2-flashinfer.so"
binary="${build_dir}/q27-dflash2-flashinfer-smoke"

"${script_dir}/build-dflash2-flashinfer.sh" "${library}"
"${nvcc}" -O2 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  -I"${flashinfer_dir}/include" \
  -I"${flashinfer_dir}/3rdparty/cutlass/include" \
  "${script_dir}/q27_dflash2_flashinfer_smoke.cu" \
  -L"${build_dir}" -lq27-dflash2-flashinfer -lcudart \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -o "${binary}"

"${binary}"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
build_dir="${repo_root}/build/q27"
library="${build_dir}/libq27-dflash2-control.so"
binary="${build_dir}/q27-dflash2-control-smoke"

"${script_dir}/build-dflash2-control.sh" "${library}"
"${nvcc}" -O2 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${script_dir}/q27_dflash2_control_smoke.cu" \
  -L"${build_dir}" -lq27-dflash2-control \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -o "${binary}"

"${binary}"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
build_dir="${repo_root}/build/q27"
library="${build_dir}/libq27-verify.so"
binary="${build_dir}/q27-verify-smoke"

bash "${script_dir}/build-q27-verify.sh" "${library}"
"${nvcc}" -O2 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${script_dir}/q27_verify_smoke.cu" \
  -L"${build_dir}" -lq27-verify \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -o "${binary}"

"${binary}"

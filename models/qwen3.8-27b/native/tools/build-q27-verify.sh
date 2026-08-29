#!/usr/bin/env bash
set -euo pipefail

# Isolated Spark-only build.  Keep this separate from the active model/MLP
# targets until q27_verify_forward_t8 has its complete M=8 kernel closure.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-verify.so}"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${native_dir}/cuda/q27_verify.cu" \
  -lcudart -o "${output}"

echo "built ${output}"

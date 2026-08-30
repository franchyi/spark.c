#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${engine_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-dflash2-conv.so}"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${engine_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${engine_dir}/cuda/q27_dflash2_conv.cu" \
  -lcublas -lcudart -o "${output}"

echo "built ${output}"

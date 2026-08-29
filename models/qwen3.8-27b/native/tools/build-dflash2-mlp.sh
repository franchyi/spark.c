#!/usr/bin/env bash
set -euo pipefail

# Isolated Spark-only build while the DFlash2 coordinator is evolving.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-dflash2-mlp.so}"
conv_library="${repo_root}/build/q27/libq27-dflash2-conv.so"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
"${script_dir}/build-dflash2-conv.sh" "${conv_library}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${native_dir}/cuda/q27_dflash2_mlp.cu" \
  -L"${repo_root}/build/q27" -lq27-dflash2-conv \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -lcublas -lcudart -o "${output}"

echo "built ${output}"

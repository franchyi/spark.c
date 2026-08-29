#!/usr/bin/env bash
set -euo pipefail

# Spark-only build for the allocation-free DFlash2 projection/model
# coordinator. Fixed attention is linked; the fixed MLP dependency remains.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-dflash2-model.so}"
attention_library="${repo_root}/build/q27/libq27-dflash2-attention.so"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
"${script_dir}/build-dflash2-attention.sh" "${attention_library}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${native_dir}/cuda/q27_dflash2_model.cu" \
  -L"${repo_root}/build/q27" -lq27-dflash2-attention \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -lcublas -lcudart -o "${output}"

echo "built ${output}"

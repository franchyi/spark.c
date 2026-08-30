#!/usr/bin/env bash
set -euo pipefail

# Spark-only build for the allocation-free DFlash2 projection/model
# coordinator. Fixed attention and grouped-conv MLP are linked directly.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${engine_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-dflash2-model.so}"
attention_library="${repo_root}/build/q27/libq27-dflash2-attention.so"
mlp_library="${repo_root}/build/q27/libq27-dflash2-mlp.so"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
"${script_dir}/build-dflash2-attention.sh" "${attention_library}"
"${script_dir}/build-dflash2-mlp.sh" "${mlp_library}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${engine_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${engine_dir}/cuda/q27_dflash2_model.cu" \
  -L"${repo_root}/build/q27" -lq27-dflash2-attention -lq27-dflash2-mlp \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -lcublas -lcudart -o "${output}"

echo "built ${output}"

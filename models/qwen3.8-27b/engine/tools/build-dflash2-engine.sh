#!/usr/bin/env bash
set -euo pipefail

# Spark-only build for the fixed batch-one target+draft DFlash2 owner.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${engine_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-dflash2-engine.so}"
library_dir="${repo_root}/build/q27"
target_library="${library_dir}/libq27-model.so"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
if [[ ! -f "${target_library}" ]]; then
  echo "Target capsule must be built first: ${target_library}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
bash "${script_dir}/build-dflash2-control.sh" \
  "${library_dir}/libq27-dflash2-control.so"
bash "${script_dir}/build-dflash2-model.sh" \
  "${library_dir}/libq27-dflash2-model.so"
bash "${script_dir}/build-dflash2-topk.sh" \
  "${library_dir}/libq27-dflash2-topk.so"
bash "${script_dir}/build-dflash2-kv.sh" \
  "${library_dir}/libq27-dflash2-kv.so"

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${engine_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  -Xlinker=-z -Xlinker=defs \
  "${engine_dir}/cuda/q27_dflash2_engine.cu" \
  -L"${library_dir}" -lq27-model -lq27-dflash2-control \
  -lq27-dflash2-model -lq27-dflash2-attention \
  -lq27-dflash2-flashinfer -lq27-dflash2-topk -lq27-dflash2-kv \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' \
  -lcublas -lcudart -o "${output}"

echo "built ${output}"

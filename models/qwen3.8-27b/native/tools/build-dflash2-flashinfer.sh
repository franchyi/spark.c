#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
flashinfer_dir="${repo_root}/vendor/_deps/flashinfer"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-dflash2-flashinfer.so}"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
if [[ ! -f "${flashinfer_dir}/include/flashinfer/attention/prefill.cuh" ]]; then
  echo "Pinned FlashInfer headers missing: ${flashinfer_dir}" >&2
  exit 1
fi
fastdiv_header="${CCCL_FASTDIV_HEADER:-}"
if [[ -z "${fastdiv_header}" ]]; then
  fastdiv_header="$(find \
    /usr/local/lib/python3.12/dist-packages/flashinfer/data/cccl \
    -path '*/libcudacxx/include/cuda/__cmath/fast_modulo_division.h' \
    -print -quit 2>/dev/null)"
fi
if [[ -z "${fastdiv_header}" || ! -f "${fastdiv_header}" ]]; then
  echo "Pinned FlashInfer CCCL fast-modulo header is missing" >&2
  exit 1
fi
cccl_root="${fastdiv_header%%/libcudacxx/include/*}"

mkdir -p "$(dirname "${output}")"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${native_dir}/include" \
  -I"${cccl_root}/libcudacxx/include" \
  -I"${cccl_root}/cub" -I"${cccl_root}/thrust" \
  -I"${flashinfer_dir}/include" \
  -I"${flashinfer_dir}/3rdparty/cutlass/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${native_dir}/cuda/q27_dflash2_flashinfer.cu" \
  -lcudart -o "${output}"

echo "built ${output}"

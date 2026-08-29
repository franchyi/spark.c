#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
build_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
output="${1:-${build_dir}/libq27-prefill-attention-layer.so}"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
for dependency in \
  "${build_dir}/libq27-prefill-core.so" \
  "${build_dir}/libq27-prefill-fp8.so" \
  "${build_dir}/libq27-prefill-attention.so"; do
  if [[ ! -f "${dependency}" ]]; then
    echo "Q27 target layer dependency is missing: ${dependency}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${output}")"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${native_dir}/cuda/q27_prefill_attention_layer.cu" \
  -L"${build_dir}" -lq27-prefill-core -lq27-prefill-fp8 \
  -lq27-prefill-attention -lcublasLt -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' -o "${output}"

if ldd "${output}" | grep -q 'not found'; then
  ldd "${output}"
  exit 1
fi
echo "built ${output}"

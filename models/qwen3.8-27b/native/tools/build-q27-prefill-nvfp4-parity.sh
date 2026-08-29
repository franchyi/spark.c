#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
binary="${output_dir}/q27-prefill-nvfp4-parity"

for library in libq27-prefill-nvfp4.so libq27-prefill-mlp.so libq27-nvfp4.so libq27-kernels.so; do
  if [[ ! -f "${output_dir}/${library}" ]]; then
    echo "required library is missing: ${output_dir}/${library}" >&2
    exit 1
  fi
done

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_prefill_nvfp4_parity.cu" \
  -L"${output_dir}" -lq27-prefill-mlp -lq27-prefill-nvfp4 \
  -lq27-nvfp4 -lq27-kernels -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${binary}"

if ldd "${binary}" | grep -q 'not found'; then
  ldd "${binary}"
  exit 1
fi
echo "built ${binary}"

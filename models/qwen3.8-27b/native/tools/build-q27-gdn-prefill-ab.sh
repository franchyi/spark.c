#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_GDN_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"

mkdir -p "${output_dir}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo --shared \
  -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker=libq27-gdn-prefill-ab.so \
  "${native_dir}/cuda/q27_gdn_prefill_ab.cu" -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' \
  -o "${output_dir}/libq27-gdn-prefill-ab.so"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_gdn_prefill_ab_bench.cu" \
  -L"${output_dir}" -lq27-gdn-prefill-ab -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' \
  -o "${output_dir}/q27-gdn-prefill-ab-bench"
ldd "${output_dir}/libq27-gdn-prefill-ab.so" | grep -q 'not found' && exit 1 || true
ldd "${output_dir}/q27-gdn-prefill-ab-bench" | grep -q 'not found' && exit 1 || true
echo "built ${output_dir}/libq27-gdn-prefill-ab.so"
echo "built ${output_dir}/q27-gdn-prefill-ab-bench"

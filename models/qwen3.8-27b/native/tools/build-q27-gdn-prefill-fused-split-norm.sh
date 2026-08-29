#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_GDN_FUSED_OUTPUT_DIR:-${repo_root}/build/q27}"
mkdir -p "${output_dir}"
for library in q27-gdn-prefill q27-gdn-prefill-wy; do
  if [[ ! -f "${output_dir}/lib${library}.so" ]]; then
    echo "missing reference dependency ${output_dir}/lib${library}.so" >&2
    exit 1
  fi
done
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo --shared \
  -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker=libq27-gdn-prefill-fused-split-norm.so \
  "${native_dir}/cuda/q27_gdn_prefill_fused_split_norm.cu" \
  -lcudart \
  -o "${output_dir}/libq27-gdn-prefill-fused-split-norm.so"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_gdn_prefill_fused_split_norm_bench.cu" \
  -L"${output_dir}" -lq27-gdn-prefill-fused-split-norm \
  -lq27-gdn-prefill -lq27-gdn-prefill-wy -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' \
  -o "${output_dir}/q27-gdn-prefill-fused-split-norm-bench"
if ldd "${output_dir}/libq27-gdn-prefill-fused-split-norm.so" | \
    grep -q 'not found'; then
  exit 1
fi
if ldd "${output_dir}/q27-gdn-prefill-fused-split-norm-bench" | \
    grep -q 'not found'; then
  exit 1
fi
echo "built ${output_dir}/libq27-gdn-prefill-fused-split-norm.so"
echo "built ${output_dir}/q27-gdn-prefill-fused-split-norm-bench"

#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_GDN_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
for library in q27-gdn-prefill q27-gdn-prefill-ab q27-gdn-prefill-wy q27-prefill-fp8; do
  if [[ ! -f "${output_dir}/lib${library}.so" ]]; then
    echo "missing dependency ${output_dir}/lib${library}.so" >&2
    exit 1
  fi
done
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo --shared \
  -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker=libq27-gdn-prefill-sublayer.so \
  "${native_dir}/cuda/q27_gdn_prefill_sublayer.cu" \
  -L"${output_dir}" -lq27-gdn-prefill -lq27-gdn-prefill-ab \
  -lq27-gdn-prefill-wy -lq27-prefill-fp8 -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' \
  -o "${output_dir}/libq27-gdn-prefill-sublayer.so"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_gdn_prefill_sublayer_bench.cu" \
  -L"${output_dir}" -lq27-gdn-prefill-sublayer -lq27-prefill-fp8 \
  -lcublas -lcudart -Xlinker=-rpath -Xlinker='$ORIGIN' \
  -o "${output_dir}/q27-gdn-prefill-sublayer-bench"
if ldd "${output_dir}/libq27-gdn-prefill-sublayer.so" | grep -q 'not found'; then exit 1; fi
if ldd "${output_dir}/q27-gdn-prefill-sublayer-bench" | grep -q 'not found'; then exit 1; fi
echo "built ${output_dir}/libq27-gdn-prefill-sublayer.so"
echo "built ${output_dir}/q27-gdn-prefill-sublayer-bench"

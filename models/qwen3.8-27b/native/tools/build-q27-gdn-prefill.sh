#!/usr/bin/env bash
set -euo pipefail

# Spark-only isolated build. This capsule has no Python, Triton, Torch, JIT,
# or framework dependency in its runtime closure.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_GDN_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
library="${output_dir}/libq27-gdn-prefill.so"
benchmark="${output_dir}/q27-gdn-prefill-bench"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${library}")" \
  "${native_dir}/cuda/q27_gdn_prefill.cu" \
  -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${library}"

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_gdn_prefill_bench.cu" \
  -L"${output_dir}" -lq27-gdn-prefill -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${benchmark}"

if ldd "${library}" | grep -q 'not found'; then
  ldd "${library}"
  exit 1
fi
if ldd "${benchmark}" | grep -q 'not found'; then
  ldd "${benchmark}"
  exit 1
fi

echo "built ${library}"
echo "built ${benchmark}"

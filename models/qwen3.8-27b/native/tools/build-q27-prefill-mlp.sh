#!/usr/bin/env bash
set -euo pipefail

# Spark-only build for the fixed batched dense-MLP coordinator.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
dependency="${output_dir}/libq27-prefill-nvfp4.so"
library="${output_dir}/libq27-prefill-mlp.so"
benchmark="${output_dir}/q27-prefill-mlp-bench"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
if [[ ! -f "${dependency}" ]]; then
  echo "build the Q27 batched NVFP4 capsule first: ${dependency}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${library}")" \
  "${native_dir}/cuda/q27_prefill_mlp.cu" \
  -L"${output_dir}" -lq27-prefill-nvfp4 -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${library}"

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_prefill_mlp_bench.cu" \
  -L"${output_dir}" -lq27-prefill-mlp -lq27-prefill-nvfp4 -lcudart \
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

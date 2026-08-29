#!/usr/bin/env bash
set -euo pipefail

# Spark-only isolated build: no Torch, Python, JIT, or M=1 source changes.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
library="${output_dir}/libq27-prefill-fp8.so"
benchmark="${output_dir}/q27-prefill-fp8-bench"
parity="${output_dir}/q27-prefill-fp8-parity"
baseline_library="${output_dir}/libq27-kernels.so"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
if [[ ! -f "${baseline_library}" ]]; then
  echo "M=1 Q27 baseline is missing: ${baseline_library}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${library}")" \
  "${native_dir}/cuda/q27_prefill_fp8.cu" \
  -lcublasLt -lcublas -lcudart -o "${library}"

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_prefill_fp8_bench.cu" \
  -L"${output_dir}" -lq27-prefill-fp8 -lq27-kernels \
  -lcublasLt -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${benchmark}"

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_prefill_fp8_parity.cu" \
  -L"${output_dir}" -lq27-prefill-fp8 -lcublasLt -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${parity}"

if ldd "${library}" | grep -q 'not found'; then
  ldd "${library}"
  exit 1
fi
if ldd "${benchmark}" | grep -q 'not found'; then
  ldd "${benchmark}"
  exit 1
fi
if ldd "${parity}" | grep -q 'not found'; then
  ldd "${parity}"
  exit 1
fi

echo "built ${library}"
echo "built ${benchmark}"
echo "built ${parity}"

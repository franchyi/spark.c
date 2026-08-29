#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
library="${output_dir}/libq27-prefill-core.so"
fixture="${output_dir}/q27-prefill-core-fixture"
baseline="${output_dir}/libq27-kernels.so"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
if [[ ! -f "${baseline}" ]]; then
  echo "M=1 Q27 baseline is missing: ${baseline}" >&2
  exit 1
fi
mkdir -p "${output_dir}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${library}")" \
  "${native_dir}/cuda/q27_prefill_core.cu" -lcudart -o "${library}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_prefill_core_fixture.cu" \
  -L"${output_dir}" -lq27-prefill-core -lq27-kernels -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${fixture}"
if ldd "${library}" | grep -q 'not found'; then
  ldd "${library}"
  exit 1
fi
if ldd "${fixture}" | grep -q 'not found'; then
  ldd "${fixture}"
  exit 1
fi
echo "built ${library}"
echo "built ${fixture}"

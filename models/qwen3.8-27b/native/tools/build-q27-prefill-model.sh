#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
library="${output_dir}/libq27-prefill-model.so"
benchmark="${output_dir}/q27-prefill-model-bench"
benchmark_full="${output_dir}/q27-prefill-model-bench-valid128"

dependencies=(
  q27-gdn-prefill-layer q27-prefill-attention-layer q27-prefill-mlp
  q27-prefill-core q27-prefill-fp8 q27-prefill-nvfp4
  q27-lm-head-bf16 q27-kernels
)
for dependency in "${dependencies[@]}"; do
  [[ -f "${output_dir}/lib${dependency}.so" ]] || {
    echo "missing ${output_dir}/lib${dependency}.so" >&2
    exit 1
  }
done

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo --shared \
  -Xcompiler=-fPIC -I"${native_dir}/include" \
  -Xlinker=-soname -Xlinker=libq27-prefill-model.so \
  "${native_dir}/cuda/q27_prefill_model.cu" \
  -L"${output_dir}" -lq27-gdn-prefill-layer \
  -lq27-prefill-attention-layer -lq27-prefill-mlp \
  -lq27-prefill-core -lq27-prefill-fp8 -lq27-lm-head-bf16 \
  -lq27-kernels -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${library}"

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_prefill_model_bench.cu" \
  -L"${output_dir}" -lq27-prefill-model -lq27-gdn-prefill-layer \
  -lq27-prefill-attention-layer -lq27-prefill-mlp \
  -lq27-prefill-nvfp4 -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${benchmark}"

"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -DQ27_PREFILL_MODEL_FIXTURE_VALID=128 -I"${native_dir}/include" \
  "${native_dir}/fixtures/q27_prefill_model_bench.cu" \
  -L"${output_dir}" -lq27-prefill-model -lq27-gdn-prefill-layer \
  -lq27-prefill-attention-layer -lq27-prefill-mlp \
  -lq27-prefill-nvfp4 -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${benchmark_full}"

if ldd "${library}" | grep -q 'not found'; then ldd "${library}"; exit 1; fi
if ldd "${benchmark}" | grep -q 'not found'; then ldd "${benchmark}"; exit 1; fi
if ldd "${benchmark_full}" | grep -q 'not found'; then ldd "${benchmark_full}"; exit 1; fi
echo "built ${library}"
echo "built ${benchmark}"
echo "built ${benchmark_full}"

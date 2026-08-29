#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
build_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
library="${build_dir}/libq27-prefill-attention-layer.so"
binary="${build_dir}/q27-prefill-attention-layer-fixture"

"${script_dir}/build-q27-prefill-attention-layer.sh" "${library}"
"${nvcc}" -O2 -std=c++20 -arch=sm_121a -lineinfo \
  -I"${native_dir}/include" \
  "${script_dir}/q27_prefill_attention_layer_fixture.cu" \
  -L"${build_dir}" -lq27-prefill-attention-layer \
  -lq27-prefill-core -lq27-prefill-fp8 -lq27-prefill-attention \
  -lcublasLt -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='${ORIGIN}' -o "${binary}"

"${binary}"
if [[ $# -eq 1 ]]; then
  "${binary}" --real "$1"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [ATTENTION_LAYER0_CHECKPOINT_LAYER3_FIXTURE_DIR]" >&2
  exit 2
fi

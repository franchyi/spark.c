#!/usr/bin/env bash
set -euo pipefail

# Spark-only experimental build. Python/CuTe DSL is used only to export the
# pinned object; the resulting shared library has a raw C ABI and no Python,
# Torch, FlashInfer dispatcher, or JIT runtime dependency.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
python_bin="${Q27_AOT_PYTHON:-python3}"
output_dir="${Q27_GDN_PREFILL_SM121_OUTPUT_DIR:-${repo_root}/build/q27}"
aot_dir="${Q27_GDN_PREFILL_SM121_AOT_DIR:-${repo_root}/build/q27-aot/gdn-prefill-sm121}"
artifact="${aot_dir}/q27_gdn_prefill_sm121_bf16_io_fp32_state_h16_hv48_d128.o"
library="${output_dir}/libq27-gdn-prefill-flashinfer-sm121.so"
exporter="${repo_root}/vendor/tools/export-q27-gdn-prefill-sm121-aot.py"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
if [[ ! -f "${artifact}" ]]; then
  if [[ "${Q27_SKIP_AOT_EXPORT:-0}" == "1" ]]; then
    echo "missing pinned SM121 GDN-prefill object: ${artifact}" >&2
    exit 1
  fi
  mkdir -p "${aot_dir}"
  PYTHONPATH="${repo_root}/vendor/_deps/flashinfer${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" "${exporter}" "${aot_dir}"
fi
if [[ ! -f "${aot_dir}/meta.json" ]]; then
  echo "missing SM121 GDN-prefill AOT metadata" >&2
  exit 1
fi

tvm_lib="${TVM_FFI_LIBRARY:-}"
if [[ -z "${tvm_lib}" ]]; then
  tvm_lib="$(find /usr/local/lib /usr/lib -type f -name 'libtvm_ffi.so*' \
    -print -quit 2>/dev/null)"
fi
dialect_archive="${CUDA_DIALECT_ARCHIVE:-}"
if [[ -z "${dialect_archive}" ]]; then
  dialect_archive="$(find /usr/local/lib /usr/lib -type f \
    -name libcuda_dialect_runtime_static.a -print -quit 2>/dev/null)"
fi
if [[ -z "${tvm_lib}" || ! -f "${tvm_lib}" ||
      -z "${dialect_archive}" || ! -f "${dialect_archive}" ]]; then
  echo "cannot locate pinned TVM-FFI/CUDA-dialect dependencies" >&2
  exit 1
fi
tvm_root="$(dirname "$(dirname "${tvm_lib}")")"

mkdir -p "${output_dir}"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  --shared -Xcompiler=-fPIC \
  -I"${native_dir}/include" -I"${tvm_root}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${library}")" \
  "${native_dir}/cuda/q27_gdn_prefill_flashinfer_sm121.cc" \
  "${artifact}" "${dialect_archive}" \
  -L"${tvm_root}/lib" -ltvm_ffi -lcuda -lcudart -ldl \
  -Xcompiler=-pthread -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${library}"

if ldd "${library}" | grep -q 'not found'; then
  ldd "${library}"
  exit 1
fi
echo "built ${library}"

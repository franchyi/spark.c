#!/usr/bin/env bash
set -euo pipefail

# Spark-only isolated build. The pinned CuTe AOT objects are M-symbolic and
# are reused verbatim; this script does not invoke Python or JIT compilation.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output="${1:-${repo_root}/build/q27/libq27-verify-nvfp4.so}"
flashinfer_root="${repo_root}/vendor/_deps/flashinfer"
aot_root="${Q27_AOT_ROOT:-${repo_root}/build/q27-aot/quantize}"
k5120_aot="${aot_root}/swizzled_bfloat16_k5120_sf0_pdl0.o"
k17408_aot="${aot_root}/swizzled_bfloat16_k17408_sf0_pdl0.o"

if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi
if [[ ! -f "${k5120_aot}" || ! -f "${k17408_aot}" ]]; then
  echo "pinned Q27 symbolic-M NVFP4 AOT objects are unavailable under ${aot_root}" >&2
  exit 1
fi
if [[ "${aot_root}" == "${repo_root}/build/q27-aot/quantize" ]]; then
  read -r k5120_hash _ < <(sha256sum "${k5120_aot}")
  read -r k17408_hash _ < <(sha256sum "${k17408_aot}")
  if [[ "${k5120_hash}" != \
          "f9ce016361014c4f54f7619d40635a5d599bd8ec7d58a514d6fed7e921006039" ||
        "${k17408_hash}" != \
          "d382df5330a614ee72c3090a9e238b8e8216e763b4d1b2343bd926d217b39afa" ]]; then
    echo "Q27 NVFP4 AOT checksum mismatch" >&2
    exit 1
  fi
fi
if [[ ! -d "${flashinfer_root}/include" ||
      ! -d "${flashinfer_root}/3rdparty/cutlass/include" ]]; then
  echo "pinned FlashInfer/CUTLASS checkout is unavailable" >&2
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
fastdiv_header="${CCCL_FASTDIV_HEADER:-}"
if [[ -z "${fastdiv_header}" ]]; then
  fastdiv_header="$(find \
    /usr/local/lib/python3.12/dist-packages/flashinfer/data/cccl \
    -path '*/libcudacxx/include/cuda/__cmath/fast_modulo_division.h' \
    -print -quit 2>/dev/null)"
fi
if [[ -z "${tvm_lib}" || ! -f "${tvm_lib}" ||
      -z "${dialect_archive}" || ! -f "${dialect_archive}" ||
      -z "${fastdiv_header}" || ! -f "${fastdiv_header}" ]]; then
  echo "cannot locate pinned TVM-FFI/CUDA-dialect/CCCL dependencies" >&2
  exit 1
fi
tvm_root="$(dirname "$(dirname "${tvm_lib}")")"
cccl_root="${fastdiv_header%%/libcudacxx/include/*}"

mkdir -p "$(dirname "${output}")"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo \
  -D__CUDA_ARCH_SPECIFIC__ --expt-relaxed-constexpr \
  -diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20015 \
  -diag-suppress 2908 --shared -Xcompiler=-fPIC \
  -I"${native_dir}/include" \
  -I"${cccl_root}/libcudacxx/include" \
  -I"${cccl_root}/cub" -I"${cccl_root}/thrust" \
  -I"${flashinfer_root}/include" \
  -I"${flashinfer_root}/3rdparty/cutlass/include" \
  -I"${flashinfer_root}/3rdparty/cutlass/tools/util/include" \
  -I"${tvm_root}/include" \
  -Xlinker=-soname -Xlinker="$(basename "${output}")" \
  "${native_dir}/cuda/q27_verify_nvfp4.cu" \
  "${k5120_aot}" "${k17408_aot}" "${dialect_archive}" \
  -L"${tvm_root}/lib" -ltvm_ffi -lcuda -lcudart -ldl \
  -Xcompiler=-pthread -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${output}"

echo "built ${output}"

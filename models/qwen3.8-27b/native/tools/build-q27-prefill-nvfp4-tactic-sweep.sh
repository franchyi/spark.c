#!/usr/bin/env bash
set -euo pipefail

# Spark-only development build.  This creates a standalone sweep executable;
# it does not rebuild or alter the production prefill NVFP4 library.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
native_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${native_dir}/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda/bin/nvcc}"
output_dir="${Q27_PREFILL_OUTPUT_DIR:-${repo_root}/build/q27}"
binary="${output_dir}/q27-prefill-nvfp4-tactic-sweep"
flashinfer_root="${repo_root}/vendor/_deps/flashinfer"

for library in libq27-prefill-nvfp4.so libq27-prefill-mlp.so; do
  if [[ ! -f "${output_dir}/${library}" ]]; then
    echo "required accepted production library is missing: ${output_dir}/${library}" >&2
    exit 1
  fi
done
if [[ ! -x "${nvcc}" ]]; then
  echo "NVCC is not executable: ${nvcc}" >&2
  exit 1
fi

fastdiv_header="${CCCL_FASTDIV_HEADER:-}"
if [[ -z "${fastdiv_header}" ]]; then
  fastdiv_header="$(find \
    /usr/local/lib/python3.12/dist-packages/flashinfer/data/cccl \
    -path '*/libcudacxx/include/cuda/__cmath/fast_modulo_division.h' \
    -print -quit 2>/dev/null)"
fi
if [[ -z "${fastdiv_header}" || ! -f "${fastdiv_header}" ]]; then
  echo "cannot locate pinned CCCL headers" >&2
  exit 1
fi
cccl_root="${fastdiv_header%%/libcudacxx/include/*}"

mkdir -p "${output_dir}/bench"
"${nvcc}" -O3 -std=c++20 -arch=sm_121a -lineinfo --threads 4 \
  -D__CUDA_ARCH_SPECIFIC__ --expt-relaxed-constexpr \
  -diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20015 \
  -diag-suppress 2908 \
  -I"${native_dir}/include" \
  -I"${cccl_root}/libcudacxx/include" \
  -I"${cccl_root}/cub" -I"${cccl_root}/thrust" \
  -I"${flashinfer_root}/include" \
  -I"${flashinfer_root}/3rdparty/cutlass/include" \
  -I"${flashinfer_root}/3rdparty/cutlass/tools/util/include" \
  "${native_dir}/fixtures/q27_prefill_nvfp4_tactic_sweep.cu" \
  -L"${output_dir}" -lq27-prefill-mlp -lq27-prefill-nvfp4 \
  -lcublasLt -lcublas -lcudart \
  -Xlinker=-rpath -Xlinker='$ORIGIN' -o "${binary}"

if ldd "${binary}" | grep -q 'not found'; then
  ldd "${binary}"
  exit 1
fi
echo "built ${binary}"

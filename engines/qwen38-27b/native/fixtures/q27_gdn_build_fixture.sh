#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/qwen-gdn-block-fixture" >&2
  exit 2
fi

fixture=$1
native_dir=$(cd "$(dirname "$0")/.." && pwd)
repo_root=$(cd "$native_dir/../../.." && pwd)
image=${SPARKSERVE_CUDA_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}
user_id=$(id -u)
group_id=$(id -g)

if [[ ! -f "$fixture/projected_qkv_bf16.bin" ]]; then
  echo "q27 GDN fixture is incomplete: $fixture" >&2
  exit 1
fi
if [[ ! -f "$repo_root/build/aot-qwen/gdn/gdn_bf16_t1_h16_hv48_k128_v128_sm121.o" ]]; then
  echo "pinned q27-compatible GDN AOT object has not been built" >&2
  exit 1
fi

docker run --rm --gpus all --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v "$fixture:/fixture:ro" \
  "$image" \
  bash -euo pipefail -c '
    tvm_lib=$(find /usr/local/lib /usr/lib -type f -name "libtvm_ffi.so*" -print -quit 2>/dev/null)
    dialect_archive=$(find /usr/local/lib /usr/lib -type f -name libcuda_dialect_runtime_static.a -print -quit 2>/dev/null)
    if [[ -z "$tvm_lib" || -z "$dialect_archive" ]]; then
      echo "cannot locate TVM-FFI or CUDA-dialect runtime" >&2
      exit 1
    fi
    tvm_root=$(dirname "$(dirname "$tvm_lib")")
    mkdir -p build/q27
    nvcc -O2 -std=c++20 -arch=sm_121a \
      -Iengines/qwen38-27b/native/include \
      -Iengines/qwen38-27b/native/csrc \
      -I"$tvm_root/include" \
      engines/qwen38-27b/native/csrc/q27_gdn.cu \
      engines/qwen38-27b/native/csrc/q27_gdn_flashinfer.cc \
      engines/qwen38-27b/native/fixtures/q27_gdn_fixture_test.cu \
      build/aot-qwen/gdn/gdn_bf16_t1_h16_hv48_k128_v128_sm121.o \
      "$dialect_archive" -L"$tvm_root/lib" -ltvm_ffi \
      -lcuda -lcudart -ldl -Xcompiler=-pthread \
      -o build/q27/q27_gdn_fixture_test
    LD_LIBRARY_PATH="$tvm_root/lib:${LD_LIBRARY_PATH:-}" \
      build/q27/q27_gdn_fixture_test /fixture
  '

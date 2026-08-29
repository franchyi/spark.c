#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cuda_image=${SPARKSERVE_CUDA_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}
jobs=${JOBS:-4}
user_id=$(id -u)
group_id=$(id -g)

expected_image_id=$(tr -d '[:space:]' < \
  "$repo_root/third_party/qwen-native-build-image.id")
actual_image_id=$(docker image inspect "$cuda_image" --format '{{.Id}}')
if [[ "$actual_image_id" != "$expected_image_id" ]]; then
  echo "Q27 build image mismatch: expected $expected_image_id, got $actual_image_id" >&2
  exit 1
fi

"$script_dir/fetch-kernel-sources.sh" "$repo_root/third_party/_deps/flashinfer"

docker run --rm --gpus all --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -e FLASHINFER_WORKSPACE_BASE=/tmp/sparkserve-q27-flashinfer \
  -e XDG_CACHE_HOME=/tmp/sparkserve-q27-xdg-cache \
  "$cuda_image" \
  bash -euo pipefail -c '
    mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"
    rm -rf build/q27 build/q27-aot
    python3 scripts/export-gdn-aot.py build/q27-aot/gdn --tokens 1
    python3 scripts/export-q27-nvfp4-aot.py build/q27-aot/quantize
    sha256sum -c third_party/q27-aot-sm121.sha256

    tvm_lib=$(find /usr/local/lib /usr/lib -type f -name "libtvm_ffi.so*" \
      -print -quit 2>/dev/null)
    dialect_archive=$(find /usr/local/lib /usr/lib -type f \
      -name libcuda_dialect_runtime_static.a -print -quit 2>/dev/null)
    fastdiv_header=$(find /usr/local/lib/python3.12/dist-packages/flashinfer/data/cccl \
      -path "*/libcudacxx/include/cuda/__cmath/fast_modulo_division.h" \
      -print -quit 2>/dev/null)
    if [[ -z "$tvm_lib" || -z "$dialect_archive" || -z "$fastdiv_header" ]]; then
      echo "cannot locate pinned TVM-FFI/CUDA-dialect/CCCL dependency" >&2
      exit 1
    fi
    tvm_root=$(dirname "$(dirname "$tvm_lib")")
    cccl_root=${fastdiv_header%%/libcudacxx/include/*}

    mkdir -p build/q27
    cp -Lf "$tvm_lib" build/q27/libtvm_ffi.so
    make -j'"$jobs"' -C engines/qwen38-27b/native \
      TVM_FFI_ROOT="$tvm_root" \
      CUDA_DIALECT_ARCHIVE="$dialect_archive" \
      CCCL_ROOT="$cccl_root" \
      capsule-shared
    LD_LIBRARY_PATH="/work/build/q27:$tvm_root/lib:${LD_LIBRARY_PATH:-}" \
      make -C engines/qwen38-27b/native \
        TVM_FFI_ROOT="$tvm_root" \
        CUDA_DIALECT_ARCHIVE="$dialect_archive" \
        CCCL_ROOT="$cccl_root" \
        verify-linkage

    find build/q27 -maxdepth 1 -type f -name "*.so" -print0 \
      | sort -z | xargs -0 sha256sum > build/q27/manifest.sha256
    for library in build/q27/libq27-*.so; do
      echo "LINKAGE $library"
      ldd "$library"
    done
    echo "AOT_MANIFEST"
    cat third_party/q27-aot-sm121.sha256
    echo "CAPSULE_MANIFEST"
    cat build/q27/manifest.sha256
  '

echo "Q27 native capsule ready in $repo_root/build/q27"

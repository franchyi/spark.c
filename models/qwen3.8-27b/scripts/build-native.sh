#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
cuda_image=${SPARK_CUDA_IMAGE:-docker.1ms.run/lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}
rust_image=${SPARK_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}
jobs=${JOBS:-4}
user_id=$(id -u)
group_id=$(id -g)
target_host=${SPARK_Q27_CARGO_TARGET:-${HOME}/.cache/spark-c-q27-target}
cargo_home=${SPARK_CARGO_HOME:-${HOME}/.cache/spark-c-cargo-home}

"$repo_root/vendor/tools/fetch-flashinfer.sh" "$repo_root/vendor/_deps/flashinfer"

docker run --rm --gpus all --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -e FLASHINFER_WORKSPACE_BASE=/tmp/q27-flashinfer \
  -e XDG_CACHE_HOME=/tmp/q27-xdg-cache \
  "$cuda_image" \
  bash -euo pipefail -c '
    mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"
    # c427 prefill artifacts may be root-owned because they are exported by a
    # separate donor container. They are not inputs to this capsule build.
    # Rebuild only the model-owned AOT directories and leave c427 untouched.
    rm -rf build/q27 build/q27-aot/gdn build/q27-aot/quantize \
      build/q27-aot/verify-t8 build/q27-aot/.verified
    python3 vendor/tools/export-gdn-aot.py build/q27-aot/gdn \
      --namespace q27 --tokens 1
    PYTHONPATH=/work/vendor/_deps/flashinfer \
      python3 vendor/tools/export-q27-gdn-verify-aot.py \
        build/q27-aot/verify-t8
    python3 models/qwen3.8-27b/native/tools/export-nvfp4-aot.py build/q27-aot/quantize
    sha256sum -c vendor/q27-aot-sm121.sha256

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
    make -j'"$jobs"' -C models/qwen3.8-27b/native \
      TVM_FFI_ROOT="$tvm_root" \
      CUDA_DIALECT_ARCHIVE="$dialect_archive" \
      CCCL_ROOT="$cccl_root" \
      capsule-shared
    LD_LIBRARY_PATH="/work/build/q27:$tvm_root/lib:${LD_LIBRARY_PATH:-}" \
      make -C models/qwen3.8-27b/native \
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
    cat vendor/q27-aot-sm121.sha256
    echo "CAPSULE_MANIFEST"
    cat build/q27/manifest.sha256
  '

echo "Q27 native capsule ready in $repo_root/build/q27"

mkdir -p "$repo_root/build/bin" "$target_host" "$cargo_home"
docker run --rm --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /usr/lib/aarch64-linux-gnu:/host-driver-lib:ro \
  -v "$target_host:/cargo-target" \
  -v "$cargo_home:/cargo-home" \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e 'RUSTFLAGS=-L native=/work/build/q27 -C link-arg=-Wl,-rpath,$ORIGIN/../q27 -C link-arg=-Wl,-rpath-link,/work/build/q27 -C link-arg=-Wl,-rpath-link,/usr/local/cuda/lib64 -C link-arg=-Wl,-rpath-link,/host-driver-lib' \
  "$rust_image" \
  cargo build --locked --release \
    --manifest-path models/qwen3.8-27b/native/Cargo.toml \
    --bin q27-eager --bin q27-serve
cp "$target_host/release/q27-eager" "$repo_root/build/bin/q27-eager"
cp "$target_host/release/q27-serve" "$repo_root/build/bin/q27-serve"

echo "Q27 native tools ready in $repo_root/build/bin"

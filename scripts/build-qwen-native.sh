#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cuda_image=${SPARKSERVE_CUDA_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}
rust_image=${SPARKSERVE_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}
jobs=${JOBS:-4}
target_host=${SPARKSERVE_RUST_TARGET:-${HOME}/.cache/sparkserve-rust-target}
cargo_home=${SPARKSERVE_CARGO_HOME:-${HOME}/.cache/sparkserve-cargo-home}
user_id=$(id -u)
group_id=$(id -g)
cuda_host_root=${SPARKSERVE_CUDA_HOST_ROOT:-$(readlink -f /usr/local/cuda)}
if [[ ! -d "$cuda_host_root/lib64" ]]; then
  echo "cannot locate the Spark CUDA toolkit at $cuda_host_root" >&2
  exit 1
fi
expected_cuda_image_id=$(tr -d '[:space:]' < "$repo_root/third_party/qwen-native-build-image.id")
actual_cuda_image_id=$(docker image inspect "$cuda_image" --format '{{.Id}}')
if [[ "$actual_cuda_image_id" != "$expected_cuda_image_id" ]]; then
  echo "Qwen build image mismatch: expected $expected_cuda_image_id, got $actual_cuda_image_id" >&2
  exit 1
fi

"$script_dir/fetch-kernel-sources.sh" "$repo_root/third_party/_deps/flashinfer"
mkdir -p "$repo_root/build/bin" "$target_host" "$cargo_home"

docker run --rm --gpus all --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -e HOME=/tmp/sparkserve-cuda-home \
  "$cuda_image" \
  bash -euo pipefail -c '
    mkdir -p "$HOME"
    rm -rf build/aot-qwen
    python3 scripts/export-gdn-aot.py build/aot-qwen/gdn --tokens 1 2 4 8 16
    python3 scripts/export-silu-nvfp4-aot.py build/aot-qwen/silu
    python3 scripts/export-quantize-nvfp4-aot.py build/aot-qwen/quantize
    sha256sum -c third_party/qwen-aot-sm121.sha256

    tvm_lib=$(find /usr/local/lib /usr/lib -type f -name "libtvm_ffi.so*" -print -quit 2>/dev/null)
    dialect_archive=$(find /usr/local/lib /usr/lib -type f -name libcuda_dialect_runtime_static.a -print -quit 2>/dev/null)
    if [[ -z "$tvm_lib" || -z "$dialect_archive" ]]; then
      echo "cannot locate TVM-FFI or CUDA-dialect runtime in $0" >&2
      exit 1
    fi
    tvm_root=$(dirname "$(dirname "$tvm_lib")")
    cute_root=$(dirname "$(dirname "$dialect_archive")")

    make -j'"$jobs"' CUDA_ARCH=sm_121a \
      GDN_AOT_OBJECT=/work/build/aot-qwen/gdn/gdn_bf16_t1_h16_hv48_k128_v128_sm121.o \
      GDN_PREFILL_AOT_OBJECTS="/work/build/aot-qwen/gdn/gdn_bf16_t2_h16_hv48_k128_v128_sm121.o /work/build/aot-qwen/gdn/gdn_bf16_t4_h16_hv48_k128_v128_sm121.o /work/build/aot-qwen/gdn/gdn_bf16_t8_h16_hv48_k128_v128_sm121.o /work/build/aot-qwen/gdn/gdn_bf16_t16_h16_hv48_k128_v128_sm121.o" \
      CUTE_NVFP4_OBJECT=/work/build/aot-qwen/silu/swizzled_bfloat16_k640_sf0_pdl0_silu.o \
      CUTE_NVFP4_QUANTIZE_OBJECT=/work/build/aot-qwen/quantize/swizzled_bfloat16_k2560_sf0_pdl0.o \
      TVM_FFI_ROOT="$tvm_root" CUTE_DSL_ROOT="$cute_root" \
      fabric-shared qsa-shared qwen-runtime-shared
    cp -Lf "$tvm_lib" build/libtvm_ffi.so
  '

mkdir -p "$target_host"
docker run --rm --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v "$target_host:/cargo-target" \
  -v "$cargo_home:/cargo-home" \
  -v "$cuda_host_root:/usr/local/cuda:ro" \
  -e HOME=/tmp/sparkserve-rust-home \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e RUSTFLAGS='-L native=/work/build -l dylib=sparkserve-fabric -l dylib=sparkserve-qwen-runtime -l dylib=sparkserve-qsa -C link-arg=-Wl,-rpath-link,/work/build -C link-arg=-Wl,-rpath-link,/usr/local/cuda/lib64' \
  "$rust_image" \
  cargo build --release -p sparkserve-runtime --features native-fabric-smoke \
    --example qwen_first_token --example qwen_decode --example qwen_serve

mkdir -p "$repo_root/build/bin"
for binary in qwen_first_token qwen_decode qwen_serve; do
  cp "$target_host/release/examples/$binary" \
    "$repo_root/build/bin/$binary"
done

echo "native Qwen binaries ready in $repo_root/build/bin"

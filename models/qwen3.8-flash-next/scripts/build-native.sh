#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
cuda_image=${SPARK_CUDA_IMAGE:-docker.1ms.run/lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}
rust_image=${FLASH_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}
jobs=${JOBS:-4}
target_host=${FLASH_RUST_TARGET:-${HOME}/.cache/flash-rust-target}
cargo_home=${FLASH_CARGO_HOME:-${HOME}/.cache/flash-cargo-home}
user_id=$(id -u)
group_id=$(id -g)
cuda_host_root=${FLASH_CUDA_HOST_ROOT:-$(readlink -f /usr/local/cuda)}
if [[ ! -d "$cuda_host_root/lib64" ]]; then
  echo "cannot locate the Spark CUDA toolkit at $cuda_host_root" >&2
  exit 1
fi
"$repo_root/vendor/tools/fetch-flashinfer.sh" "$repo_root/vendor/_deps/flashinfer"
mkdir -p "$repo_root/build/bin" "$target_host" "$cargo_home"

docker run --rm --gpus all --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -e HOME=/tmp/flash-cuda-home \
  "$cuda_image" \
  bash -euo pipefail -c '
    mkdir -p "$HOME"
    rm -rf build/aot-qwen build/flash-next
    python3 vendor/tools/export-gdn-aot.py build/aot-qwen/gdn --tokens 1 2 4 8 16
    python3 models/qwen3.8-flash-next/native/tools/export-silu-nvfp4-aot.py build/aot-qwen/silu
    python3 models/qwen3.8-flash-next/native/tools/export-quantize-nvfp4-aot.py build/aot-qwen/quantize
    sha256sum -c vendor/qwen-aot-sm121.sha256

    tvm_lib=$(find /usr/local/lib /usr/lib -type f -name "libtvm_ffi.so*" -print -quit 2>/dev/null)
    dialect_archive=$(find /usr/local/lib /usr/lib -type f -name libcuda_dialect_runtime_static.a -print -quit 2>/dev/null)
    if [[ -z "$tvm_lib" || -z "$dialect_archive" ]]; then
      echo "cannot locate TVM-FFI or CUDA-dialect runtime in $0" >&2
      exit 1
    fi
    tvm_root=$(dirname "$(dirname "$tvm_lib")")
    cute_root=$(dirname "$(dirname "$dialect_archive")")

    make -f models/qwen3.8-flash-next/native/kernels.mk -j'"$jobs"' CUDA_ARCH=sm_121a \
      GDN_AOT_OBJECT=/work/build/aot-qwen/gdn/gdn_bf16_t1_h16_hv48_k128_v128_sm121.o \
      GDN_PREFILL_AOT_OBJECTS="/work/build/aot-qwen/gdn/gdn_bf16_t2_h16_hv48_k128_v128_sm121.o /work/build/aot-qwen/gdn/gdn_bf16_t4_h16_hv48_k128_v128_sm121.o /work/build/aot-qwen/gdn/gdn_bf16_t8_h16_hv48_k128_v128_sm121.o /work/build/aot-qwen/gdn/gdn_bf16_t16_h16_hv48_k128_v128_sm121.o" \
      CUTE_NVFP4_OBJECT=/work/build/aot-qwen/silu/swizzled_bfloat16_k640_sf0_pdl0_silu.o \
      CUTE_NVFP4_QUANTIZE_OBJECT=/work/build/aot-qwen/quantize/swizzled_bfloat16_k2560_sf0_pdl0.o \
      TVM_FFI_ROOT="$tvm_root" CUTE_DSL_ROOT="$cute_root" \
      fabric-shared qwen-runtime-shared
    cp -Lf "$tvm_lib" build/flash-next/libtvm_ffi.so
  '

mkdir -p "$target_host"
docker run --rm --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v "$target_host:/cargo-target" \
  -v "$cargo_home:/cargo-home" \
  -v "$cuda_host_root:/usr/local/cuda:ro" \
  -e HOME=/tmp/flash-rust-home \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e RUSTFLAGS='-L native=/work/build/flash-next -l dylib=flash-fabric -l dylib=flash-qwen-runtime -C link-arg=-Wl,-rpath-link,/work/build/flash-next -C link-arg=-Wl,-rpath-link,/usr/local/cuda/lib64' \
  "$rust_image" \
  cargo build --release -p spark-flash-next --features native-fabric-smoke \
    --example qwen_first_token --example qwen_decode --example qwen_serve \
    --example qwen_expert_sidecar_v2

mkdir -p "$repo_root/build/bin"
for binary in qwen_first_token qwen_decode qwen_serve qwen_expert_sidecar_v2; do
  cp "$target_host/release/examples/$binary" \
    "$repo_root/build/bin/$binary"
done

echo "native Qwen binaries ready in $repo_root/build/bin"

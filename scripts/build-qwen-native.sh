#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cuda_image=${SPARKSERVE_CUDA_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}
rust_image=${SPARKSERVE_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}
jobs=${JOBS:-4}
target_host=${SPARKSERVE_RUST_TARGET:-${HOME}/.cache/sparkserve-rust-target}

"$script_dir/fetch-kernel-sources.sh" "$repo_root/third_party/_deps/flashinfer"

docker run --rm --gpus all --network host \
  -v "$repo_root:/work" -w /work \
  "$cuda_image" \
  bash -euo pipefail -c '
    rm -rf build/aot-qwen
    python3 scripts/export-gdn-aot.py build/aot-qwen/gdn --tokens 1 2 4 8 16
    python3 scripts/export-silu-nvfp4-aot.py build/aot-qwen/silu
    python3 scripts/export-quantize-nvfp4-aot.py build/aot-qwen/quantize
    printf "%s  %s\n" \
      4779bfc774d485240ef2b0ae4be8bd3a6b45619cad2c02b4f236e5a58c972163 \
      build/aot-qwen/gdn/gdn_bf16_t1_h16_hv48_k128_v128_sm121.o \
      | sha256sum -c -
    printf "%s  %s\n" \
      8cbc3a588037ba109978c019da0f774c87f3968193376a341a8fc35624376916 \
      build/aot-qwen/quantize/swizzled_bfloat16_k2560_sf0_pdl0.o \
      | sha256sum -c -

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
docker run --rm --network host \
  -v "$repo_root:/work" -w /work \
  -v "$target_host:/cargo-target" \
  -e CARGO_HOME=/work/.cache/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e RUSTFLAGS='-L native=/work/build -l dylib=sparkserve-fabric -l dylib=sparkserve-qwen-runtime -l dylib=sparkserve-qsa -C link-arg=-Wl,--allow-shlib-undefined' \
  "$rust_image" \
  cargo build --release -p sparkserve-runtime --features native-fabric-smoke \
    --example qwen_first_token --example qwen_decode --example qwen_serve

mkdir -p "$repo_root/build/bin"
for binary in qwen_first_token qwen_decode qwen_serve; do
  cp "$target_host/release/examples/$binary" \
    "$repo_root/build/bin/$binary"
done

echo "native Qwen binaries ready in $repo_root/build/bin"

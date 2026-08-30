#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
cuda_image=${SPARK_CUDA_IMAGE:-docker.1ms.run/lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}
jobs=${JOBS:-4}
user_id=$(id -u)
group_id=$(id -g)
source_cache=${SPARK_SOURCE_CACHE:-${HOME}/.cache/spark-c/sources}
flashinfer_host="$source_cache/flashinfer"
sglang_host="$source_cache/sglang-c427"
aot_cache=${SPARK_AOT_CACHE:-${HOME}/.cache/spark-c/aot}
c427_host="$aot_cache/q27/c427"

"$repo_root/tools/fetch-flashinfer.sh" "$flashinfer_host"
if [[ ! -f "$c427_host/manifest.json" ]] || \
   ! grep -Fq c4271c3fe1262fc2adbd162c33b25de5255251c5 "$c427_host/manifest.json" || \
   [[ ! -f "$c427_host/000-fused_qkvzba_split_reshape_cat_contiguous_kernel.cubin" ]] || \
   [[ ! -f "$c427_host/014-chunk_fwd_kernel_o.cubin" ]]; then
  "$script_dir/fetch-c427.sh" "$sglang_host"
fi
mkdir -p "$aot_cache/q27/flashinfer" "$aot_cache/q27/xdg"

docker run --rm --gpus all --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v "$source_cache:/opt/spark-sources:ro" \
  -v "$aot_cache:/opt/spark-aot" \
  -e PYTHONPATH=/opt/spark-sources/flashinfer \
  -e FLASHINFER_WORKSPACE_BASE=/opt/spark-aot/q27/flashinfer \
  -e XDG_CACHE_HOME=/opt/spark-aot/q27/xdg \
  "$cuda_image" \
  bash -euo pipefail -c '
    mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"
    c427=/opt/spark-aot/q27/c427
    if [[ ! -f "$c427/manifest.json" ]] || \
       ! grep -Fq c4271c3fe1262fc2adbd162c33b25de5255251c5 "$c427/manifest.json" || \
       [[ ! -f "$c427/000-fused_qkvzba_split_reshape_cat_contiguous_kernel.cubin" ]] || \
       [[ ! -f "$c427/001-_causal_conv1d_fwd_kernel.cubin" ]] || \
       [[ ! -f "$c427/002-fused_qkv_split_gdn_prefill_kernel.cubin" ]] || \
       [[ ! -f "$c427/003-l2norm_fwd_kernel.cubin" ]] || \
       [[ ! -f "$c427/005-chunk_local_cumsum_scalar_kernel.cubin" ]] || \
       [[ ! -f "$c427/009-chunk_gated_delta_rule_fwd_kkt_solve_kernel.cubin" ]] || \
       [[ ! -f "$c427/012-recompute_w_u_fwd_kernel.cubin" ]] || \
       [[ ! -f "$c427/013-chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin" ]] || \
       [[ ! -f "$c427/014-chunk_fwd_kernel_o.cubin" ]]; then
      rm -rf /opt/spark-aot/q27/c427
      python3 models/qwen3.8-27b/engine/tools/export-c427-gdn.py \
        --sglang-root /opt/spark-sources/sglang-c427 \
        --output /opt/spark-aot/q27/c427
    else
      echo "reusing Q27 c427 SM121 artifacts"
    fi
    if [[ ! -f build/q27-aot/verify-t8/q27_verify_gdn_bf16_t8_h16_hv48_k128_v128_sm121.o ]] || \
       ! sha256sum -c models/qwen3.8-27b/engine/aot.sha256 >/dev/null 2>&1; then
      rm -rf build/q27-aot/gdn build/q27-aot/quantize \
        build/q27-aot/verify-t8 build/q27-aot/.verified
      python3 tools/export-gdn.py build/q27-aot/gdn \
        --namespace q27 --tokens 1
      python3 models/qwen3.8-27b/engine/tools/export-gdn-verify.py \
          build/q27-aot/verify-t8
      python3 models/qwen3.8-27b/engine/tools/export-nvfp4-aot.py \
        build/q27-aot/quantize
    else
      echo "reusing verified Q27 SM121 AOT objects"
    fi
    sha256sum -c models/qwen3.8-27b/engine/aot.sha256

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
    make -j'"$jobs"' -C models/qwen3.8-27b/engine \
      FLASHINFER_ROOT=/opt/spark-sources/flashinfer \
      TVM_FFI_ROOT="$tvm_root" \
      CUDA_DIALECT_ARCHIVE="$dialect_archive" \
      CCCL_ROOT="$cccl_root" \
      capsule-shared
    LD_LIBRARY_PATH="/work/build/q27:$tvm_root/lib:${LD_LIBRARY_PATH:-}" \
      make -C models/qwen3.8-27b/engine \
        FLASHINFER_ROOT=/opt/spark-sources/flashinfer \
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
    cat models/qwen3.8-27b/engine/aot.sha256
    echo "CAPSULE_MANIFEST"
    cat build/q27/manifest.sha256
  '

echo "Q27 target capsule ready in $repo_root/build/q27"

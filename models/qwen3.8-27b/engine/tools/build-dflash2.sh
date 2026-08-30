#!/usr/bin/env bash
set -euo pipefail

# Incremental Spark-only build for the model-specific DFlash2 server.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
engine_dir="$repo_root/models/qwen3.8-27b/engine"
library_dir="$repo_root/build/q27"
engine="$library_dir/libq27-dflash2-engine.so"
server="$repo_root/build/bin/q27-serve-dflash2"

cuda_image=${SPARK_CUDA_IMAGE:-lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}
rust_image=${SPARK_RUST_IMAGE:-rust:1.89.0}
jobs=${JOBS:-4}
user_id=$(id -u)
group_id=$(id -g)
cargo_target=${SPARK_Q27_DFLASH2_CARGO_TARGET:-${HOME}/.cache/spark-c/cargo/q27-target}
cargo_home=${SPARK_Q27_DFLASH2_CARGO_HOME:-${HOME}/.cache/spark-c/cargo/home}
source_cache=${SPARK_SOURCE_CACHE:-${HOME}/.cache/spark-c/sources}

if [[ ! -f "$library_dir/libq27-model.so" ]]; then
  echo "Build the current Q27 target capsule first: $library_dir/libq27-model.so" >&2
  exit 1
fi

"$repo_root/tools/fetch-flashinfer.sh" "$source_cache/flashinfer"

engine_stale=false
for library in \
  libq27-dflash2-control.so libq27-dflash2-conv.so \
  libq27-dflash2-flashinfer.so libq27-dflash2-attention.so \
  libq27-dflash2-mlp.so libq27-dflash2-model.so \
  libq27-dflash2-topk.so libq27-dflash2-kv.so \
  libq27-dflash2-engine.so; do
  if [[ ! -f "$library_dir/$library" ]]; then
    engine_stale=true
    break
  fi
done
if [[ "$engine_stale" == false ]] && \
    [[ -n "$(find "$engine_dir/cuda" "$engine_dir/include" "$engine_dir/tools" \
      -type f \( -name 'q27_dflash2*' -o -name 'build-dflash2-*.sh' \) \
      -newer "$engine" -print -quit)" ]]; then
  engine_stale=true
fi
if [[ "$engine_stale" == false && "$library_dir/libq27-model.so" -nt "$engine" ]]; then
  engine_stale=true
fi

if [[ "$engine_stale" == true ]]; then
  docker run --rm --network host --user "$user_id:$group_id" \
    -v "$repo_root:/work" -w /work \
    -v "$source_cache:/opt/spark-sources:ro" \
    -e FLASHINFER_ROOT=/opt/spark-sources/flashinfer \
    "$cuda_image" \
    bash -euo pipefail \
      models/qwen3.8-27b/engine/tools/build-dflash2-engine.sh
else
  echo "Q27 DFlash2 engine is current: $engine"
fi

mkdir -p "$repo_root/build/bin" "$cargo_target" "$cargo_home"
docker run --rm --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /usr/lib/aarch64-linux-gnu:/host-driver-lib:ro \
  -v "$cargo_target:/cargo-target" \
  -v "$cargo_home:/cargo-home" \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e 'RUSTFLAGS=-L native=/work/build/q27 -C link-arg=-Wl,-rpath,$ORIGIN/../q27 -C link-arg=-Wl,-rpath-link,/work/build/q27 -C link-arg=-Wl,-rpath-link,/usr/local/cuda/targets/sbsa-linux/lib -C link-arg=-Wl,-rpath-link,/host-driver-lib' \
  "$rust_image" \
  cargo build --locked --release --jobs "$jobs" \
    --manifest-path models/qwen3.8-27b/engine/Cargo.toml \
    --bin q27-serve-dflash2 --bin q27-pack-scales
cp -f "$cargo_target/release/q27-serve-dflash2" "$server"
cp -f "$cargo_target/release/q27-pack-scales" \
  "$repo_root/build/bin/q27-pack-scales"

echo "Q27 DFlash2 server ready: $server"

#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
rust_image=${FLASH_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}
target_host=${FLASH_RUST_FUSED_TARGET:-${HOME}/.cache/spark-c/cargo/flash-fused-target}
cargo_home=${FLASH_CARGO_HOME:-${HOME}/.cache/spark-c/cargo/home}
cuda_host_root=${FLASH_CUDA_HOST_ROOT:-$(readlink -f /usr/local/cuda)}
user_id=$(id -u)
group_id=$(id -g)
library_dir="$repo_root/build/flash-next"

for required in \
  "$library_dir/libflash-fabric.so" \
  "$library_dir/libflash-qwen-runtime.so" \
  "$library_dir/libflash-qwen-fused-moe.so" \
  "$library_dir/libflash-qwen-runtime-fused.so" \
  "$library_dir/libtvm_ffi.so"; do
  if [[ ! -e "$required" ]]; then
    echo "missing fused Qwen build artifact: $required" >&2
    exit 1
  fi
done
if [[ ! -d "$cuda_host_root/lib64" ]]; then
  echo "cannot locate the Spark CUDA toolkit at $cuda_host_root" >&2
  exit 1
fi

mkdir -p "$repo_root/build/bin" "$target_host" "$cargo_home"
docker run --rm --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v "$target_host:/cargo-target" \
  -v "$cargo_home:/cargo-home" \
  -v "$cuda_host_root:/usr/local/cuda:ro" \
  -e HOME=/tmp/flash-rust-home \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e RUSTFLAGS='-L native=/work/build/flash-next -l dylib=flash-fabric -l dylib=flash-qwen-runtime-fused -l dylib=flash-qwen-runtime -C link-arg=-Wl,-rpath-link,/work/build/flash-next -C link-arg=-Wl,-rpath-link,/usr/local/cuda/lib64' \
  "$rust_image" \
  cargo build --locked --release -p spark-flash-next --features cuda \
    --bin qwen_serve

cp "$target_host/release/qwen_serve" \
  "$repo_root/build/bin/qwen_serve_fused"
echo "fused Qwen server ready at $repo_root/build/bin/qwen_serve_fused"

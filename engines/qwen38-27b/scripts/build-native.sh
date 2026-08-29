#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
rust_image=${SPARKSERVE_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}
target_host=${SPARKSERVE_Q27_TARGET:-${HOME}/.cache/sparkserve-q27-target}
cargo_home=${SPARKSERVE_CARGO_HOME:-${HOME}/.cache/sparkserve-cargo-home}
user_id=$(id -u)
group_id=$(id -g)

mkdir -p "$repo_root/build/bin" "$target_host" "$cargo_home"
make -C "$repo_root/engines/qwen38-27b/native" kernels mapping
docker run --rm --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v "$target_host:/cargo-target" \
  -v "$cargo_home:/cargo-home" \
  -e HOME=/tmp/sparkserve-q27-home \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  "$rust_image" \
  cargo build --release -p sparkserve-q27 --bin q27-inspect --bin q27-map-inspect
cp "$target_host/release/q27-inspect" "$repo_root/build/bin/q27-inspect"
cp "$target_host/release/q27-map-inspect" "$repo_root/build/bin/q27-map-inspect"
echo "q27 checkpoint inspectors ready in $repo_root/build/bin"

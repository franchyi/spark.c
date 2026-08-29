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
"$repo_root/scripts/build-q27-capsule.sh"
docker run --rm --network host --user "$user_id:$group_id" \
  -v "$repo_root:/work" -w /work \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v "$target_host:/cargo-target" \
  -v "$cargo_home:/cargo-home" \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e 'RUSTFLAGS=-L native=/work/build/q27 -C link-arg=-Wl,-rpath,$ORIGIN/../q27 -C link-arg=-Wl,-rpath-link,/work/build/q27 -C link-arg=-Wl,-rpath-link,/usr/local/cuda/lib64' \
  "$rust_image" \
  cargo build --release -p sparkserve-q27 --bin q27-inspect --bin q27-map-inspect --bin q27-pack-scales --bin q27-inspect-scales --bin q27-eager
cp "$target_host/release/q27-inspect" "$repo_root/build/bin/q27-inspect"
cp "$target_host/release/q27-map-inspect" "$repo_root/build/bin/q27-map-inspect"
cp "$target_host/release/q27-pack-scales" "$repo_root/build/bin/q27-pack-scales"
cp "$target_host/release/q27-inspect-scales" "$repo_root/build/bin/q27-inspect-scales"
cp "$target_host/release/q27-eager" "$repo_root/build/bin/q27-eager"
echo "q27 native tools ready in $repo_root/build/bin"

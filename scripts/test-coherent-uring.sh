#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_image="${SPARKSERVE_CUDA_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}"
rust_image="${SPARKSERVE_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}"

docker run --rm --gpus all \
  -v "${repo_root}:/work" -w /work \
  "${cuda_image}" \
  make -B NVCC=/usr/local/cuda/bin/nvcc fabric-shared

docker run --rm --network host \
  -v "${repo_root}:/work" -w /work \
  -e CARGO_HOME=/work/target/cargo-home \
  -e RUSTFLAGS="-L native=/work/build -l dylib=sparkserve-fabric -C link-arg=-Wl,--allow-shlib-undefined" \
  "${rust_image}" \
  cargo build --release --example coherent_uring_smoke \
    --features native-fabric-smoke

docker run --rm --gpus all --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  -v "${repo_root}:/work" -w /work \
  -e LD_LIBRARY_PATH=/work/build:/usr/local/cuda/lib64 \
  "${cuda_image}" \
  /work/target/release/examples/coherent_uring_smoke

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xqa_fixture="${1:-tests/fixtures/private/qsa-xqa}"
pack_fixture="${2:-tests/fixtures/private/qsa-kv-pack}"
index_fixture="${3:-tests/fixtures/private/qsa-index-prep}"
topk_fixture="${4:-tests/fixtures/private/qsa-topk}"
expand_fixture="${5:-tests/fixtures/private/qsa-expand}"
cuda_image="${SPARKSERVE_CUDA_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}"
rust_image="${SPARKSERVE_RUST_IMAGE:-docker.1ms.run/rust:1.89.0}"

docker run --rm --gpus all \
  -v "${repo_root}:/work" -w /work \
  "${cuda_image}" \
  make -B NVCC=/usr/local/cuda/bin/nvcc fabric-shared qsa-shared

docker run --rm --network host \
  -v "${repo_root}:/work" -w /work \
  -e CARGO_HOME=/work/target/cargo-home \
  -e RUSTFLAGS="-L native=/work/build -l dylib=sparkserve-fabric -l dylib=sparkserve-qsa -C link-arg=-Wl,--allow-shlib-undefined" \
  "${rust_image}" \
  cargo build --release --example qsa_frontend_smoke --example qsa_xqa_smoke \
    --features native-qsa-smoke

docker run --rm --gpus all --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  -v "${repo_root}:/work" -w /work \
  -e LD_LIBRARY_PATH=/work/build:/usr/local/cuda/lib64 \
  "${cuda_image}" \
  /work/target/release/examples/qsa_frontend_smoke \
    "/work/${index_fixture}" "/work/${topk_fixture}" "/work/${expand_fixture}"

docker run --rm --gpus all --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 \
  -v "${repo_root}:/work" -w /work \
  -e LD_LIBRARY_PATH=/work/build:/usr/local/cuda/lib64 \
  "${cuda_image}" \
  /work/target/release/examples/qsa_xqa_smoke \
    "/work/${xqa_fixture}" "/work/${pack_fixture}"

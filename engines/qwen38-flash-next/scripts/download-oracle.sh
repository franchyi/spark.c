#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SPARK_ENGINE_IMAGE:-qwen38-flash-dgx}"
MODEL="${SPARK_ENGINE_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${SPARK_ENGINE_CACHE:-${HOME}/.cache/huggingface}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${HF_CACHE}"
docker run --rm \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_ENDPOINT="${HF_ENDPOINT}" \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  --entrypoint python3 "${IMAGE}" \
  -c "from huggingface_hub import snapshot_download; print(snapshot_download('${MODEL}', max_workers=8))"

#!/usr/bin/env bash
set -euo pipefail

model_dir="${SPARKSERVE_MODEL_DIR:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}"
image="${SPARKSERVE_FLASH_IMAGE:-sparkserve/sglang:qwen38flashnext-sm121}"
container="${SPARKSERVE_FLASH_CONTAINER:-qwen38-flash-next-smoke}"
port="${SPARKSERVE_FLASH_PORT:-8890}"
cache_dir="${SPARKSERVE_FLASH_CACHE:-/home/chaoyi/.cache/sglang-flash-next}"

if [[ ! -f "${model_dir}/config.json" ]]; then
  echo "missing model config: ${model_dir}/config.json" >&2
  exit 2
fi

if [[ ! -f "${model_dir}/.sparkserve/ple.ssple" ]]; then
  echo "missing SparkServe PLE index: ${model_dir}/.sparkserve/ple.ssple" >&2
  echo "run: uv run sparkserve ple-index ${model_dir} ${model_dir}/.sparkserve/ple.ssple" >&2
  exit 2
fi

if find "${model_dir}/.cache" -type f -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
  echo "model download is incomplete: ${model_dir}" >&2
  exit 2
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${container}"; then
  docker rm -f "${container}" >/dev/null
fi

mkdir -p "${cache_dir}"

docker run -d \
  --name "${container}" \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  -e PYTHONUNBUFFERED=1 \
  -e SPARKSERVE_PLE_INDEX=/model/.sparkserve/ple.ssple \
  -e SPARKSERVE_MODEL_ROOT=/model \
  -e SPARKSERVE_PLE_CACHE_MIB="${SPARKSERVE_PLE_CACHE_MIB:-512}" \
  -e SPARKSERVE_PLE_WORKERS="${SPARKSERVE_PLE_WORKERS:-16}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_CACHE_DIR=/cache/sglang \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "${model_dir}:/model:ro" \
  -v "${cache_dir}:/cache" \
  "${image}" \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name qwen3.8-flash-next-nvfp4 \
    --trust-remote-code \
    --quantization modelopt_fp4 \
    --language-model-only \
    --ple-offload-embedding \
    --weight-loader-drop-cache-after-load \
    --prefill-attention-backend triton \
    --decode-attention-backend trtllm_mha \
    --page-size 64 \
    --chunked-prefill-size 4096 \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 64 \
    --mem-fraction-static "${SPARKSERVE_MEM_FRACTION_STATIC:-0.82}" \
    --context-length 32768 \
    --max-running-requests 1 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --sampling-defaults model \
    --allow-auto-truncate \
    --host 0.0.0.0 \
    --port "${port}"

echo "started ${container} on port ${port}"
echo "follow logs: docker logs -f ${container}"

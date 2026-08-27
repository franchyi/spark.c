#!/usr/bin/env bash
set -euo pipefail

port="${SPARKSERVE_FLASH_PORT:-8890}"
base_url="${SPARKSERVE_FLASH_URL:-http://127.0.0.1:${port}}"

curl --fail --silent --show-error "${base_url}/v1/models"
printf '\n'
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-nvfp4","messages":[{"role":"user","content":"Reply with exactly: Spark ready"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' \
  "${base_url}/v1/chat/completions"
printf '\n'

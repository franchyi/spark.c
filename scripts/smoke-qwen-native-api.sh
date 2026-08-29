#!/usr/bin/env bash
set -euo pipefail

base_url=${1:-http://127.0.0.1:8020}
model=${SPARKSERVE_QWEN_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
timeout=${SPARKSERVE_QWEN_SMOKE_TIMEOUT:-900}

health=$(curl -fsS --max-time 10 "$base_url/health")
grep -Fq '"status":"ok"' <<<"$health"

models=$(curl -fsS --max-time 10 "$base_url/v1/models")
grep -Fq "\"id\":\"$model\"" <<<"$models"

chat=$(curl -fsS --max-time "$timeout" \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word.\"}],\"temperature\":0,\"top_p\":1,\"max_tokens\":2}" \
  "$base_url/v1/chat/completions")
grep -Fq '"object":"chat.completion"' <<<"$chat"
grep -Fq '"role":"assistant"' <<<"$chat"

responses=$(curl -fsS --max-time "$timeout" \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$model\",\"input\":\"Reply with one word.\",\"temperature\":0,\"top_p\":1,\"max_output_tokens\":2}" \
  "$base_url/v1/responses")
grep -Fq '"object":"response"' <<<"$responses"

stream=$(curl -NfsS --max-time "$timeout" \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word.\"}],\"temperature\":0,\"top_p\":1,\"max_tokens\":2,\"stream\":true}" \
  "$base_url/v1/chat/completions")
grep -Fq '"object":"chat.completion.chunk"' <<<"$stream"
grep -Fq 'data: [DONE]' <<<"$stream"

echo "native Qwen API smoke passed at $base_url"

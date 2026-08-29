#!/usr/bin/env bash
set -euo pipefail

port=${SPARK_ENGINE_PORT:-30000}
base=${SPARK_ENGINE_BASE_URL:-http://127.0.0.1:${port}}
model=${SPARK_ENGINE_MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}
smoke_dir=$(mktemp -d "/tmp/q27-native-smoke.XXXXXX")
trap 'rm -rf "$smoke_dir"' EXIT

curl -fsS "$base/v1/models" -o "$smoke_dir/models.json"
grep -Fq "$model" "$smoke_dir/models.json"

curl -fsS "$base/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  -o "$smoke_dir/chat.json"
grep -Fq '"choices"' "$smoke_dir/chat.json"

curl -fsSN --max-time 180 "$base/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"temperature\":0,\"stream\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  -o "$smoke_dir/stream.txt"
grep -Fq 'data: [DONE]' "$smoke_dir/stream.txt"

echo "engine=qwen38-27b-native smoke=pass endpoint=$base"

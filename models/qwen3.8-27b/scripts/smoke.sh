#!/usr/bin/env bash
set -euo pipefail

port="${SPARK_ENGINE_PORT:-8888}"
base="http://127.0.0.1:${port}"
model="qwen3.8-27b-sglang"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-27b-smoke.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT

curl -fsS "${base}/v1/models" -o "${tmp_dir}/models.json"
grep -q "${model}" "${tmp_dir}/models.json"

curl -fsS "${base}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-sglang","messages":[{"role":"user","content":"Reply with exactly: SPARK_OK"}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  -o "${tmp_dir}/chat.json"
grep -q '"choices"' "${tmp_dir}/chat.json"

curl -fsSN --max-time 180 "${base}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-sglang","messages":[{"role":"user","content":"Count from one to three."}],"max_tokens":32,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}' \
  -o "${tmp_dir}/stream.txt"
grep -q '^data:' "${tmp_dir}/stream.txt"

echo "engine=qwen38-27b smoke=pass endpoint=${base}"

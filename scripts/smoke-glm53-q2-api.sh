#!/usr/bin/env bash
set -euo pipefail

base_url=${1:-http://127.0.0.1:8010}

models=$(curl -fsS --max-time 10 "$base_url/v1/models")
grep -Fq '"name":"GLM 5.3 Flash"' <<<"$models"

chat=$(curl -fsS --max-time 60 \
  -H 'Content-Type: application/json' \
  --data '{"model":"glm-5.2-chat","messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],"temperature":0,"max_tokens":32}' \
  "$base_url/v1/chat/completions")
grep -Fq '"content":"The capital of France is Paris."' <<<"$chat"
grep -Fq '"finish_reason":"stop"' <<<"$chat"

responses=$(curl -fsS --max-time 60 \
  -H 'Content-Type: application/json' \
  --data '{"model":"glm-5.2-chat","input":"Name the largest ocean on Earth in one short sentence.","temperature":0,"max_output_tokens":24}' \
  "$base_url/v1/responses")
grep -Fq '"status":"completed"' <<<"$responses"
grep -Fq '"text":"The largest ocean on Earth is the Pacific Ocean."' <<<"$responses"

stream=$(curl -NfsS --max-time 60 \
  -H 'Content-Type: application/json' \
  --data '{"model":"glm-5.2-chat","messages":[{"role":"user","content":"Reply with exactly four words about the sky."}],"temperature":0,"max_tokens":16,"stream":true}' \
  "$base_url/v1/chat/completions")
grep -Fq '"object":"chat.completion.chunk"' <<<"$stream"
grep -Fq 'data: [DONE]' <<<"$stream"

echo "GLM-5.3 Q2 API smoke passed at $base_url"

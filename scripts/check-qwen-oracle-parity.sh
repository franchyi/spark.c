#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
model_root=${SPARKSERVE_QWEN_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}
input_token=${SPARKSERVE_QWEN_INPUT_TOKEN:-9707}
steps=${SPARKSERVE_QWEN_PARITY_STEPS:-4}
oracle_url=${SPARKSERVE_FLASH_URL:-http://127.0.0.1:8890}
oracle_container=${SPARKSERVE_FLASH_CONTAINER:-qwen38-flash-next-smoke}
native_binary="$repo_root/build/bin/qwen_decode"

if [[ ! -x "$native_binary" ]]; then
  echo "missing native Qwen decoder: $native_binary" >&2
  exit 1
fi
if ! command -v jq >/dev/null; then
  echo "Qwen parity requires jq" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$repo_root/build:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
native_log=$(mktemp)
oracle_json=$(mktemp)
oracle_started=0
cleanup() {
  rm -f "$native_log" "$oracle_json"
  if [[ $oracle_started -eq 1 ]] &&
     docker ps --format '{{.Names}}' | grep -Fxq "$oracle_container"; then
    docker stop "$oracle_container" >/dev/null || true
    echo "stopped parity oracle $oracle_container; container and logs retained"
  fi
}
trap cleanup EXIT

"$native_binary" "$model_root" "$input_token" "$steps" | tee "$native_log"
mapfile -t native_ids < <(
  sed -n 's/.* output \([0-9][0-9]*\) .*/\1/p' "$native_log"
)
if [[ ${#native_ids[@]} -ne $steps ]]; then
  echo "native decoder returned ${#native_ids[@]} tokens, expected $steps" >&2
  exit 1
fi

"$script_dir/run-flash-next-smoke.sh"
oracle_started=1
ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 "$oracle_url/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [[ $ready -ne 1 ]]; then
  echo "Qwen SGLang oracle did not become ready at $oracle_url" >&2
  exit 1
fi

curl -fsS --max-time 900 \
  -H 'Content-Type: application/json' \
  --data "{\"input_ids\":[$input_token],\"sampling_params\":{\"temperature\":0,\"top_p\":1,\"top_k\":-1,\"max_new_tokens\":$steps,\"ignore_eos\":true},\"return_logprob\":true}" \
  "$oracle_url/generate" >"$oracle_json"
mapfile -t oracle_ids < <(jq -er '.output_ids[]' "$oracle_json")
if [[ ${#oracle_ids[@]} -ne $steps ]]; then
  echo "oracle returned ${#oracle_ids[@]} tokens, expected $steps" >&2
  jq . "$oracle_json" >&2
  exit 1
fi

for ((index = 0; index < steps; index++)); do
  if [[ "${native_ids[index]}" != "${oracle_ids[index]}" ]]; then
    echo "Qwen oracle mismatch at generated position $index: native ${native_ids[index]}, oracle ${oracle_ids[index]}" >&2
    exit 1
  fi
done

echo "Qwen raw-token oracle parity passed: input $input_token -> ${native_ids[*]}"

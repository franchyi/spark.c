#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
engine_dir=$(cd "$script_dir/.." && pwd)
model=${1:-${SPARK_ENGINE_MODEL:-${GLM53_Q2_MODEL:-/home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf}}}

benchmark="$repo_root/build/glm-5.3-flash-q2/ds4-bench"
prompt="$engine_dir/native/bench/promessi_sposi.txt"
if [[ ! -x "$benchmark" || ! -f "$prompt" ]]; then
  echo "missing Q2 benchmark build or prompt; run 'make build' in the GLM model directory" >&2
  exit 1
fi

exec "$benchmark" \
  --cuda \
  -m "$model" \
  --prompt-file "$prompt" \
  --ctx-start 2048 \
  --ctx-max 2048 \
  --ctx-alloc 2177 \
  --gen-tokens 128

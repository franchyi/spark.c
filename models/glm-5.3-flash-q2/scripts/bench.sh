#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
source_root=${DS4_GLM53_Q2_ROOT:-"$repo_root/vendor/_deps/ds4-glm53-q2"}
model=${1:-${SPARK_ENGINE_MODEL:-${GLM53_Q2_MODEL:-/home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf}}}

benchmark="$source_root/ds4-bench"
prompt="$source_root/speed-bench/promessi_sposi.txt"
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

#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
checkpoint=${SPARK_ENGINE_MODEL:?set SPARK_ENGINE_MODEL to the Qwen3.8-27B snapshot}
sidecar=${SPARK_ENGINE_SIDECAR:?set SPARK_ENGINE_SIDECAR to q27-scales-v1.bin}
bind=${SPARK_ENGINE_BIND:-0.0.0.0:${SPARK_ENGINE_PORT:-30000}}
model_id=${SPARK_ENGINE_MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}
capacity=${SPARK_ENGINE_CONTEXT_CAPACITY:-4096}

export LD_LIBRARY_PATH="$repo_root/build/q27:${LD_LIBRARY_PATH:-}"
exec "$repo_root/build/bin/q27-serve" \
  "$checkpoint" "$sidecar" "$bind" "$model_id" "$capacity"

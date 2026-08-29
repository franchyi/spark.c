#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
checkpoint=${SPARK_ENGINE_MODEL:?set SPARK_ENGINE_MODEL to the Qwen3.8-27B snapshot}
sidecar=${SPARK_ENGINE_SIDECAR:?set SPARK_ENGINE_SIDECAR to q27-scales-v1.bin}
token=${SPARK_ENGINE_TOKEN:-248045}
capacity=${SPARK_ENGINE_CONTEXT_CAPACITY:-1}

export LD_LIBRARY_PATH="$repo_root/build/q27:${LD_LIBRARY_PATH:-}"
exec "$repo_root/build/bin/q27-eager" \
  "$checkpoint" "$sidecar" "$token" "$capacity"

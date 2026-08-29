#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
model=${SPARK_ENGINE_MODEL:?set SPARK_ENGINE_MODEL to the local Qwen3.8-27B snapshot directory}
exec "$repo_root/build/bin/q27-inspect" "$model"

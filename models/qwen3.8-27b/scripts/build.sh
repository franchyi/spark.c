#!/usr/bin/env bash
set -euo pipefail

model_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"$model_dir/engine/tools/build-target.sh"
"$model_dir/engine/tools/build-dflash2.sh"

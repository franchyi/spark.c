#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
source_root=${1:-"$repo_root/vendor/_deps/ds4-glm53-q2"}
jobs=${JOBS:-4}
revision=a60a2a0d25137a849a101e04e86ea830a346073a

"$repo_root/models/glm-5.3-flash-q2/tools/fetch-ds4.sh" "$source_root"

actual=$(git -C "$source_root" rev-parse HEAD)
if [[ "$actual" != "$revision" ]]; then
  echo "ds4 GLM-5.3 revision mismatch: $actual != $revision" >&2
  exit 1
fi

make -C "$source_root" ds4-server ds4-bench CUDA_ARCH=sm_121 -j"$jobs"
echo "GLM-5.3 Q2 binaries ready in $source_root"

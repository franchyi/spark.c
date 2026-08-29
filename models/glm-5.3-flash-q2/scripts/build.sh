#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
native_root="$repo_root/models/glm-5.3-flash-q2/native"
jobs=${JOBS:-4}

make -C "$native_root" verify
make -C "$native_root" server bench CUDA_ARCH=sm_121 -j"$jobs"
echo "GLM-5.3 Q2 binaries ready in $repo_root/build/glm-5.3-flash-q2"

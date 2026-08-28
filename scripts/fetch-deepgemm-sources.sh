#!/usr/bin/env bash
set -euo pipefail

deepgemm_revision=fa3a5ca07d768dd0f9089f70a445208b166c48d1
cutlass_revision=f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
github_proxy=https://ghfast.top/https://github.com/
destination=${1:-third_party/_deps/deepgemm}

if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
  echo "DeepGEMM source destination exists but is not a git checkout: $destination" >&2
  exit 1
fi

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$destination"
  git -C "$destination" init -q
  git -C "$destination" remote add origin \
    "${github_proxy}sgl-project/DeepGEMM.git"
fi

if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to replace modified DeepGEMM sources in $destination" >&2
  exit 1
fi

if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$deepgemm_revision" ]]; then
  git -C "$destination" fetch --depth=1 origin "$deepgemm_revision"
  git -C "$destination" checkout --detach -q FETCH_HEAD
fi

git -C "$destination" \
  -c "url.${github_proxy}.insteadOf=https://github.com/" \
  submodule update --init --depth=1 third-party/cutlass

actual_cutlass=$(git -C "$destination/third-party/cutlass" rev-parse HEAD)
if [[ "$actual_cutlass" != "$cutlass_revision" ]]; then
  echo "CUTLASS revision mismatch: $actual_cutlass != $cutlass_revision" >&2
  exit 1
fi

manifest=$(cd "$(dirname "$0")/.." && pwd)/third_party/deepgemm-mqa/source-files.sha256
(cd "$destination" && sha256sum -c "$manifest")
echo "DeepGEMM paged-MQA sources ready: $destination"

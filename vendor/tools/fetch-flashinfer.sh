#!/usr/bin/env bash
set -euo pipefail

flashinfer_revision=906181e3f4cf4bcc81835fb480db4011bbd80b62
cutlass_revision=b46b16d003484063bca4ed365e44095c4c6ed633
github_proxy=https://ghfast.top/https://github.com/
destination=${1:-vendor/_deps/flashinfer}

if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
  echo "kernel source destination exists but is not a git checkout: $destination" >&2
  exit 1
fi

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$destination"
  git -C "$destination" init -q
  git -C "$destination" remote add origin \
    "${github_proxy}flashinfer-ai/flashinfer.git"
fi

if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to replace modified kernel sources in $destination" >&2
  exit 1
fi

if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$flashinfer_revision" ]]; then
  git -C "$destination" fetch --depth=1 origin "$flashinfer_revision"
  git -C "$destination" checkout --detach -q FETCH_HEAD
fi

git -C "$destination" \
  -c "url.${github_proxy}.insteadOf=https://github.com/" \
  submodule update --init --depth=1 3rdparty/cutlass

actual_cutlass=$(git -C "$destination/3rdparty/cutlass" rev-parse HEAD)
if [[ "$actual_cutlass" != "$cutlass_revision" ]]; then
  echo "CUTLASS revision mismatch: $actual_cutlass != $cutlass_revision" >&2
  exit 1
fi

vendor_root=$(cd "$(dirname "$0")/.." && pwd)
manifest="$vendor_root/flashinfer-nvfp4/source-files.sha256"
(cd "$destination" && sha256sum -c "$manifest")
xqa_manifest="$vendor_root/flashinfer-xqa/source-files.sha256"
(cd "$destination" && sha256sum -c "$xqa_manifest")
echo "kernel sources ready: $destination"

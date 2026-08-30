#!/usr/bin/env bash
set -euo pipefail

flashinfer_revision=906181e3f4cf4bcc81835fb480db4011bbd80b62
cutlass_revision=b46b16d003484063bca4ed365e44095c4c6ed633
destination=${1:-${HOME}/.cache/spark-c/sources/flashinfer}

if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
  echo "kernel source destination exists but is not a git checkout: $destination" >&2
  exit 1
fi

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$destination"
  git -C "$destination" init -q
  git -C "$destination" remote add origin https://github.com/flashinfer-ai/flashinfer.git
fi
git -C "$destination" remote set-url origin https://github.com/flashinfer-ai/flashinfer.git

if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to replace modified kernel sources in $destination" >&2
  exit 1
fi

if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$flashinfer_revision" ]]; then
  git -C "$destination" fetch --depth=1 origin "$flashinfer_revision"
  git -C "$destination" checkout --detach -q FETCH_HEAD
fi

git -C "$destination" submodule update --init --depth=1 3rdparty/cutlass

actual_cutlass=$(git -C "$destination/3rdparty/cutlass" rev-parse HEAD)
if [[ "$actual_cutlass" != "$cutlass_revision" ]]; then
  echo "CUTLASS revision mismatch: $actual_cutlass != $cutlass_revision" >&2
  exit 1
fi

repo_root=$(cd "$(dirname "$0")/.." && pwd)
manifest="$repo_root/tools/flashinfer-nvfp4.sha256"
(cd "$destination" && sha256sum -c "$manifest")
xqa_manifest="$repo_root/tools/flashinfer-xqa.sha256"
(cd "$destination" && sha256sum -c "$xqa_manifest")
echo "kernel sources ready: $destination"

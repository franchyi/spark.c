#!/usr/bin/env bash
set -euo pipefail

revision=c4271c3fe1262fc2adbd162c33b25de5255251c5
proxy=https://ghfast.top/https://github.com/
destination=${1:-${HOME}/.cache/spark-c/sources/sglang-c427}

if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
  echo "c427 source destination is not a git checkout: $destination" >&2
  exit 1
fi
if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$destination"
  git -C "$destination" init -q
  git -C "$destination" remote add origin "${proxy}sgl-project/sglang.git"
fi
if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to replace modified c427 sources in $destination" >&2
  exit 1
fi
if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]]; then
  git -C "$destination" fetch --depth=1 origin "$revision"
  git -C "$destination" checkout --detach -q FETCH_HEAD
fi
echo "c427 sources ready: $destination"

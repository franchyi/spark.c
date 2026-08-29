#!/usr/bin/env bash
set -euo pipefail

revision=a60a2a0d25137a849a101e04e86ea830a346073a
github_proxy=https://ghfast.top/https://github.com/
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
destination=${1:-"$repo_root/vendor/_deps/ds4-glm53-q2"}

if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
  echo "ds4 GLM-5.3 source destination exists but is not a git checkout: $destination" >&2
  exit 1
fi

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$destination"
  git -C "$destination" init -q
  git -C "$destination" remote add origin \
    "${github_proxy}antirez/ds4.git"
fi

if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
  echo "refusing to replace modified ds4 GLM-5.3 sources in $destination" >&2
  exit 1
fi

if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]]; then
  git -C "$destination" fetch --depth=1 origin "$revision"
  git -C "$destination" checkout --detach -q FETCH_HEAD
fi

actual=$(git -C "$destination" rev-parse HEAD)
if [[ "$actual" != "$revision" ]]; then
  echo "ds4 GLM-5.3 revision mismatch: $actual != $revision" >&2
  exit 1
fi

manifest="$repo_root/vendor/ds4-glm53/source-files.sha256"
(cd "$destination" && sha256sum -c "$manifest")
echo "ds4 GLM-5.3 sources ready: $destination"

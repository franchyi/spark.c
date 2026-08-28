#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
source_root=${DS4_GLM53_Q2_ROOT:-"$repo_root/third_party/_deps/ds4-glm53-q2"}
model=${1:-${GLM53_Q2_MODEL:-}}

if [[ -z "$model" ]]; then
  echo "usage: $0 MODEL" >&2
  exit 2
fi

server="$source_root/ds4-server"
if [[ ! -x "$server" ]]; then
  echo "missing $server; run scripts/build-glm53-q2.sh first" >&2
  exit 1
fi

expected_bytes=96505816384
actual_bytes=$(stat -c %s "$model")
if [[ "$actual_bytes" != "$expected_bytes" ]]; then
  echo "GLM-5.3 Q2 size mismatch: $actual_bytes != $expected_bytes" >&2
  exit 1
fi

if [[ ${GLM53_Q2_VERIFY_SHA:-0} == 1 ]]; then
  expected_sha=e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32
  actual_sha=$(sha256sum "$model" | awk '{print $1}')
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "GLM-5.3 Q2 SHA-256 mismatch: $actual_sha != $expected_sha" >&2
    exit 1
  fi
fi

minimum_available_kib=$((110 * 1024 * 1024))
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < minimum_available_kib )) &&
   [[ ${GLM53_Q2_ALLOW_LOW_MEMORY:-0} != 1 ]]; then
  available_gib=$(awk -v kib="$available_kib" 'BEGIN {printf "%.1f", kib / 1048576}')
  echo "resident Q2 requires about 110 GiB MemAvailable before startup; found ${available_gib} GiB" >&2
  echo "stop nonessential services or set GLM53_Q2_ALLOW_LOW_MEMORY=1 to bypass the preflight" >&2
  exit 1
fi

host=${GLM53_Q2_HOST:-127.0.0.1}
port=${GLM53_Q2_PORT:-8010}
context=${GLM53_Q2_CONTEXT:-2048}
max_tokens=${GLM53_Q2_MAX_TOKENS:-128}

exec "$server" \
  --cuda \
  -c "$context" \
  -n "$max_tokens" \
  -m "$model" \
  --host "$host" \
  --port "$port"

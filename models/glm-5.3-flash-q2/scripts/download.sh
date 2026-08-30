#!/usr/bin/env bash
set -euo pipefail

revision="d0d6394cad1046c6d8ad87fa9b0939b4760cb94f"
filename="GLM-5.3-Flash-Q2.gguf"
destination="${SPARK_ENGINE_MODEL:-${HOME}/models/antirez/glm-5.3-flash-gguf/${filename}}"
endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
expected_bytes=96505816384
expected_sha="e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32"

mkdir -p "$(dirname -- "${destination}")"
curl -fL -C - \
  -o "${destination}" \
  "${endpoint}/antirez/glm-5.3-flash-gguf/resolve/${revision}/${filename}?download=true"

actual_bytes="$(stat -c %s "${destination}")"
if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
  echo "GLM-5.3 Q2 size mismatch: ${actual_bytes} != ${expected_bytes}" >&2
  exit 1
fi
actual_sha="$(sha256sum "${destination}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
  echo "GLM-5.3 Q2 SHA-256 mismatch: ${actual_sha} != ${expected_sha}" >&2
  exit 1
fi
echo "model=${destination} bytes=${actual_bytes} sha256=${actual_sha}"

#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ENGINE_DIR}/../.." && pwd)"
model="${SPARK_ENGINE_MODEL:-/home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf}"
export GLM53_Q2_MODEL="${model}"
echo "engine=glm53-q2 revision=a60a2a0d25137a849a101e04e86ea830a346073a checkpoint=${model} prompt_tokens=2048 generated_tokens=128 concurrency=1"
exec "${REPO_ROOT}/scripts/bench-glm53-q2.sh" "${model}"

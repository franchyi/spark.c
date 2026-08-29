#!/usr/bin/env bash
set -euo pipefail

name="sparkserve-qwen38-flash-oracle"
if ! docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
  echo "container ${name} does not exist; nothing to stop"
  exit 0
fi
if docker ps --format '{{.Names}}' | grep -qx "${name}"; then
  docker stop "${name}" >/dev/null
  echo "stopped ${name}; container and logs retained"
else
  echo "container ${name} is already stopped"
fi

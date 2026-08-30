SHELL := /usr/bin/env bash

.PHONY: download build prepare setup serve smoke bench stop

download:
	@../../spark download qwen27

build:
	@./scripts/build.sh

prepare:
	@../../spark prepare qwen27

setup:
	@../../spark setup qwen27

serve:
	@../../spark serve qwen27

smoke:
	@./scripts/smoke.sh

bench:
	@python3 ./scripts/bench.py --engine spark-q27 \
		--case "$${Q27_BENCH_CASE:-decode}" --sample 1 \
		--base-url "$${SPARK_ENGINE_BASE_URL:-http://127.0.0.1:30000/v1}"

stop:
	@./scripts/stop.sh

SHELL := /usr/bin/env bash

.PHONY: build serve smoke bench stop

build:
	@./scripts/build.sh

serve:
	@./scripts/serve.sh

smoke:
	@./scripts/smoke.sh

bench:
	@python3 ./scripts/bench.py --engine spark-q27 \
		--case "$${Q27_BENCH_CASE:-decode}" --sample 1 \
		--base-url "$${SPARK_ENGINE_BASE_URL:-http://127.0.0.1:30000/v1}"

stop:
	@./scripts/stop.sh

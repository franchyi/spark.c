SHELL := /usr/bin/env bash

.PHONY: build inspect eager test-fp8 serve smoke bench stop provenance fetch-oracle \
	oracle-serve oracle-smoke oracle-bench oracle-stop

build:
	@./scripts/build-native.sh

inspect:
	@./scripts/inspect-native.sh

eager:
	@./scripts/run-eager.sh

test-fp8:
	@$(MAKE) -C native test-fp8

fetch-oracle:
	@./scripts/fetch-oracle.sh

serve:
	@./scripts/serve-native.sh

smoke:
	@./scripts/smoke-native.sh

bench:
	@./scripts/bench-native.py

stop:
	@./scripts/stop-native.sh

oracle-serve: fetch-oracle
	@./scripts/serve.sh

oracle-smoke:
	@./scripts/smoke.sh

oracle-bench:
	@./scripts/bench.sh

oracle-stop:
	@./scripts/stop.sh

provenance:
	@sed -n '1,240p' VENDOR.md

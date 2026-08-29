SHELL := /usr/bin/env bash

.PHONY: build serve smoke bench stop provenance fetch-oracle

build: fetch-oracle

fetch-oracle:
	@./scripts/fetch-oracle.sh

serve:
	@./scripts/serve.sh

smoke:
	@./scripts/smoke.sh

bench:
	@./scripts/bench.sh

stop:
	@./scripts/stop.sh

provenance:
	@sed -n '1,240p' VENDOR.md

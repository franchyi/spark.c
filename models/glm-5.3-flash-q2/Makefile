SHELL := /usr/bin/env bash

.PHONY: build download serve smoke bench stop provenance

build:
	@./scripts/build.sh

download:
	@./scripts/download.sh

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

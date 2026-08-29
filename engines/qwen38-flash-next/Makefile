SHELL := /usr/bin/env bash

.PHONY: build serve smoke bench stop parity provenance \
	fetch-oracle oracle-build oracle-download oracle-serve oracle-smoke oracle-stop

build:
	@./scripts/build-native.sh

serve:
	@./scripts/serve-native.sh

smoke:
	@./scripts/smoke-native.sh

bench:
	@./scripts/bench-native.sh

stop:
	@./scripts/stop.sh

parity:
	@./scripts/parity-native.sh

fetch-oracle:
	@./scripts/fetch-oracle.sh

oracle-build: fetch-oracle
	@./scripts/build-oracle.sh

oracle-download: oracle-build
	@./scripts/download-oracle.sh

oracle-serve: fetch-oracle
	@./scripts/serve-oracle.sh

oracle-smoke: fetch-oracle
	@./scripts/smoke-oracle.sh

oracle-stop:
	@./scripts/stop-oracle.sh

provenance:
	@sed -n '1,260p' VENDOR.md

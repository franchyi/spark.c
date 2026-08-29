SHELL := /usr/bin/env bash

.PHONY: build index-ple serve smoke bench stop provenance \
	fetch-oracle oracle-build oracle-download oracle-serve oracle-smoke oracle-stop

build:
	@./scripts/build-native.sh

index-ple:
	@python3 native/tools/ple_index.py \
		"$${SPARK_ENGINE_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}" \
		--revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594

serve:
	@./scripts/serve-native.sh

smoke:
	@./scripts/smoke-native.sh

bench:
	@./scripts/bench-native.sh

stop:
	@./scripts/stop.sh

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

SHELL := /usr/bin/env bash

.PHONY: build build-fused index-ple serve smoke bench stop

build:
	@./scripts/build.sh

build-fused:
	@./engine/tools/build-fused-server.sh

index-ple:
	@python3 engine/tools/ple_index.py \
		"$${SPARK_ENGINE_MODEL:-/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4}" \
		--revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594

serve:
	@./scripts/serve.sh

smoke:
	@./scripts/smoke.sh

bench:
	@./scripts/bench.sh

stop:
	@./scripts/stop.sh

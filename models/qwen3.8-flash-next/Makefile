SHELL := /usr/bin/env bash

.PHONY: download build build-fused prepare setup serve smoke bench stop

download:
	@../../spark download flash-next

build:
	@./scripts/build.sh

build-fused:
	@./engine/tools/build-fused-server.sh

prepare:
	@../../spark prepare flash-next

setup:
	@../../spark setup flash-next

serve:
	@../../spark serve flash-next

smoke:
	@./scripts/smoke.sh

bench:
	@./scripts/bench.sh

stop:
	@./scripts/stop.sh

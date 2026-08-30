SHELL := /usr/bin/env bash

.PHONY: build download prepare setup serve smoke bench stop

build:
	@./scripts/build.sh

download:
	@../../spark download glm

prepare:
	@../../spark prepare glm

setup:
	@../../spark setup glm

serve:
	@../../spark serve glm

smoke:
	@./scripts/smoke.sh

bench:
	@./scripts/bench.sh

stop:
	@./scripts/stop.sh

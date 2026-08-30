.DEFAULT_GOAL := help

.PHONY: help list

help:
	@echo "Spark.C quick deployment"
	@echo "  ./spark setup {qwen27|flash-next|glm}"
	@echo "  ./spark serve {qwen27|flash-next|glm}"
	@echo "  ./spark run   {qwen27|flash-next|glm}  # setup + serve"
	@echo "Run './spark --help' for paths, ports, and lower-level commands."

list:
	@echo qwen3.8-27b
	@echo qwen3.8-flash-next
	@echo glm-5.3-flash-q2

qwen27-%:
	@$(MAKE) --no-print-directory -C models/qwen3.8-27b $*

flash-%:
	@$(MAKE) --no-print-directory -C models/qwen3.8-flash-next $*

glm-%:
	@$(MAKE) --no-print-directory -C models/glm-5.3-flash-q2 $*

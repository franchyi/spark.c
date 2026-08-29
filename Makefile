.DEFAULT_GOAL := help

.PHONY: help list

help:
	@echo "Spark.C model commands"
	@echo "  make qwen27-{build|serve|smoke|bench|stop|provenance}"
	@echo "  make flash-{build|serve|smoke|bench|stop|provenance}"
	@echo "  make glm-{build|download|serve|smoke|bench|stop|provenance}"

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

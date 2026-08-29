SHELL := /usr/bin/env bash

NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -O3 -std=c++20 -arch=$(CUDA_ARCH) -lineinfo
REPO_ROOT := ../../..
BUILD_DIR := $(REPO_ROOT)/build/q27
INCLUDES := -Iinclude
KERNEL_SHARED := $(BUILD_DIR)/libq27-kernels.so
FP8_FIXTURE_TEST := $(BUILD_DIR)/q27-fp8-fixture-test

.PHONY: kernels test-fp8 clean

kernels: $(KERNEL_SHARED)

test-fp8: $(FP8_FIXTURE_TEST)
	test -n "$(Q27_FP8_FIXTURE)"
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(FP8_FIXTURE_TEST) "$(Q27_FP8_FIXTURE)"

$(BUILD_DIR):
	mkdir -p $@

$(KERNEL_SHARED): csrc/q27_fp8_decode.cu include/q27_kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) --shared -Xcompiler -fPIC \
		csrc/q27_fp8_decode.cu -o $@

$(FP8_FIXTURE_TEST): fixtures/q27_fp8_fixture_test.cu include/q27_kernels.h $(KERNEL_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) fixtures/q27_fp8_fixture_test.cu \
		-L$(BUILD_DIR) -lq27-kernels -o $@

clean:
	rm -rf $(BUILD_DIR)

SHELL := /usr/bin/env bash

CC ?= cc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_121
BUILD_DIR ?= ../../../build/glm-5.3-flash-q2
MMQ_BUILD_DIR := $(BUILD_DIR)/cuda/mmq

HOST_CPU_FLAG ?= -march=native
INCLUDES := -Iinclude -Icuda/mmq
CFLAGS ?= -O3 -ffast-math -g $(HOST_CPU_FLAG) -Wall -Wextra -std=c99 \
	-D_GNU_SOURCE -fno-finite-math-only $(INCLUDES)

ifeq ($(filter sm_121 sm_121a,$(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
else
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a \
	-DDS4_CUDA_HAVE_MXF4=1
endif

NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) \
	-Xcompiler $(HOST_CPU_FLAG) -Xcompiler -pthread $(INCLUDES)
CUDA_LDLIBS ?= -lm -Xcompiler -pthread \
	-L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 \
	-lcudart -lcublas

MMQ_OBJS := \
	$(MMQ_BUILD_DIR)/ggml_stubs.o \
	$(MMQ_BUILD_DIR)/mmq_runtime.o \
	$(MMQ_BUILD_DIR)/mmq_d2r.o \
	$(MMQ_BUILD_DIR)/quantize.o \
	$(MMQ_BUILD_DIR)/mmid.o \
	$(MMQ_BUILD_DIR)/mmvq.o \
	$(MMQ_BUILD_DIR)/repack.o

CORE_OBJS := \
	$(BUILD_DIR)/engine.o \
	$(BUILD_DIR)/distributed.o \
	$(BUILD_DIR)/tensor_parallel.o \
	$(BUILD_DIR)/storage.o \
	$(BUILD_DIR)/gpu.o \
	$(BUILD_DIR)/layer_pack.o \
	$(MMQ_OBJS)

SERVER := $(BUILD_DIR)/glm-server
BENCH := $(BUILD_DIR)/glm-bench

.PHONY: all server bench clean

all: server bench
server: $(SERVER)
bench: $(BENCH)

$(SERVER): $(BUILD_DIR)/server.o $(BUILD_DIR)/cli.o \
		$(BUILD_DIR)/kv_store.o $(BUILD_DIR)/radix_tree.o \
		$(BUILD_DIR)/gpu_args.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(BENCH): $(BUILD_DIR)/benchmark.o $(BUILD_DIR)/cli.o \
		$(BUILD_DIR)/gpu_args.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(BUILD_DIR) $(MMQ_BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/gpu.o: cuda/runtime.cu cuda/iq2_tables.inc \
		include/gpu.h include/multi_gpu.h cuda/mmq/mmq_runtime.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(MMQ_BUILD_DIR)/%.o: cuda/mmq/%.cu | $(MMQ_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

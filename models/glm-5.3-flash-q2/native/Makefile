SHELL := /usr/bin/env bash

# Exact GLM-5.3 Q2 CUDA build for DGX Spark. The source tree is pinned ds4;
# this Makefile intentionally exposes only the shipping server and benchmark.
CC ?= cc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_121
BUILD_DIR ?= ../../../build/glm-5.3-flash-q2
MMQ_BUILD_DIR := $(BUILD_DIR)/cuda/mmq

NATIVE_CPU_FLAG ?= -march=native
CFLAGS ?= -O3 -ffast-math -g $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 \
	-D_GNU_SOURCE -fno-finite-math-only

ifeq ($(filter sm_121 sm_121a,$(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
else
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a \
	-DDS4_CUDA_HAVE_MXF4=1
endif

NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) \
	-Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread \
	-L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 \
	-lcudart -lcublas
MMQ_INCLUDES := -Icuda/mmq

MMQ_OBJS := \
	$(MMQ_BUILD_DIR)/ds4_ggml_stubs.o \
	$(MMQ_BUILD_DIR)/ds4_mmq.o \
	$(MMQ_BUILD_DIR)/ds4_mmq_d2r.o \
	$(MMQ_BUILD_DIR)/quantize.o \
	$(MMQ_BUILD_DIR)/mmid.o \
	$(MMQ_BUILD_DIR)/mmvq.o \
	$(MMQ_BUILD_DIR)/ds4_repack.o

CORE_OBJS := \
	$(BUILD_DIR)/ds4.o \
	$(BUILD_DIR)/ds4_distributed.o \
	$(BUILD_DIR)/ds4_tp.o \
	$(BUILD_DIR)/ds4_ssd.o \
	$(BUILD_DIR)/ds4_cuda.o \
	$(BUILD_DIR)/ds4_layer_pack.o \
	$(MMQ_OBJS)

SERVER := $(BUILD_DIR)/ds4-server
BENCH := $(BUILD_DIR)/ds4-bench

.PHONY: all server bench verify clean

all: server bench
server: $(SERVER)
bench: $(BENCH)

verify:
	sha256sum -c SOURCE.sha256

$(SERVER): $(BUILD_DIR)/ds4_server.o $(BUILD_DIR)/ds4_help.o \
		$(BUILD_DIR)/ds4_kvstore.o $(BUILD_DIR)/rax.o \
		$(BUILD_DIR)/ds4_gpu_args.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(BENCH): $(BUILD_DIR)/ds4_bench.o $(BUILD_DIR)/ds4_help.o \
		$(BUILD_DIR)/ds4_gpu_args.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(BUILD_DIR) $(MMQ_BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h \
		ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h \
		ds4_ssd.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_ssd.o: ds4_ssd.c ds4_ssd.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_layer_pack.o: ds4_layer_pack.c ds4_layer_pack.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_help.o: ds4_help.c ds4_help.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h \
		ds4_gpu_mgpu.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_server.o: ds4_server.c ds4.h ds4_ssd.h \
		ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h \
		ds4_distributed.h ds4_help.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h \
		ds4_ssd.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/rax.o: rax.c rax.h rax_malloc.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h \
		ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(MMQ_BUILD_DIR)/%.o: cuda/mmq/%.cu | $(MMQ_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

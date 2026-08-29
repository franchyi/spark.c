SHELL := /usr/bin/env bash

# Spark-only, model-specific build. This compiles the exact Qwen3.8-27B
# SM121 capsule; it is deliberately not a shape or backend dispatcher.
NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH := sm_121a
NVCCFLAGS ?= -O3 -std=c++20 -arch=$(CUDA_ARCH) -lineinfo
REPO_ROOT := ../../..
BUILD_DIR := $(REPO_ROOT)/build/q27
AOT_DIR := $(REPO_ROOT)/build/q27-aot
FLASHINFER_ROOT := $(REPO_ROOT)/vendor/_deps/flashinfer
FLASHINFER_INCLUDE := $(FLASHINFER_ROOT)/include
CUTLASS_ROOT := $(FLASHINFER_ROOT)/3rdparty/cutlass
TVM_FFI_ROOT ?=
CUDA_DIALECT_ARCHIVE ?=
CCCL_ROOT ?=

INCLUDES := -Iinclude
FLASHINFER_INCLUDES := -I$(FLASHINFER_INCLUDE) \
	-I$(CUTLASS_ROOT)/include \
	-I$(CUTLASS_ROOT)/tools/util/include
CCCL_INCLUDES := -I$(CCCL_ROOT)/libcudacxx/include \
	-I$(CCCL_ROOT)/cub -I$(CCCL_ROOT)/thrust
FLASHINFER_FLAGS := -D__CUDA_ARCH_SPECIFIC__ --expt-relaxed-constexpr \
	-diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20015 \
	-diag-suppress 2908
SHARED_FLAGS := --shared -Xcompiler=-fPIC
ORIGIN_RPATH := -Xlinker=-rpath -Xlinker='$$ORIGIN'

GDN_AOT_OBJECT := $(AOT_DIR)/gdn/gdn_bf16_t1_h16_hv48_k128_v128_sm121.o
NVFP4_K5120_AOT_OBJECT := \
	$(AOT_DIR)/quantize/swizzled_bfloat16_k5120_sf0_pdl0.o
NVFP4_K17408_AOT_OBJECT := \
	$(AOT_DIR)/quantize/swizzled_bfloat16_k17408_sf0_pdl0.o
AOT_OBJECTS := $(GDN_AOT_OBJECT) $(NVFP4_K5120_AOT_OBJECT) \
	$(NVFP4_K17408_AOT_OBJECT)
AOT_MANIFEST := $(REPO_ROOT)/vendor/q27-aot-sm121.sha256
AOT_STAMP := $(AOT_DIR)/.verified

KERNEL_SHARED := $(BUILD_DIR)/libq27-kernels.so
MAPPING_SHARED := $(BUILD_DIR)/libq27-mapping.so
GDN_SHARED := $(BUILD_DIR)/libq27-gdn.so
NVFP4_SHARED := $(BUILD_DIR)/libq27-nvfp4.so
MLP_SHARED := $(BUILD_DIR)/libq27-mlp.so
ATTENTION_SHARED := $(BUILD_DIR)/libq27-attention.so
GDN_BLOCK_SHARED := $(BUILD_DIR)/libq27-gdn-block.so
MODEL_SHARED := $(BUILD_DIR)/libq27-model.so
LM_HEAD_BF16_SHARED := $(BUILD_DIR)/libq27-lm-head-bf16.so
PREFILL_CORE_SHARED := $(BUILD_DIR)/libq27-prefill-core.so
PREFILL_FP8_SHARED := $(BUILD_DIR)/libq27-prefill-fp8.so
PREFILL_NVFP4_SHARED := $(BUILD_DIR)/libq27-prefill-nvfp4.so
PREFILL_MLP_SHARED := $(BUILD_DIR)/libq27-prefill-mlp.so
PREFILL_ATTENTION_SHARED := $(BUILD_DIR)/libq27-prefill-attention.so
PREFILL_ATTENTION_LAYER_SHARED := \
	$(BUILD_DIR)/libq27-prefill-attention-layer.so
GDN_PREFILL_SHARED := $(BUILD_DIR)/libq27-gdn-prefill.so
GDN_PREFILL_AB_SHARED := $(BUILD_DIR)/libq27-gdn-prefill-ab.so
GDN_PREFILL_WY_SHARED := $(BUILD_DIR)/libq27-gdn-prefill-wy.so
GDN_PREFILL_SUBLAYER_SHARED := \
	$(BUILD_DIR)/libq27-gdn-prefill-sublayer.so
GDN_PREFILL_LAYER_SHARED := $(BUILD_DIR)/libq27-gdn-prefill-layer.so
GDN_PREFILL_M512_SHARED := $(BUILD_DIR)/libq27-gdn-prefill-m512.so
PREFILL_MODEL_SHARED := $(BUILD_DIR)/libq27-prefill-model.so
PREFILL_SHARED := $(PREFILL_CORE_SHARED) $(PREFILL_FP8_SHARED) \
	$(PREFILL_NVFP4_SHARED) $(PREFILL_MLP_SHARED) \
	$(PREFILL_ATTENTION_SHARED) $(PREFILL_ATTENTION_LAYER_SHARED) \
	$(GDN_PREFILL_SHARED) $(GDN_PREFILL_AB_SHARED) \
	$(GDN_PREFILL_WY_SHARED) $(GDN_PREFILL_SUBLAYER_SHARED) \
	$(GDN_PREFILL_LAYER_SHARED) $(GDN_PREFILL_M512_SHARED) \
	$(PREFILL_MODEL_SHARED)
CAPSULE_SHARED := $(KERNEL_SHARED) $(MAPPING_SHARED) $(GDN_SHARED) \
	$(NVFP4_SHARED) $(MLP_SHARED) $(ATTENTION_SHARED) $(GDN_BLOCK_SHARED) \
	$(LM_HEAD_BF16_SHARED) $(MODEL_SHARED) $(PREFILL_SHARED)

FP8_FIXTURE_TEST := $(BUILD_DIR)/q27-fp8-fixture-test
CORE_FIXTURE_TEST := $(BUILD_DIR)/q27-core-fixture-test
LM_HEAD_BENCH := $(BUILD_DIR)/q27-lm-head-bench

.PHONY: all capsule-shared prefill-shared kernels mapping gdn nvfp4 mlp attention \
	gdn-block model lm-head-bf16 lm-head-bench verify-aot verify-linkage \
	test-core test-fp8 clean

all: capsule-shared

capsule-shared: $(CAPSULE_SHARED)
prefill-shared: $(PREFILL_SHARED)

kernels: $(KERNEL_SHARED)
mapping: $(MAPPING_SHARED)
gdn: $(GDN_SHARED)
nvfp4: $(NVFP4_SHARED)
mlp: $(MLP_SHARED)
attention: $(ATTENTION_SHARED)
gdn-block: $(GDN_BLOCK_SHARED)
model: $(MODEL_SHARED)
lm-head-bf16: $(LM_HEAD_BF16_SHARED)
lm-head-bench: $(LM_HEAD_BENCH)
verify-aot: $(AOT_STAMP)

verify-linkage: capsule-shared
	@set -euo pipefail; \
	for library in $(CAPSULE_SHARED); do \
		if ldd "$$library" | grep -q 'not found'; then \
			ldd "$$library"; \
			exit 1; \
		fi; \
	done

test-fp8: $(FP8_FIXTURE_TEST)
	test -n "$(Q27_FP8_FIXTURE)"
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(FP8_FIXTURE_TEST) "$(Q27_FP8_FIXTURE)"

test-core: $(CORE_FIXTURE_TEST)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(CORE_FIXTURE_TEST)

$(BUILD_DIR):
	mkdir -p $@

$(AOT_STAMP): $(AOT_OBJECTS) $(AOT_MANIFEST)
	cd $(REPO_ROOT) && sha256sum -c vendor/q27-aot-sm121.sha256
	touch $@

$(KERNEL_SHARED): cuda/q27_fp8_decode.cu cuda/q27_core.cu \
		cuda/q27_bf16_ab.cu include/q27_kernels.h include/q27_bf16_ab.h \
		| $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) \
		cuda/q27_fp8_decode.cu cuda/q27_core.cu cuda/q27_bf16_ab.cu \
		-lcublas -o $@

$(MAPPING_SHARED): cuda/q27_mapping.cc include/q27_mapping.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) \
		cuda/q27_mapping.cc -lcudart -o $@

$(GDN_SHARED): cuda/q27_gdn.cu cuda/q27_gdn_flashinfer.cc \
		cuda/q27_gdn_flashinfer.h include/q27_gdn.h $(AOT_STAMP) \
		| $(BUILD_DIR)
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUDA_DIALECT_ARCHIVE)"
	test -f "$(CUDA_DIALECT_ARCHIVE)"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Icuda -I$(TVM_FFI_ROOT)/include \
		$(SHARED_FLAGS) -Xlinker=-soname -Xlinker=$(notdir $@) \
		cuda/q27_gdn.cu cuda/q27_gdn_flashinfer.cc $(GDN_AOT_OBJECT) \
		$(CUDA_DIALECT_ARCHIVE) -L$(TVM_FFI_ROOT)/lib -ltvm_ffi \
		-lcuda -lcudart -ldl -Xcompiler=-pthread $(ORIGIN_RPATH) -o $@

$(NVFP4_SHARED): cuda/q27_nvfp4.cu include/q27_nvfp4.h \
		$(AOT_STAMP) | $(BUILD_DIR)
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUDA_DIALECT_ARCHIVE)"
	test -f "$(CUDA_DIALECT_ARCHIVE)"
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_FLAGS) $(INCLUDES) \
		$(CCCL_INCLUDES) $(FLASHINFER_INCLUDES) \
		-I$(TVM_FFI_ROOT)/include $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) cuda/q27_nvfp4.cu \
		$(NVFP4_K5120_AOT_OBJECT) $(NVFP4_K17408_AOT_OBJECT) \
		$(CUDA_DIALECT_ARCHIVE) -L$(TVM_FFI_ROOT)/lib -ltvm_ffi \
		-lcuda -lcudart -ldl -Xcompiler=-pthread $(ORIGIN_RPATH) -o $@

$(MLP_SHARED): cuda/q27_mlp.cu include/q27_mlp.h $(KERNEL_SHARED) \
		$(NVFP4_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) cuda/q27_mlp.cu \
		-L$(BUILD_DIR) -lq27-nvfp4 -lq27-kernels $(ORIGIN_RPATH) -o $@

$(ATTENTION_SHARED): cuda/q27_attention.cu cuda/q27_attention_flashinfer.cu \
		include/q27_attention.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_FLAGS) $(INCLUDES) \
		$(CCCL_INCLUDES) $(FLASHINFER_INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) \
		cuda/q27_attention.cu cuda/q27_attention_flashinfer.cu \
		-lcudart -o $@

$(GDN_BLOCK_SHARED): cuda/q27_gdn_block.cc include/q27_gdn_block.h \
		$(KERNEL_SHARED) $(GDN_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) cuda/q27_gdn_block.cc \
		-L$(BUILD_DIR) -lq27-gdn -lq27-kernels $(ORIGIN_RPATH) -o $@

$(MODEL_SHARED): cuda/q27_model.cu include/q27_model.h $(KERNEL_SHARED) \
		$(GDN_BLOCK_SHARED) $(MLP_SHARED) $(ATTENTION_SHARED) \
		$(LM_HEAD_BF16_SHARED) $(PREFILL_MODEL_SHARED) \
		$(PREFILL_NVFP4_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) cuda/q27_model.cu \
		-L$(BUILD_DIR) -lq27-gdn-block -lq27-gdn -lq27-mlp \
		-lq27-nvfp4 -lq27-attention -lq27-lm-head-bf16 -lq27-kernels \
		-lq27-prefill-model -lq27-prefill-nvfp4 -lcublas \
		$(ORIGIN_RPATH) -o $@

$(LM_HEAD_BF16_SHARED): cuda/q27_lm_head_bf16.cu \
		include/q27_lm_head_bf16.h include/q27_kernels.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) \
		cuda/q27_lm_head_bf16.cu -o $@

$(PREFILL_CORE_SHARED): cuda/q27_prefill_core.cu \
		include/q27_prefill_core.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< -lcudart -o $@

$(PREFILL_FP8_SHARED): cuda/q27_prefill_fp8.cu \
		include/q27_prefill_fp8.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-lcublasLt -lcublas -lcudart -o $@

$(PREFILL_NVFP4_SHARED): cuda/q27_prefill_nvfp4.cu \
		include/q27_prefill_nvfp4.h $(AOT_STAMP) | $(BUILD_DIR)
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUDA_DIALECT_ARCHIVE)"
	test -f "$(CUDA_DIALECT_ARCHIVE)"
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_FLAGS) $(INCLUDES) \
		$(CCCL_INCLUDES) $(FLASHINFER_INCLUDES) \
		-I$(TVM_FFI_ROOT)/include $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		$(NVFP4_K5120_AOT_OBJECT) $(NVFP4_K17408_AOT_OBJECT) \
		$(CUDA_DIALECT_ARCHIVE) -L$(TVM_FFI_ROOT)/lib -ltvm_ffi \
		-lcuda -lcudart -ldl -Xcompiler=-pthread $(ORIGIN_RPATH) -o $@

$(PREFILL_MLP_SHARED): cuda/q27_prefill_mlp.cu \
		include/q27_prefill_mlp.h $(PREFILL_NVFP4_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-L$(BUILD_DIR) -lq27-prefill-nvfp4 -lcudart \
		$(ORIGIN_RPATH) -o $@

$(PREFILL_ATTENTION_SHARED): cuda/q27_prefill_attention.cu \
		include/q27_prefill_attention.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_FLAGS) $(INCLUDES) \
		$(CCCL_INCLUDES) $(FLASHINFER_INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< -lcudart -o $@

$(PREFILL_ATTENTION_LAYER_SHARED): cuda/q27_prefill_attention_layer.cu \
		include/q27_prefill_attention_layer.h $(PREFILL_CORE_SHARED) \
		$(PREFILL_FP8_SHARED) $(PREFILL_ATTENTION_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-L$(BUILD_DIR) -lq27-prefill-core -lq27-prefill-fp8 \
		-lq27-prefill-attention -lcublasLt -lcublas -lcudart \
		$(ORIGIN_RPATH) -o $@

$(GDN_PREFILL_SHARED): cuda/q27_gdn_prefill.cu \
		include/q27_gdn_prefill.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-lcublas -lcudart $(ORIGIN_RPATH) -o $@

$(GDN_PREFILL_AB_SHARED): cuda/q27_gdn_prefill_ab.cu \
		include/q27_gdn_prefill_ab.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-lcublas -lcudart $(ORIGIN_RPATH) -o $@

$(GDN_PREFILL_WY_SHARED): cuda/q27_gdn_prefill_wy.cu \
		include/q27_gdn_prefill_wy.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-lcublas -lcudart $(ORIGIN_RPATH) -o $@

$(GDN_PREFILL_SUBLAYER_SHARED): cuda/q27_gdn_prefill_sublayer.cu \
		include/q27_gdn_prefill_sublayer.h $(GDN_PREFILL_SHARED) \
		$(GDN_PREFILL_AB_SHARED) $(GDN_PREFILL_WY_SHARED) \
		$(PREFILL_FP8_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-L$(BUILD_DIR) -lq27-gdn-prefill -lq27-gdn-prefill-ab \
		-lq27-gdn-prefill-wy -lq27-prefill-fp8 -lcublas -lcudart \
		$(ORIGIN_RPATH) -o $@

$(GDN_PREFILL_LAYER_SHARED): cuda/q27_gdn_prefill_layer.cu \
		include/q27_gdn_prefill_layer.h $(GDN_PREFILL_SUBLAYER_SHARED) \
		$(PREFILL_CORE_SHARED) $(PREFILL_FP8_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-L$(BUILD_DIR) -lq27-gdn-prefill-sublayer -lq27-prefill-core \
		-lq27-prefill-fp8 -lcublasLt -lcublas -lcudart \
		$(ORIGIN_RPATH) -o $@

$(GDN_PREFILL_M512_SHARED): cuda/q27_gdn_prefill_m512.cu \
		include/q27_gdn_prefill_m512.h $(GDN_PREFILL_SHARED) \
		$(GDN_PREFILL_AB_SHARED) $(GDN_PREFILL_WY_SHARED) \
		$(PREFILL_CORE_SHARED) $(PREFILL_FP8_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-L$(BUILD_DIR) -lq27-gdn-prefill -lq27-gdn-prefill-ab \
		-lq27-gdn-prefill-wy -lq27-prefill-core -lq27-prefill-fp8 \
		-lcublasLt -lcublas -lcudart $(ORIGIN_RPATH) -o $@

$(PREFILL_MODEL_SHARED): cuda/q27_prefill_model.cu \
		include/q27_prefill_model.h $(GDN_PREFILL_LAYER_SHARED) \
		$(GDN_PREFILL_M512_SHARED) \
		$(PREFILL_ATTENTION_LAYER_SHARED) $(PREFILL_MLP_SHARED) \
		$(PREFILL_CORE_SHARED) $(PREFILL_FP8_SHARED) \
		$(LM_HEAD_BF16_SHARED) $(KERNEL_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SHARED_FLAGS) \
		-Xlinker=-soname -Xlinker=$(notdir $@) $< \
		-L$(BUILD_DIR) -lq27-gdn-prefill-layer \
		-lq27-gdn-prefill-m512 \
		-lq27-prefill-attention-layer -lq27-prefill-mlp \
		-lq27-prefill-core -lq27-prefill-fp8 -lq27-lm-head-bf16 \
		-lq27-kernels -lcublas -lcudart $(ORIGIN_RPATH) -o $@

$(LM_HEAD_BENCH): fixtures/q27_lm_head_bench.cu \
		include/q27_lm_head_bf16.h include/q27_mapping.h include/q27_kernels.h \
		$(LM_HEAD_BF16_SHARED) $(KERNEL_SHARED) $(MAPPING_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) fixtures/q27_lm_head_bench.cu \
		-L$(BUILD_DIR) -lq27-lm-head-bf16 -lq27-kernels -lq27-mapping \
		-lcublas $(ORIGIN_RPATH) -o $@

$(FP8_FIXTURE_TEST): fixtures/q27_fp8_fixture_test.cu \
		include/q27_kernels.h $(KERNEL_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) fixtures/q27_fp8_fixture_test.cu \
		-L$(BUILD_DIR) -lq27-kernels $(ORIGIN_RPATH) -o $@

$(CORE_FIXTURE_TEST): fixtures/q27_core_fixture_test.cu \
		include/q27_kernels.h $(KERNEL_SHARED) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) fixtures/q27_core_fixture_test.cu \
		-L$(BUILD_DIR) -lq27-kernels $(ORIGIN_RPATH) -o $@

clean:
	rm -rf $(BUILD_DIR)

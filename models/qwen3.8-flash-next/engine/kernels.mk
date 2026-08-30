CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Werror
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror
AR ?= ar

MODEL_DIR := models/qwen3.8-flash-next
ENGINE_DIR := $(MODEL_DIR)/engine
KERNEL_DIR := $(ENGINE_DIR)/kernels
TOOL_DIR := $(ENGINE_DIR)/tools
BUILD_DIR := build/flash-next
CUDA_QWEN_GDN_SHARED := $(BUILD_DIR)/libflash-qwen-gdn.so
CUDA_QWEN_MOE_SHARED := $(BUILD_DIR)/libflash-qwen-moe.so
CUDA_MOE_GATE_SHARED := $(BUILD_DIR)/libflash-moe-gate.so
CUDA_SHARED_EXPERT_SHARED := $(BUILD_DIR)/libflash-shared-expert.so
CUDA_MOE_JOIN_SHARED := $(BUILD_DIR)/libflash-moe-join.so
CUDA_MHC_SHARED := $(BUILD_DIR)/libflash-mhc.so
CUDA_QSA_DECODE_XQA_MHA_OBJECT := $(BUILD_DIR)/qsa-decode-xqa-mha.o
CUDA_QSA_SCORE_OBJECT := $(BUILD_DIR)/qsa-score-tilelang.o
CUDA_QSA_SHARED := $(BUILD_DIR)/libflash-qsa.so
CUDA_QWEN_QSA_BLOCK_SHARED := $(BUILD_DIR)/libflash-qwen-qsa-block.so
CUDA_QWEN_PLE_BLOCK_SHARED := $(BUILD_DIR)/libflash-qwen-ple-block.so
CUDA_QWEN_DECODE_GLUE_SHARED := $(BUILD_DIR)/libflash-qwen-decode-glue.so
CUDA_QWEN_RUNTIME_SHARED := $(BUILD_DIR)/libflash-qwen-runtime.so
CUDA_QWEN_FUSED_MOE_SHARED := $(BUILD_DIR)/libflash-qwen-fused-moe.so
CUDA_QWEN_RUNTIME_FUSED_SHARED := $(BUILD_DIR)/libflash-qwen-runtime-fused.so
CUDA_FABRIC_SHARED := $(BUILD_DIR)/libflash-fabric.so
CUDA_QWEN_EXPERT_PACK_SHARED := $(BUILD_DIR)/libflash-qwen-expert-pack.so
NVCC ?= nvcc
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -O2 -std=c++20 -arch=$(CUDA_ARCH)
FLASHINFER_ROOT ?= $(HOME)/.cache/spark-c/sources/flashinfer
FLASHINFER_INCLUDE ?= $(FLASHINFER_ROOT)/include
CUTLASS_ROOT ?= $(FLASHINFER_ROOT)/3rdparty/cutlass
# CUDA 13.0 exposes exact sm_121 code generation but does not set the helper
# macro checked by this FlashInfer revision's architecture guard.
FLASHINFER_ARCH_FLAGS ?= -D__CUDA_ARCH_SPECIFIC__ --expt-relaxed-constexpr \
	-diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20015 \
	-diag-suppress 2908
FLASHINFER_INCLUDES := -I$(FLASHINFER_INCLUDE) \
	-I$(FLASHINFER_ROOT)/csrc/nv_internal \
	-I$(FLASHINFER_ROOT)/csrc/nv_internal/include \
	-I$(CUTLASS_ROOT)/include \
	-I$(CUTLASS_ROOT)/tools/util/include
FLASHINFER_XQA_INCLUDE := -I$(FLASHINFER_ROOT)/csrc/xqa
# FlashInfer generates the SM120 family fused-MoE instantiations at JIT-build
# time; they are intentionally not checked into the upstream repository.  Set
# this to the complete `cutlass_instantiations/120` output directory produced
# by `gen_cutlass_fused_moe_sm120_module`.  SM121 uses that same backend and
# generated directory (FlashInfer maps backend 121 to its SM120 module).
FLASHINFER_FUSED_MOE_GENERATED_DIR ?=
FLASHINFER_FUSED_MOE_GENERATED_SOURCES = $(shell \
	if test -n "$(FLASHINFER_FUSED_MOE_GENERATED_DIR)" && \
		test -d "$(FLASHINFER_FUSED_MOE_GENERATED_DIR)"; then \
		find "$(FLASHINFER_FUSED_MOE_GENERATED_DIR)" -type f \
			-name '*.generated.cu' -print | LC_ALL=C sort; \
	fi)
# Pinned FlashInfer 906181e groups its 88 SM120 FP4/FP8xFP4 operations eight
# per translation unit, split by CTA-M and mixed-input mode: 11 files total.
FLASHINFER_FUSED_MOE_GENERATED_SOURCE_COUNT := 11
FLASHINFER_FUSED_MOE_INCLUDES := $(FLASHINFER_INCLUDES) \
	-I$(FLASHINFER_ROOT)/csrc/fused_moe/cutlass_backend \
	-I$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include \
	-I$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include \
	-I$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels \
	-I$(FLASHINFER_FUSED_MOE_GENERATED_DIR)
# Keep sm_121a code generation on GB10.  FlashInfer deliberately compiles the
# SM120-family generated registry into a distinct SM121 cubin because executing
# an sm_120 cubin on SM121 can raise cudaErrorIllegalInstruction.
FLASHINFER_FUSED_MOE_FLAGS := $(FLASHINFER_ARCH_FLAGS) \
	-DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 \
	-DCOMPILE_BLACKWELL_TMA_GEMMS \
	-DCOMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS \
	-DENABLE_BF16 -DENABLE_FP8 -DENABLE_FP4 \
	-DUSING_OSS_CUTLASS_MOE_GEMM \
	-DCUTLASS_ENABLE_GDC_FOR_SM100=1
# Match FlashInfer's pinned JIT build mode exactly for the generated SM121 MoE
# module.  In particular, its architecture-specific FP4 conversions require
# compute_121a (a plain sm_121 target rejects the E2M1 conversion opcodes), and
# CUTLASS 4.5 selects an ill-formed epilogue aggregate under C++20.
FLASHINFER_FUSED_MOE_GENCODE ?= -gencode=arch=compute_121a,code=sm_121a
FLASHINFER_FUSED_MOE_NVCCFLAGS := $(filter-out -std=% -arch=%,$(NVCCFLAGS)) \
	-std=c++17 $(FLASHINFER_FUSED_MOE_GENCODE)
FLASHINFER_FUSED_MOE_SOURCES := \
	$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu \
	$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu \
	$(FLASHINFER_ROOT)/csrc/nv_internal/cpp/common/envUtils.cpp \
	$(FLASHINFER_ROOT)/csrc/nv_internal/cpp/common/logger.cpp \
	$(FLASHINFER_ROOT)/csrc/nv_internal/cpp/common/stringUtils.cpp \
	$(FLASHINFER_ROOT)/csrc/nv_internal/cpp/common/tllmException.cpp \
	$(FLASHINFER_ROOT)/csrc/nv_internal/cpp/common/memoryUtils.cu \
	$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu \
	$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp \
	$(FLASHINFER_ROOT)/csrc/nv_internal/tensorrt_llm/kernels/lora/lora.cpp
FLASHINFER_FUSED_MOE_OBJECT_DIR := $(BUILD_DIR)/flashinfer-fused-moe-objects
FLASHINFER_FUSED_MOE_ALL_SOURCES := $(FLASHINFER_FUSED_MOE_SOURCES) \
	$(FLASHINFER_FUSED_MOE_GENERATED_SOURCES) \
	$(KERNEL_DIR)/cuda/qwen_fused_moe_flashinfer.cu
FLASHINFER_FUSED_MOE_OBJECTS := $(foreach source,$(FLASHINFER_FUSED_MOE_ALL_SOURCES),\
	$(FLASHINFER_FUSED_MOE_OBJECT_DIR)/$(notdir $(basename $(source))).o)

# Compile the generated CUTLASS registry one translation unit per make job.
# A single NVCC link command otherwise serializes all 11 expensive SM120 units
# internally and leaves `make -j` idle.
define FLASHINFER_FUSED_MOE_COMPILE_RULE
$(FLASHINFER_FUSED_MOE_OBJECT_DIR)/$(notdir $(basename $(1))).o: $(1) $(KERNEL_DIR)/include/flash/qwen_fused_moe_flashinfer_api.h
	mkdir -p $$(FLASHINFER_FUSED_MOE_OBJECT_DIR)
	$$(NVCC) $$(FLASHINFER_FUSED_MOE_NVCCFLAGS) $$(FLASHINFER_FUSED_MOE_FLAGS) \
		-Xcompiler=-fPIC -I$$(KERNEL_DIR)/include \
		$$(FLASHINFER_FUSED_MOE_INCLUDES) -c $$< -o $$@
endef
$(foreach source,$(FLASHINFER_FUSED_MOE_ALL_SOURCES),\
	$(eval $(call FLASHINFER_FUSED_MOE_COMPILE_RULE,$(source))))
TILELANG_QSA_SCORE_ROOT := $(KERNEL_DIR)/generated/qsa-score
TILELANG_QSA_SCORE_INCLUDE := -I$(TILELANG_QSA_SCORE_ROOT)/include
QWEN_XQA_FLAGS := --expt-relaxed-constexpr \
	-DBEAM_WIDTH=1 -DUSE_INPUT_KV=0 -DUSE_CUSTOM_BARRIER=1 \
	-DTOKENS_PER_PAGE=64 -DHEAD_ELEMS=256 -DINPUT_FP16=0 \
	-DDTYPE=__nv_bfloat16 -DCACHE_ELEM_ENUM=0 -DHEAD_GRP_SIZE=12 \
	-DSLIDING_WINDOW=0 -DLOW_PREC_OUTPUT=0 -DSPEC_DEC=0 \
	-DMLA_WRAPPER=0 -DUSE_SM90_MHA=0
CUTE_NVFP4_OBJECT ?=
CUTE_NVFP4_QUANTIZE_OBJECT ?=
GDN_AOT_OBJECT ?=
GDN_PREFILL_AOT_OBJECTS ?=
TVM_FFI_ROOT ?=
CUTE_DSL_ROOT ?=

.PHONY: clean fabric-shared qwen-expert-pack-shared qwen-moe-shared qwen-gdn-shared \
	qwen-qsa-block-shared qwen-ple-block-shared qwen-decode-glue-shared \
	qwen-runtime-shared qwen-fused-moe-flashinfer-shared \
	qwen-runtime-fused-moe-shared check-flashinfer-fused-moe-generated \
	qsa-shared moe-gate-shared shared-expert-shared moe-join-shared mhc-shared

fabric-shared: $(CUDA_FABRIC_SHARED)

qwen-expert-pack-shared: $(CUDA_QWEN_EXPERT_PACK_SHARED)

qwen-moe-shared: $(CUDA_QWEN_MOE_SHARED)

qwen-gdn-shared: $(CUDA_QWEN_GDN_SHARED)

qwen-qsa-block-shared: $(CUDA_QWEN_QSA_BLOCK_SHARED)

qwen-ple-block-shared: $(CUDA_QWEN_PLE_BLOCK_SHARED)

qwen-decode-glue-shared: $(CUDA_QWEN_DECODE_GLUE_SHARED)

qwen-runtime-shared: $(CUDA_QWEN_RUNTIME_SHARED)

qwen-fused-moe-flashinfer-shared: check-flashinfer-fused-moe-generated $(CUDA_QWEN_FUSED_MOE_SHARED)

qwen-runtime-fused-moe-shared: $(CUDA_QWEN_RUNTIME_FUSED_SHARED)

# Do not silently substitute the ordinary grouped-GEMM files for the generated
# full fused-MoE registry.  The indexed v1 target remains available without
# this check; only the explicit full-bank target requires generated sources.
check-flashinfer-fused-moe-generated:
	@test -n "$(FLASHINFER_FUSED_MOE_GENERATED_DIR)" || { \
		echo "FLASHINFER_FUSED_MOE_GENERATED_DIR is required; point it at FlashInfer's complete cutlass_instantiations/120 directory" >&2; \
		exit 2; \
	}
	@test -d "$(FLASHINFER_FUSED_MOE_GENERATED_DIR)" || { \
		echo "FlashInfer fused-MoE generated directory does not exist: $(FLASHINFER_FUSED_MOE_GENERATED_DIR)" >&2; \
		exit 2; \
	}
	@test -n "$(strip $(FLASHINFER_FUSED_MOE_GENERATED_SOURCES))" || { \
		echo "No *.generated.cu instantiations found under $(FLASHINFER_FUSED_MOE_GENERATED_DIR); run FlashInfer gen_cutlass_fused_moe_sm120_module first" >&2; \
		exit 2; \
	}
	@set -- $(FLASHINFER_FUSED_MOE_GENERATED_SOURCES); \
		test "$$#" -eq "$(FLASHINFER_FUSED_MOE_GENERATED_SOURCE_COUNT)" || { \
			echo "Incomplete or mismatched FlashInfer SM120 fused-MoE registry: found $$# generated sources, expected $(FLASHINFER_FUSED_MOE_GENERATED_SOURCE_COUNT) for pinned 906181e" >&2; \
			exit 2; \
		}

qsa-shared: $(CUDA_QSA_SHARED)

moe-gate-shared: $(CUDA_MOE_GATE_SHARED)

shared-expert-shared: $(CUDA_SHARED_EXPERT_SHARED)

moe-join-shared: $(CUDA_MOE_JOIN_SHARED)

mhc-shared: $(CUDA_MHC_SHARED)

$(CUDA_QWEN_GDN_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/cuda/gdn_block_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/cuda/qwen_gdn_aux.cu $(KERNEL_DIR)/internal/gdn_decode_backend.h $(KERNEL_DIR)/internal/gdn_decode_flashinfer_backend.h $(KERNEL_DIR)/internal/gdn_block_backend.h $(KERNEL_DIR)/internal/mhc_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h $(KERNEL_DIR)/include/flash/qwen_gdn_aux_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(GDN_PREFILL_AOT_OBJECTS)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	for object in $(GDN_PREFILL_AOT_OBJECTS); do test -f "$$object"; done
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_CUDA -DFLASH_WITH_FLASHINFER_GDN_AOT \
		-DFLASH_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-DFLASH_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		-I$(TVM_FFI_ROOT)/include $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc \
		$(KERNEL_DIR)/cuda/gdn_block_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu \
		$(KERNEL_DIR)/cuda/qwen_gdn_aux.cu "$(GDN_AOT_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_GDN_SHARED)

$(CUDA_FABRIC_SHARED): $(KERNEL_DIR)/fabric/coherent_region.cc $(KERNEL_DIR)/fabric/cuda_runtime.cc $(KERNEL_DIR)/include/flash/fabric_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -I$(KERNEL_DIR)/include \
		$(KERNEL_DIR)/fabric/coherent_region.cc $(KERNEL_DIR)/fabric/cuda_runtime.cc \
		-lcublas -o $(CUDA_FABRIC_SHARED)

$(CUDA_QWEN_EXPERT_PACK_SHARED): $(KERNEL_DIR)/cuda/qwen_expert_pack.cu $(KERNEL_DIR)/include/flash/qwen_expert_pack_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -I$(KERNEL_DIR)/include \
		$(KERNEL_DIR)/cuda/qwen_expert_pack.cu -o $(CUDA_QWEN_EXPERT_PACK_SHARED)

$(CUDA_QWEN_MOE_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_grouped_indexed_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu $(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu $(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/internal/nvfp4_grouped_backend.h $(KERNEL_DIR)/internal/nvfp4_silu_backend.h $(KERNEL_DIR)/internal/nvfp4_quantize_backend.h $(KERNEL_DIR)/internal/moe_route_backend.h $(KERNEL_DIR)/internal/moe_gate_backend.h $(KERNEL_DIR)/internal/shared_expert_backend.h $(KERNEL_DIR)/internal/moe_join_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	test -f "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(FLASHINFER_ARCH_FLAGS) \
		-diag-suppress 177 -diag-suppress 549 -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DFLASH_WITH_FLASHINFER_INDEXED_GROUPED_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DFLASH_WITH_FLASHINFER_MOE_ROUTE \
		-DFLASH_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DFLASH_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DFLASH_WITH_SGLANG_FUSED_MOE_JOIN \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(FLASHINFER_INCLUDES) \
		-I$(TVM_FFI_ROOT)/include $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu \
		$(KERNEL_DIR)/cuda/nvfp4_grouped_indexed_flashinfer.cu \
		$(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu $(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu \
		$(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/cuda/moe_gate_sglang.cu \
		$(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/cuda/moe_join_sglang.cu \
		"$(CUTE_NVFP4_OBJECT)" "$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_MOE_SHARED)

$(CUDA_MOE_GATE_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/internal/moe_gate_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_CUBLAS_MOE_GATE -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_gate_sglang.cu -lcublas \
		-o $(CUDA_MOE_GATE_SHARED)

$(CUDA_SHARED_EXPERT_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/internal/shared_expert_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_CUBLAS_SHARED_EXPERT -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/shared_expert_sglang.cu -lcublas \
		-o $(CUDA_SHARED_EXPERT_SHARED)

$(CUDA_MOE_JOIN_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/internal/moe_join_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_FUSED_MOE_JOIN -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_join_sglang.cu \
		-o $(CUDA_MOE_JOIN_SHARED)

$(CUDA_MHC_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/internal/mhc_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/mhc_sglang.cu -lcublas \
		-o $(CUDA_MHC_SHARED)

$(CUDA_QSA_SCORE_OBJECT): $(KERNEL_DIR)/cuda/qsa_score_tilelang.cu $(KERNEL_DIR)/internal/qsa_score_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h $(TILELANG_QSA_SCORE_ROOT)/generated/device_kernel.cu $(wildcard $(TILELANG_QSA_SCORE_ROOT)/include/tl_templates/cuda/*.h) $(wildcard $(TILELANG_QSA_SCORE_ROOT)/include/tl_templates/cuda/instruction/*.h)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -DENABLE_BF16 -Xcompiler=-fPIC \
		-I. -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(TILELANG_QSA_SCORE_INCLUDE) \
		-I$(CUTLASS_ROOT)/include -c $(KERNEL_DIR)/cuda/qsa_score_tilelang.cu \
		-o $(CUDA_QSA_SCORE_OBJECT)

$(CUDA_QSA_DECODE_XQA_MHA_OBJECT): $(FLASHINFER_ROOT)/csrc/xqa/mha.cu $(wildcard $(FLASHINFER_ROOT)/csrc/xqa/*.h) $(wildcard $(FLASHINFER_ROOT)/csrc/xqa/*.cuh)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) -DNDEBUG=1 -Xcompiler=-fPIC \
		$(FLASHINFER_XQA_INCLUDE) -c $(FLASHINFER_ROOT)/csrc/xqa/mha.cu \
		-o $(CUDA_QSA_DECODE_XQA_MHA_OBJECT)

$(CUDA_QSA_SHARED): $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_index_prep_sglang.cu $(KERNEL_DIR)/cuda/qsa_topk_sglang.cu $(KERNEL_DIR)/cuda/qsa_expand_sglang.cu $(KERNEL_DIR)/cuda/qsa_kv_pack_sglang.cu $(KERNEL_DIR)/cuda/qsa_decode_xqa_flashinfer.cu $(KERNEL_DIR)/internal/qsa_index_prep_backend.h $(KERNEL_DIR)/internal/qsa_topk_backend.h $(KERNEL_DIR)/internal/qsa_expand_backend.h $(KERNEL_DIR)/internal/qsa_score_backend.h $(KERNEL_DIR)/internal/qsa_kv_pack_backend.h $(KERNEL_DIR)/internal/qsa_decode_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_QSA_INDEX_PREP \
		-DFLASH_WITH_SGLANG_QSA_TOPK \
		-DFLASH_WITH_SGLANG_QSA_EXPAND \
		-DFLASH_WITH_TILELANG_QSA_SCORE \
		-DFLASH_WITH_SGLANG_QSA_KV_PACK \
		-DFLASH_WITH_FLASHINFER_XQA_DECODE -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(FLASHINFER_XQA_INCLUDE) $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/qsa_index_prep_sglang.cu \
		$(KERNEL_DIR)/cuda/qsa_topk_sglang.cu \
		$(KERNEL_DIR)/cuda/qsa_expand_sglang.cu \
		$(KERNEL_DIR)/cuda/qsa_kv_pack_sglang.cu \
		$(KERNEL_DIR)/cuda/qsa_decode_xqa_flashinfer.cu \
		$(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) -lcuda -o $(CUDA_QSA_SHARED)

$(CUDA_QWEN_QSA_BLOCK_SHARED): $(KERNEL_DIR)/cuda/qwen_qsa_block.cu $(KERNEL_DIR)/include/flash/qwen_qsa_block_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -I$(KERNEL_DIR)/include \
		$(KERNEL_DIR)/cuda/qwen_qsa_block.cu -lcublas -lcudart \
		-o $(CUDA_QWEN_QSA_BLOCK_SHARED)

$(CUDA_QWEN_PLE_BLOCK_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/ple_gather.cu $(KERNEL_DIR)/cuda/qwen_ple_block.cu $(KERNEL_DIR)/internal/ple_gather_backend.h $(KERNEL_DIR)/include/flash/qwen_ple_block_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_PLE_GATHER -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/ple_gather.cu \
		$(KERNEL_DIR)/cuda/qwen_ple_block.cu -lcublas -lcudart \
		-o $(CUDA_QWEN_PLE_BLOCK_SHARED)

$(CUDA_QWEN_DECODE_GLUE_SHARED): $(KERNEL_DIR)/cuda/qwen_decode_glue.cu $(KERNEL_DIR)/include/flash/qwen_decode_glue_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -I$(KERNEL_DIR)/include \
		$(KERNEL_DIR)/cuda/qwen_decode_glue.cu -lcublas -lcudart \
		-o $(CUDA_QWEN_DECODE_GLUE_SHARED)

# Full-bank NVFP4 fused MoE.  This deliberately follows FlashInfer's own
# `gen_cutlass_fused_moe_sm120_module` source and flag set instead of treating
# the checked-in generic grouped-GEMM files as a replacement for its generated
# tactic registry.  Backend 121 is implemented by the same SM120 module; the
# opt-in target defaults to FlashInfer's exact sm_121a JIT gencode for GB10.
$(CUDA_QWEN_FUSED_MOE_SHARED): $(FLASHINFER_FUSED_MOE_OBJECTS) | check-flashinfer-fused-moe-generated
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(FLASHINFER_FUSED_MOE_NVCCFLAGS) -shared \
		$(FLASHINFER_FUSED_MOE_OBJECTS) \
		-lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_FUSED_MOE_SHARED)

# Small link overlay for the Rust engine's model-local ABI.  The ordinary
# runtime remains the implementation donor for every existing kernel, while
# recompiling qwen_runtime_direct.cc with this one feature macro activates the
# fused-MoE forwarding shims.  Link the Rust binary to this artifact instead of
# libflash-qwen-runtime.so when opting in.  $ORIGIN keeps both donor libraries
# resolvable after the build directory is copied as a unit.
$(CUDA_QWEN_RUNTIME_FUSED_SHARED): $(CUDA_QWEN_RUNTIME_SHARED) $(CUDA_QWEN_FUSED_MOE_SHARED) $(KERNEL_DIR)/qwen_runtime_direct.cc $(KERNEL_DIR)/include/flash/qwen_runtime_api.h $(KERNEL_DIR)/include/flash/qwen_fused_moe_flashinfer_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DFLASH_QWEN_RUNTIME_WITH_FUSED_MOE \
		-I$(KERNEL_DIR)/include $(KERNEL_DIR)/qwen_runtime_direct.cc \
		-L$(BUILD_DIR) \
		-lflash-qwen-runtime -lflash-qwen-fused-moe \
		-Xlinker -rpath -Xlinker '$$ORIGIN' \
		-Xlinker -soname -Xlinker libflash-qwen-runtime-fused.so \
		-lcuda -lcudart -o $(CUDA_QWEN_RUNTIME_FUSED_SHARED)

$(CUDA_QWEN_RUNTIME_SHARED): $(GDN_AOT_OBJECT) $(GDN_PREFILL_AOT_OBJECTS) $(CUTE_NVFP4_OBJECT) $(CUTE_NVFP4_QUANTIZE_OBJECT) $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/qwen_runtime_direct.cc $(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/cuda/gdn_block_sglang.cu $(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_grouped_indexed_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu $(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu $(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/cuda/ple_gather.cu $(KERNEL_DIR)/cuda/qsa_index_prep_sglang.cu $(KERNEL_DIR)/cuda/qsa_topk_sglang.cu $(KERNEL_DIR)/cuda/qsa_expand_sglang.cu $(KERNEL_DIR)/cuda/qsa_kv_pack_sglang.cu $(KERNEL_DIR)/cuda/qsa_decode_xqa_flashinfer.cu $(KERNEL_DIR)/cuda/qwen_expert_pack.cu $(KERNEL_DIR)/cuda/qwen_gdn_aux.cu $(KERNEL_DIR)/cuda/qwen_qsa_block.cu $(KERNEL_DIR)/cuda/qwen_ple_block.cu $(KERNEL_DIR)/cuda/qwen_decode_glue.cu $(KERNEL_DIR)/include/flash/kernel_api.h $(KERNEL_DIR)/include/flash/qwen_runtime_api.h $(KERNEL_DIR)/include/flash/qwen_expert_pack_api.h $(KERNEL_DIR)/include/flash/qwen_decode_glue_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(GDN_PREFILL_AOT_OBJECTS)"
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	for object in $(GDN_PREFILL_AOT_OBJECTS); do test -f "$$object"; done
	test -f "$(CUTE_NVFP4_OBJECT)"
	test -f "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) --use_fast_math $(FLASHINFER_ARCH_FLAGS) -diag-suppress 177 \
		-diag-suppress 549 -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_CUDA \
		-DFLASH_WITH_FLASHINFER_GDN_AOT \
		-DFLASH_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-DFLASH_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DFLASH_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DFLASH_WITH_FLASHINFER_INDEXED_GROUPED_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DFLASH_WITH_FLASHINFER_MOE_ROUTE \
		-DFLASH_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DFLASH_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DFLASH_WITH_SGLANG_FUSED_MOE_JOIN \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC \
		-DFLASH_WITH_SGLANG_PLE_GATHER \
		-DFLASH_WITH_SGLANG_QSA_INDEX_PREP \
		-DFLASH_WITH_SGLANG_QSA_TOPK \
		-DFLASH_WITH_SGLANG_QSA_EXPAND \
		-DFLASH_WITH_TILELANG_QSA_SCORE \
		-DFLASH_WITH_SGLANG_QSA_KV_PACK \
		-DFLASH_WITH_FLASHINFER_XQA_DECODE \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(FLASHINFER_INCLUDES) $(FLASHINFER_XQA_INCLUDE) -I$(TVM_FFI_ROOT)/include \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/qwen_runtime_direct.cc \
		$(KERNEL_DIR)/cuda/gdn_decode.cu \
		$(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/cuda/gdn_block_sglang.cu \
		$(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu \
		$(KERNEL_DIR)/cuda/nvfp4_grouped_indexed_flashinfer.cu \
		$(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu $(KERNEL_DIR)/cuda/moe_route_flashinfer.cu \
		$(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/cuda/shared_expert_sglang.cu \
		$(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu \
		$(KERNEL_DIR)/cuda/ple_gather.cu $(KERNEL_DIR)/cuda/qsa_index_prep_sglang.cu \
		$(KERNEL_DIR)/cuda/qsa_topk_sglang.cu $(KERNEL_DIR)/cuda/qsa_expand_sglang.cu \
		$(KERNEL_DIR)/cuda/qsa_kv_pack_sglang.cu $(KERNEL_DIR)/cuda/qsa_decode_xqa_flashinfer.cu \
		$(KERNEL_DIR)/cuda/qwen_expert_pack.cu \
		$(KERNEL_DIR)/cuda/qwen_gdn_aux.cu $(KERNEL_DIR)/cuda/qwen_qsa_block.cu \
		$(KERNEL_DIR)/cuda/qwen_ple_block.cu $(KERNEL_DIR)/cuda/qwen_decode_glue.cu \
		"$(GDN_AOT_OBJECT)" "$(CUTE_NVFP4_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) \
		"$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_RUNTIME_SHARED)

clean:
	rm -rf -- $(BUILD_DIR)

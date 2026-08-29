CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Werror
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror
AR ?= ar

MODEL_DIR := models/qwen3.8-flash-next
NATIVE_DIR := $(MODEL_DIR)/native
KERNEL_DIR := $(NATIVE_DIR)/kernels
TOOL_DIR := $(NATIVE_DIR)/tools
BUILD_DIR := build/flash-next
CONTRACT_TEST := $(BUILD_DIR)/kernel-contract-test
HEADER_C_TEST := $(BUILD_DIR)/kernel-header-c-test
CUDA_GDN_TEST := $(BUILD_DIR)/gdn-decode-cuda-test
CUDA_GDN_FIXTURE_TEST := $(BUILD_DIR)/gdn-fixture-test
CUDA_GDN_PREFILL_FIXTURE_TEST := $(BUILD_DIR)/gdn-prefill-fixture-test
CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST := $(BUILD_DIR)/qwen-gdn-block-fixture-test
CUDA_QWEN_GDN_SHARED := $(BUILD_DIR)/libflash-qwen-gdn.so
CUDA_NVFP4_TEST := $(BUILD_DIR)/nvfp4-dense-cuda-test
CUDA_NVFP4_FIXTURE_TEST := $(BUILD_DIR)/nvfp4-dense-fixture-test
CUDA_GROUPED_NVFP4_TEST := $(BUILD_DIR)/nvfp4-grouped-cuda-test
CUDA_GROUPED_NVFP4_FIXTURE_TEST := $(BUILD_DIR)/nvfp4-grouped-fixture-test
CUDA_SILU_NVFP4_TEST := $(BUILD_DIR)/nvfp4-silu-cute-test
CUDA_QWEN_MOE_FIXTURE_TEST := $(BUILD_DIR)/qwen-moe-fixture-test
CUDA_QWEN_MOE_SHARED := $(BUILD_DIR)/libflash-qwen-moe.so
CUDA_QWEN_FULL_LAYER_FIXTURE_TEST := $(BUILD_DIR)/qwen-full-layer-fixture-test
CUDA_MOE_ROUTE_TEST := $(BUILD_DIR)/moe-route-cuda-test
CUDA_MOE_GATE_FIXTURE_TEST := $(BUILD_DIR)/moe-gate-fixture-test
CUDA_MOE_GATE_SHARED := $(BUILD_DIR)/libflash-moe-gate.so
CUDA_SHARED_EXPERT_FIXTURE_TEST := $(BUILD_DIR)/shared-expert-fixture-test
CUDA_SHARED_EXPERT_SHARED := $(BUILD_DIR)/libflash-shared-expert.so
CUDA_MOE_JOIN_FIXTURE_TEST := $(BUILD_DIR)/moe-join-fixture-test
CUDA_MOE_JOIN_SHARED := $(BUILD_DIR)/libflash-moe-join.so
CUDA_MHC_FIXTURE_TEST := $(BUILD_DIR)/mhc-fixture-test
CUDA_MHC_SHARED := $(BUILD_DIR)/libflash-mhc.so
CUDA_COHERENT_REGION_TEST := $(BUILD_DIR)/coherent-region-cuda-test
CUDA_PLE_GATHER_FIXTURE_TEST := $(BUILD_DIR)/ple-gather-fixture-test
CUDA_QSA_TOPK_FIXTURE_TEST := $(BUILD_DIR)/qsa-topk-fixture-test
CUDA_QSA_EXPAND_FIXTURE_TEST := $(BUILD_DIR)/qsa-expand-fixture-test
CUDA_QSA_SCORE_FIXTURE_TEST := $(BUILD_DIR)/qsa-score-fixture-test
CUDA_QSA_INDEX_PREP_FIXTURE_TEST := $(BUILD_DIR)/qsa-index-prep-fixture-test
CUDA_QSA_KV_PACK_FIXTURE_TEST := $(BUILD_DIR)/qsa-kv-pack-fixture-test
CUDA_QSA_DECODE_XQA_FIXTURE_TEST := $(BUILD_DIR)/qsa-decode-xqa-fixture-test
CUDA_QSA_DECODE_XQA_MHA_OBJECT := $(BUILD_DIR)/qsa-decode-xqa-mha.o
CUDA_QSA_SCORE_OBJECT := $(BUILD_DIR)/qsa-score-tilelang.o
CUDA_QSA_SHARED := $(BUILD_DIR)/libflash-qsa.so
CUDA_QWEN_QSA_BLOCK_SHARED := $(BUILD_DIR)/libflash-qwen-qsa-block.so
CUDA_QWEN_PLE_BLOCK_SHARED := $(BUILD_DIR)/libflash-qwen-ple-block.so
CUDA_QWEN_DECODE_GLUE_SHARED := $(BUILD_DIR)/libflash-qwen-decode-glue.so
CUDA_QWEN_RUNTIME_SHARED := $(BUILD_DIR)/libflash-qwen-runtime.so
CUDA_FABRIC_SHARED := $(BUILD_DIR)/libflash-fabric.so
CUDA_QWEN_EXPERT_PACK_SHARED := $(BUILD_DIR)/libflash-qwen-expert-pack.so
CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST := $(BUILD_DIR)/qwen-expert-pack-fixture-test
NVCC ?= nvcc
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -O2 -std=c++20 -arch=$(CUDA_ARCH)
FLASHINFER_ROOT ?= vendor/_deps/flashinfer
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
TILELANG_QSA_SCORE_INCLUDE := -Ivendor/tilelang-qsa-score/include
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

.PHONY: test test-cpp test-cuda clean \
	fabric-shared qwen-expert-pack-shared qwen-moe-shared qwen-gdn-shared \
	qwen-qsa-block-shared qwen-ple-block-shared qwen-decode-glue-shared \
	qwen-runtime-shared qsa-shared moe-gate-shared shared-expert-shared \
	moe-join-shared mhc-shared test-cuda-fabric \
	test-cuda-qwen-expert-pack-fixture test-cuda-gdn test-cuda-gdn-fixture \
	test-cuda-gdn-prefill-fixture test-cuda-qwen-gdn-block-fixture \
	test-cuda-nvfp4 test-cuda-nvfp4-fixture test-cuda-grouped-nvfp4 \
	test-cuda-grouped-nvfp4-fixture test-cuda-silu-nvfp4 \
	test-cuda-silu-nvfp4-fixture test-cuda-moe-route \
	test-cuda-moe-gate-fixture test-cuda-shared-expert-fixture \
	test-cuda-moe-join-fixture test-cuda-mhc-fixture \
	test-cuda-ple-gather-fixture test-cuda-qsa-topk-fixture \
	test-cuda-qsa-expand-fixture test-cuda-qsa-score-fixture \
	test-cuda-qsa-index-prep-fixture test-cuda-qsa-kv-pack-fixture \
	test-cuda-qsa-decode-xqa-fixture test-cuda-qwen-moe-fixture \
	test-cuda-qwen-full-layer-fixture

test: test-cpp

test-cpp: $(CONTRACT_TEST) $(HEADER_C_TEST)
	$(CONTRACT_TEST)
	$(HEADER_C_TEST)

test-cuda: test-cuda-fabric test-cuda-gdn test-cuda-nvfp4 test-cuda-grouped-nvfp4 test-cuda-moe-route

test-cuda-fabric: $(CUDA_COHERENT_REGION_TEST)
	$(CUDA_COHERENT_REGION_TEST)

test-cuda-qwen-expert-pack-fixture: $(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST)
	$(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST)

fabric-shared: $(CUDA_FABRIC_SHARED)

qwen-expert-pack-shared: $(CUDA_QWEN_EXPERT_PACK_SHARED)

qwen-moe-shared: $(CUDA_QWEN_MOE_SHARED)

qwen-gdn-shared: $(CUDA_QWEN_GDN_SHARED)

qwen-qsa-block-shared: $(CUDA_QWEN_QSA_BLOCK_SHARED)

qwen-ple-block-shared: $(CUDA_QWEN_PLE_BLOCK_SHARED)

qwen-decode-glue-shared: $(CUDA_QWEN_DECODE_GLUE_SHARED)

qwen-runtime-shared: $(CUDA_QWEN_RUNTIME_SHARED)

qsa-shared: $(CUDA_QSA_SHARED)

moe-gate-shared: $(CUDA_MOE_GATE_SHARED)

shared-expert-shared: $(CUDA_SHARED_EXPERT_SHARED)

moe-join-shared: $(CUDA_MOE_JOIN_SHARED)

mhc-shared: $(CUDA_MHC_SHARED)

test-cuda-gdn: $(CUDA_GDN_TEST)
	$(CUDA_GDN_TEST)

test-cuda-gdn-fixture: $(CUDA_GDN_FIXTURE_TEST)
	test -n "$(QWEN_GDN_FIXTURE)"
	LD_LIBRARY_PATH=$(TVM_FFI_ROOT)/lib:$$LD_LIBRARY_PATH \
		$(CUDA_GDN_FIXTURE_TEST) "$(QWEN_GDN_FIXTURE)"

test-cuda-gdn-prefill-fixture: $(CUDA_GDN_PREFILL_FIXTURE_TEST)
	test -n "$(QWEN_GDN_PREFILL_FIXTURE)"
	LD_LIBRARY_PATH=$(TVM_FFI_ROOT)/lib:$$LD_LIBRARY_PATH \
		$(CUDA_GDN_PREFILL_FIXTURE_TEST) "$(QWEN_GDN_PREFILL_FIXTURE)"

test-cuda-qwen-gdn-block-fixture: $(CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST)
	test -n "$(QWEN_GDN_FIXTURE)"
	LD_LIBRARY_PATH=$(TVM_FFI_ROOT)/lib:$$LD_LIBRARY_PATH \
		$(CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST) "$(QWEN_GDN_FIXTURE)"

test-cuda-nvfp4: $(CUDA_NVFP4_TEST)
	$(CUDA_NVFP4_TEST)

test-cuda-nvfp4-fixture: $(CUDA_NVFP4_FIXTURE_TEST)
	test -n "$(NVFP4_FIXTURE)"
	$(CUDA_NVFP4_FIXTURE_TEST) "$(NVFP4_FIXTURE)"

test-cuda-grouped-nvfp4: $(CUDA_GROUPED_NVFP4_TEST)
	$(CUDA_GROUPED_NVFP4_TEST)

test-cuda-grouped-nvfp4-fixture: $(CUDA_GROUPED_NVFP4_FIXTURE_TEST)
	test -n "$(NVFP4_GROUPED_FIXTURE)"
	$(CUDA_GROUPED_NVFP4_FIXTURE_TEST) "$(NVFP4_GROUPED_FIXTURE)"

test-cuda-silu-nvfp4: $(CUDA_SILU_NVFP4_TEST)
	LD_LIBRARY_PATH=$(TVM_FFI_ROOT)/lib:$$LD_LIBRARY_PATH $(CUDA_SILU_NVFP4_TEST)

test-cuda-silu-nvfp4-fixture: $(CUDA_SILU_NVFP4_TEST)
	test -n "$(NVFP4_SILU_FIXTURE)"
	LD_LIBRARY_PATH=$(TVM_FFI_ROOT)/lib:$$LD_LIBRARY_PATH $(CUDA_SILU_NVFP4_TEST) "$(NVFP4_SILU_FIXTURE)"

test-cuda-moe-route: $(CUDA_MOE_ROUTE_TEST)
	$(CUDA_MOE_ROUTE_TEST)

test-cuda-moe-gate-fixture: $(CUDA_MOE_GATE_FIXTURE_TEST)
	test -n "$(QWEN_ROUTER_FIXTURE)"
	$(CUDA_MOE_GATE_FIXTURE_TEST) "$(QWEN_ROUTER_FIXTURE)"

test-cuda-shared-expert-fixture: $(CUDA_SHARED_EXPERT_FIXTURE_TEST)
	test -n "$(QWEN_SHARED_EXPERT_FIXTURE)"
	$(CUDA_SHARED_EXPERT_FIXTURE_TEST) "$(QWEN_SHARED_EXPERT_FIXTURE)"

test-cuda-moe-join-fixture: $(CUDA_MOE_JOIN_FIXTURE_TEST)
	test -n "$(QWEN_JOINED_MOE_FIXTURE)"
	$(CUDA_MOE_JOIN_FIXTURE_TEST) "$(QWEN_JOINED_MOE_FIXTURE)"

test-cuda-mhc-fixture: $(CUDA_MHC_FIXTURE_TEST)
	test -n "$(QWEN_MHC_FIXTURE)"
	$(CUDA_MHC_FIXTURE_TEST) "$(QWEN_MHC_FIXTURE)"

test-cuda-ple-gather-fixture: $(CUDA_PLE_GATHER_FIXTURE_TEST)
	test -n "$(PLE_GATHER_FIXTURE)"
	$(CUDA_PLE_GATHER_FIXTURE_TEST) "$(PLE_GATHER_FIXTURE)"

test-cuda-qsa-topk-fixture: $(CUDA_QSA_TOPK_FIXTURE_TEST)
	test -n "$(QSA_TOPK_FIXTURE)"
	$(CUDA_QSA_TOPK_FIXTURE_TEST) "$(QSA_TOPK_FIXTURE)"

test-cuda-qsa-expand-fixture: $(CUDA_QSA_EXPAND_FIXTURE_TEST)
	test -n "$(QSA_EXPAND_FIXTURE)"
	$(CUDA_QSA_EXPAND_FIXTURE_TEST) "$(QSA_EXPAND_FIXTURE)"

test-cuda-qsa-score-fixture: $(CUDA_QSA_SCORE_FIXTURE_TEST)
	test -n "$(QSA_SCORE_FIXTURE)"
	$(CUDA_QSA_SCORE_FIXTURE_TEST) "$(QSA_SCORE_FIXTURE)"

test-cuda-qsa-index-prep-fixture: $(CUDA_QSA_INDEX_PREP_FIXTURE_TEST)
	test -n "$(QSA_INDEX_PREP_FIXTURE)"
	$(CUDA_QSA_INDEX_PREP_FIXTURE_TEST) "$(QSA_INDEX_PREP_FIXTURE)"

test-cuda-qsa-kv-pack-fixture: $(CUDA_QSA_KV_PACK_FIXTURE_TEST)
	test -n "$(QSA_KV_PACK_FIXTURE)"
	$(CUDA_QSA_KV_PACK_FIXTURE_TEST) "$(QSA_KV_PACK_FIXTURE)"

test-cuda-qsa-decode-xqa-fixture: $(CUDA_QSA_DECODE_XQA_FIXTURE_TEST)
	test -n "$(QSA_DECODE_XQA_FIXTURE)"
	$(CUDA_QSA_DECODE_XQA_FIXTURE_TEST) "$(QSA_DECODE_XQA_FIXTURE)"

test-cuda-qwen-moe-fixture: $(CUDA_QWEN_MOE_FIXTURE_TEST)
	test -n "$(QWEN_MOE_FIXTURE)"
	LD_LIBRARY_PATH=$(TVM_FFI_ROOT)/lib:$$LD_LIBRARY_PATH $(CUDA_QWEN_MOE_FIXTURE_TEST) "$(QWEN_MOE_FIXTURE)"

test-cuda-qwen-full-layer-fixture: $(CUDA_QWEN_FULL_LAYER_FIXTURE_TEST)
	test -n "$(QWEN_GDN_FIXTURE)"
	test -n "$(QWEN_MOE_FIXTURE)"
	LD_LIBRARY_PATH=$(TVM_FFI_ROOT)/lib:$$LD_LIBRARY_PATH \
		$(CUDA_QWEN_FULL_LAYER_FIXTURE_TEST) "$(QWEN_GDN_FIXTURE)" "$(QWEN_MOE_FIXTURE)"

$(CONTRACT_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/tests/kernel_contract_test.cc $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(KERNEL_DIR)/include $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/tests/kernel_contract_test.cc -o $(CONTRACT_TEST)

$(HEADER_C_TEST): $(KERNEL_DIR)/tests/kernel_header_c_test.c $(KERNEL_DIR)/include/flash/kernel_api.h $(KERNEL_DIR)/include/flash/qwen_expert_pack_api.h $(KERNEL_DIR)/include/flash/qwen_gdn_aux_api.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(KERNEL_DIR)/include $(KERNEL_DIR)/tests/kernel_header_c_test.c -o $(HEADER_C_TEST)

$(CUDA_GDN_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/internal/gdn_decode_backend.h $(KERNEL_DIR)/tests/gdn_decode_cuda_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_CUDA -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu \
		$(KERNEL_DIR)/tests/gdn_decode_cuda_test.cu -o $(CUDA_GDN_TEST)

$(CUDA_GDN_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/internal/gdn_decode_backend.h $(KERNEL_DIR)/internal/gdn_decode_flashinfer_backend.h $(KERNEL_DIR)/tests/gdn_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_CUDA \
		-DFLASH_WITH_FLASHINFER_GDN_AOT -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		-I$(TVM_FFI_ROOT)/include $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc \
		$(KERNEL_DIR)/tests/gdn_fixture_test.cu "$(GDN_AOT_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_GDN_FIXTURE_TEST)

$(CUDA_GDN_PREFILL_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/internal/gdn_decode_backend.h $(KERNEL_DIR)/internal/gdn_decode_flashinfer_backend.h $(KERNEL_DIR)/tests/gdn_prefill_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(GDN_PREFILL_AOT_OBJECTS)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	for object in $(GDN_PREFILL_AOT_OBJECTS); do test -f "$$object"; done
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_CUDA \
		-DFLASH_WITH_FLASHINFER_GDN_AOT \
		-DFLASH_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) -I$(TVM_FFI_ROOT)/include \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu \
		$(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc \
		$(KERNEL_DIR)/tests/gdn_prefill_fixture_test.cu "$(GDN_AOT_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_GDN_PREFILL_FIXTURE_TEST)

$(CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/cuda/gdn_block_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/internal/gdn_decode_backend.h $(KERNEL_DIR)/internal/gdn_decode_flashinfer_backend.h $(KERNEL_DIR)/internal/gdn_block_backend.h $(KERNEL_DIR)/internal/mhc_backend.h $(KERNEL_DIR)/tests/qwen_gdn_block_fixture_test.cc $(KERNEL_DIR)/include/flash/kernel_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(GDN_PREFILL_AOT_OBJECTS)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	for object in $(GDN_PREFILL_AOT_OBJECTS); do test -f "$$object"; done
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_CUDA \
		-DFLASH_WITH_FLASHINFER_GDN_AOT \
		-DFLASH_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-DFLASH_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		-I$(TVM_FFI_ROOT)/include $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc \
		$(KERNEL_DIR)/cuda/gdn_block_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu \
		$(KERNEL_DIR)/tests/qwen_gdn_block_fixture_test.cc "$(GDN_AOT_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST)

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

$(CUDA_NVFP4_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_dense_flashinfer.cu $(KERNEL_DIR)/internal/nvfp4_dense_backend.h $(KERNEL_DIR)/tests/nvfp4_dense_cuda_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DFLASH_WITH_FLASHINFER_NVFP4 \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(FLASHINFER_INCLUDES) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_dense_flashinfer.cu \
		$(KERNEL_DIR)/tests/nvfp4_dense_cuda_test.cu -o $(CUDA_NVFP4_TEST)

$(CUDA_NVFP4_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_dense_flashinfer.cu $(KERNEL_DIR)/internal/nvfp4_dense_backend.h $(KERNEL_DIR)/tests/nvfp4_dense_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DFLASH_WITH_FLASHINFER_NVFP4 \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(FLASHINFER_INCLUDES) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_dense_flashinfer.cu \
		$(KERNEL_DIR)/tests/nvfp4_dense_fixture_test.cu -o $(CUDA_NVFP4_FIXTURE_TEST)

$(CUDA_GROUPED_NVFP4_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/internal/nvfp4_grouped_backend.h $(KERNEL_DIR)/tests/nvfp4_grouped_cuda_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
		-DFLASH_WITH_FLASHINFER_GROUPED_NVFP4 -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(FLASHINFER_INCLUDES) $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu \
		$(KERNEL_DIR)/tests/nvfp4_grouped_cuda_test.cu -o $(CUDA_GROUPED_NVFP4_TEST)

$(CUDA_GROUPED_NVFP4_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/internal/nvfp4_grouped_backend.h $(KERNEL_DIR)/tests/nvfp4_grouped_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
		-DFLASH_WITH_FLASHINFER_GROUPED_NVFP4 -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(FLASHINFER_INCLUDES) $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu \
		$(KERNEL_DIR)/tests/nvfp4_grouped_fixture_test.cu \
		-o $(CUDA_GROUPED_NVFP4_FIXTURE_TEST)

$(CUDA_SILU_NVFP4_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu $(KERNEL_DIR)/internal/nvfp4_silu_backend.h $(KERNEL_DIR)/tests/nvfp4_silu_cute_test.cc $(KERNEL_DIR)/include/flash/kernel_api.h
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) -I$(TVM_FFI_ROOT)/include \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu \
		$(KERNEL_DIR)/tests/nvfp4_silu_cute_test.cc "$(CUTE_NVFP4_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcuda -lcudart -ldl \
		-Xcompiler=-pthread \
		-o $(CUDA_SILU_NVFP4_TEST)

$(CUDA_MOE_ROUTE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/internal/moe_route_backend.h $(KERNEL_DIR)/tests/moe_route_cuda_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_FLASHINFER_MOE_ROUTE \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/tests/moe_route_cuda_test.cu \
		-o $(CUDA_MOE_ROUTE_TEST)

$(CUDA_COHERENT_REGION_TEST): $(KERNEL_DIR)/fabric/coherent_region.cc $(KERNEL_DIR)/fabric/cuda_runtime.cc $(KERNEL_DIR)/tests/coherent_region_cuda_test.cu $(KERNEL_DIR)/include/flash/fabric_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(KERNEL_DIR)/include $(KERNEL_DIR)/fabric/coherent_region.cc \
		$(KERNEL_DIR)/fabric/cuda_runtime.cc \
		$(KERNEL_DIR)/tests/coherent_region_cuda_test.cu -lcublas \
		-o $(CUDA_COHERENT_REGION_TEST)

$(CUDA_FABRIC_SHARED): $(KERNEL_DIR)/fabric/coherent_region.cc $(KERNEL_DIR)/fabric/cuda_runtime.cc $(KERNEL_DIR)/include/flash/fabric_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -I$(KERNEL_DIR)/include \
		$(KERNEL_DIR)/fabric/coherent_region.cc $(KERNEL_DIR)/fabric/cuda_runtime.cc \
		-lcublas -o $(CUDA_FABRIC_SHARED)

$(CUDA_QWEN_EXPERT_PACK_SHARED): $(KERNEL_DIR)/cuda/qwen_expert_pack.cu $(KERNEL_DIR)/include/flash/qwen_expert_pack_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -I$(KERNEL_DIR)/include \
		$(KERNEL_DIR)/cuda/qwen_expert_pack.cu -o $(CUDA_QWEN_EXPERT_PACK_SHARED)

$(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST): $(KERNEL_DIR)/cuda/qwen_expert_pack.cu $(KERNEL_DIR)/tests/qwen_expert_pack_fixture_test.cu $(KERNEL_DIR)/include/flash/qwen_expert_pack_api.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(KERNEL_DIR)/include $(KERNEL_DIR)/cuda/qwen_expert_pack.cu \
		$(KERNEL_DIR)/tests/qwen_expert_pack_fixture_test.cu \
		-o $(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST)

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

$(CUDA_MOE_GATE_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/internal/moe_gate_backend.h $(KERNEL_DIR)/tests/moe_gate_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math \
		-DFLASH_WITH_SGLANG_CUBLAS_MOE_GATE -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_gate_sglang.cu \
		$(KERNEL_DIR)/tests/moe_gate_fixture_test.cu -lcublas \
		-o $(CUDA_MOE_GATE_FIXTURE_TEST)

$(CUDA_MOE_GATE_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/internal/moe_gate_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_CUBLAS_MOE_GATE -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_gate_sglang.cu -lcublas \
		-o $(CUDA_MOE_GATE_SHARED)

$(CUDA_SHARED_EXPERT_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/internal/shared_expert_backend.h $(KERNEL_DIR)/tests/shared_expert_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) \
		-DFLASH_WITH_SGLANG_CUBLAS_SHARED_EXPERT -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/shared_expert_sglang.cu \
		$(KERNEL_DIR)/tests/shared_expert_fixture_test.cu -lcublas \
		-o $(CUDA_SHARED_EXPERT_FIXTURE_TEST)

$(CUDA_SHARED_EXPERT_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/internal/shared_expert_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_CUBLAS_SHARED_EXPERT -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/shared_expert_sglang.cu -lcublas \
		-o $(CUDA_SHARED_EXPERT_SHARED)

$(CUDA_MOE_JOIN_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/internal/moe_join_backend.h $(KERNEL_DIR)/tests/moe_join_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math \
		-DFLASH_WITH_SGLANG_FUSED_MOE_JOIN -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_join_sglang.cu \
		$(KERNEL_DIR)/tests/moe_join_fixture_test.cu -o $(CUDA_MOE_JOIN_FIXTURE_TEST)

$(CUDA_MOE_JOIN_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/internal/moe_join_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_FUSED_MOE_JOIN -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/moe_join_sglang.cu \
		-o $(CUDA_MOE_JOIN_SHARED)

$(CUDA_MHC_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/internal/mhc_backend.h $(KERNEL_DIR)/tests/mhc_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/mhc_sglang.cu \
		$(KERNEL_DIR)/tests/mhc_fixture_test.cu -lcublas -o $(CUDA_MHC_FIXTURE_TEST)

$(CUDA_MHC_SHARED): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/internal/mhc_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/mhc_sglang.cu -lcublas \
		-o $(CUDA_MHC_SHARED)

$(CUDA_PLE_GATHER_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/ple_gather.cu $(KERNEL_DIR)/fabric/coherent_region.cc $(KERNEL_DIR)/internal/ple_gather_backend.h $(KERNEL_DIR)/tests/ple_gather_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h $(KERNEL_DIR)/include/flash/fabric_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_SGLANG_PLE_GATHER \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/ple_gather.cu \
		$(KERNEL_DIR)/fabric/coherent_region.cc \
		$(KERNEL_DIR)/tests/ple_gather_fixture_test.cu -o $(CUDA_PLE_GATHER_FIXTURE_TEST)

$(CUDA_QSA_TOPK_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_topk_sglang.cu $(KERNEL_DIR)/internal/qsa_topk_backend.h $(KERNEL_DIR)/tests/qsa_topk_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_SGLANG_QSA_TOPK \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_topk_sglang.cu \
		$(KERNEL_DIR)/tests/qsa_topk_fixture_test.cu -o $(CUDA_QSA_TOPK_FIXTURE_TEST)

$(CUDA_QSA_EXPAND_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_expand_sglang.cu $(KERNEL_DIR)/internal/qsa_expand_backend.h $(KERNEL_DIR)/tests/qsa_expand_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_SGLANG_QSA_EXPAND \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_expand_sglang.cu \
		$(KERNEL_DIR)/tests/qsa_expand_fixture_test.cu -o $(CUDA_QSA_EXPAND_FIXTURE_TEST)

$(CUDA_QSA_SCORE_OBJECT): $(KERNEL_DIR)/cuda/qsa_score_tilelang.cu $(KERNEL_DIR)/internal/qsa_score_backend.h $(KERNEL_DIR)/include/flash/kernel_api.h vendor/tilelang-qsa-score/generated/device_kernel.cu $(wildcard vendor/tilelang-qsa-score/include/tl_templates/cuda/*.h) $(wildcard vendor/tilelang-qsa-score/include/tl_templates/cuda/instruction/*.h)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -DENABLE_BF16 -Xcompiler=-fPIC \
		-I. -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(TILELANG_QSA_SCORE_INCLUDE) \
		-I$(CUTLASS_ROOT)/include -c $(KERNEL_DIR)/cuda/qsa_score_tilelang.cu \
		-o $(CUDA_QSA_SCORE_OBJECT)

$(CUDA_QSA_SCORE_FIXTURE_TEST): $(CUDA_QSA_SCORE_OBJECT) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/tests/qsa_score_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_TILELANG_QSA_SCORE \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/tests/qsa_score_fixture_test.cu $(CUDA_QSA_SCORE_OBJECT) \
		-o $(CUDA_QSA_SCORE_FIXTURE_TEST)

$(CUDA_QSA_INDEX_PREP_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_index_prep_sglang.cu $(KERNEL_DIR)/internal/qsa_index_prep_backend.h $(KERNEL_DIR)/tests/qsa_index_prep_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_SGLANG_QSA_INDEX_PREP \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_index_prep_sglang.cu \
		$(KERNEL_DIR)/tests/qsa_index_prep_fixture_test.cu -o $(CUDA_QSA_INDEX_PREP_FIXTURE_TEST)

$(CUDA_QSA_KV_PACK_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_kv_pack_sglang.cu $(KERNEL_DIR)/internal/qsa_kv_pack_backend.h $(KERNEL_DIR)/tests/qsa_kv_pack_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFLASH_WITH_SGLANG_QSA_KV_PACK \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_kv_pack_sglang.cu \
		$(KERNEL_DIR)/tests/qsa_kv_pack_fixture_test.cu -o $(CUDA_QSA_KV_PACK_FIXTURE_TEST)

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

$(CUDA_QSA_DECODE_XQA_FIXTURE_TEST): $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/qsa_decode_xqa_flashinfer.cu $(KERNEL_DIR)/internal/qsa_decode_backend.h $(KERNEL_DIR)/tests/qsa_decode_xqa_fixture_test.cu $(KERNEL_DIR)/include/flash/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) \
		-DFLASH_WITH_FLASHINFER_XQA_DECODE -I$(KERNEL_DIR)/include -I$(KERNEL_DIR) \
		$(FLASHINFER_XQA_INCLUDE) $(KERNEL_DIR)/kernel_contract.cc \
		$(KERNEL_DIR)/cuda/qsa_decode_xqa_flashinfer.cu \
		$(KERNEL_DIR)/tests/qsa_decode_xqa_fixture_test.cu \
		$(CUDA_QSA_DECODE_XQA_MHA_OBJECT) -lcuda \
		-o $(CUDA_QSA_DECODE_XQA_FIXTURE_TEST)

$(CUDA_QWEN_MOE_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu $(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu $(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/internal/nvfp4_grouped_backend.h $(KERNEL_DIR)/internal/nvfp4_silu_backend.h $(KERNEL_DIR)/internal/nvfp4_quantize_backend.h $(KERNEL_DIR)/internal/moe_route_backend.h $(KERNEL_DIR)/internal/moe_gate_backend.h $(KERNEL_DIR)/internal/shared_expert_backend.h $(KERNEL_DIR)/internal/moe_join_backend.h $(KERNEL_DIR)/internal/mhc_backend.h $(KERNEL_DIR)/tests/qwen_moe_fixture_test.cc $(KERNEL_DIR)/include/flash/kernel_api.h
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	test -f "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(FLASHINFER_ARCH_FLAGS) -diag-suppress 177 \
		-diag-suppress 549 -DFLASH_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DFLASH_WITH_FLASHINFER_MOE_ROUTE \
		-DFLASH_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DFLASH_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DFLASH_WITH_SGLANG_FUSED_MOE_JOIN \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(FLASHINFER_INCLUDES) -I$(TVM_FFI_ROOT)/include \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu \
		$(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu $(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu \
		$(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/cuda/moe_gate_sglang.cu \
		$(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/cuda/moe_join_sglang.cu \
		$(KERNEL_DIR)/cuda/mhc_sglang.cu \
		$(KERNEL_DIR)/tests/qwen_moe_fixture_test.cc \
		"$(CUTE_NVFP4_OBJECT)" "$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_MOE_FIXTURE_TEST)

$(CUDA_QWEN_FULL_LAYER_FIXTURE_TEST): $(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu $(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/cuda/gdn_block_sglang.cu $(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu $(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu $(KERNEL_DIR)/cuda/moe_route_flashinfer.cu $(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/cuda/shared_expert_sglang.cu $(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu $(KERNEL_DIR)/tests/qwen_gdn_block_fixture_test.cc $(KERNEL_DIR)/tests/qwen_moe_fixture_test.cc $(KERNEL_DIR)/tests/qwen_full_layer_fixture_test.cc $(KERNEL_DIR)/include/flash/kernel_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	test -f "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(FLASHINFER_ARCH_FLAGS) -diag-suppress 177 \
		-diag-suppress 549 -DFLASH_FIXTURE_LIBRARY \
		-DFLASH_WITH_FLASHINFER_GDN_AOT \
		-DFLASH_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DFLASH_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DFLASH_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DFLASH_WITH_FLASHINFER_MOE_ROUTE \
		-DFLASH_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DFLASH_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DFLASH_WITH_SGLANG_FUSED_MOE_JOIN \
		-DFLASH_WITH_SGLANG_CUBLAS_MHC \
		-I$(KERNEL_DIR)/include -I$(KERNEL_DIR) $(FLASHINFER_INCLUDES) -I$(TVM_FFI_ROOT)/include \
		$(KERNEL_DIR)/kernel_contract.cc $(KERNEL_DIR)/cuda/gdn_decode.cu \
		$(KERNEL_DIR)/cuda/gdn_decode_flashinfer_cute.cc $(KERNEL_DIR)/cuda/gdn_block_sglang.cu \
		$(KERNEL_DIR)/cuda/nvfp4_grouped_flashinfer.cu $(KERNEL_DIR)/cuda/nvfp4_silu_cute.cu \
		$(KERNEL_DIR)/cuda/nvfp4_quantize_cute.cu $(KERNEL_DIR)/cuda/moe_route_flashinfer.cu \
		$(KERNEL_DIR)/cuda/moe_gate_sglang.cu $(KERNEL_DIR)/cuda/shared_expert_sglang.cu \
		$(KERNEL_DIR)/cuda/moe_join_sglang.cu $(KERNEL_DIR)/cuda/mhc_sglang.cu \
		$(KERNEL_DIR)/tests/qwen_gdn_block_fixture_test.cc \
		$(KERNEL_DIR)/tests/qwen_moe_fixture_test.cc \
		$(KERNEL_DIR)/tests/qwen_full_layer_fixture_test.cc \
		"$(GDN_AOT_OBJECT)" "$(CUTE_NVFP4_OBJECT)" \
		"$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_FULL_LAYER_FIXTURE_TEST)

clean:
	rm -rf -- $(BUILD_DIR)

CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Werror
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror
AR ?= ar

BUILD_DIR := build
CONTRACT_TEST := $(BUILD_DIR)/kernel-contract-test
HEADER_C_TEST := $(BUILD_DIR)/kernel-header-c-test
CUDA_GDN_TEST := $(BUILD_DIR)/gdn-decode-cuda-test
CUDA_GDN_FIXTURE_TEST := $(BUILD_DIR)/gdn-fixture-test
CUDA_GDN_PREFILL_FIXTURE_TEST := $(BUILD_DIR)/gdn-prefill-fixture-test
CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST := $(BUILD_DIR)/qwen-gdn-block-fixture-test
CUDA_QWEN_GDN_SHARED := $(BUILD_DIR)/libsparkserve-qwen-gdn.so
CUDA_NVFP4_TEST := $(BUILD_DIR)/nvfp4-dense-cuda-test
CUDA_NVFP4_FIXTURE_TEST := $(BUILD_DIR)/nvfp4-dense-fixture-test
CUDA_GROUPED_NVFP4_TEST := $(BUILD_DIR)/nvfp4-grouped-cuda-test
CUDA_GROUPED_NVFP4_FIXTURE_TEST := $(BUILD_DIR)/nvfp4-grouped-fixture-test
CUDA_SILU_NVFP4_TEST := $(BUILD_DIR)/nvfp4-silu-cute-test
CUDA_QWEN_MOE_FIXTURE_TEST := $(BUILD_DIR)/qwen-moe-fixture-test
CUDA_QWEN_MOE_SHARED := $(BUILD_DIR)/libsparkserve-qwen-moe.so
CUDA_QWEN_FULL_LAYER_FIXTURE_TEST := $(BUILD_DIR)/qwen-full-layer-fixture-test
CUDA_MOE_ROUTE_TEST := $(BUILD_DIR)/moe-route-cuda-test
CUDA_MOE_GATE_FIXTURE_TEST := $(BUILD_DIR)/moe-gate-fixture-test
CUDA_MOE_GATE_SHARED := $(BUILD_DIR)/libsparkserve-moe-gate.so
CUDA_SHARED_EXPERT_FIXTURE_TEST := $(BUILD_DIR)/shared-expert-fixture-test
CUDA_SHARED_EXPERT_SHARED := $(BUILD_DIR)/libsparkserve-shared-expert.so
CUDA_MOE_JOIN_FIXTURE_TEST := $(BUILD_DIR)/moe-join-fixture-test
CUDA_MOE_JOIN_SHARED := $(BUILD_DIR)/libsparkserve-moe-join.so
CUDA_MHC_FIXTURE_TEST := $(BUILD_DIR)/mhc-fixture-test
CUDA_MHC_SHARED := $(BUILD_DIR)/libsparkserve-mhc.so
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
CUDA_QSA_SHARED := $(BUILD_DIR)/libsparkserve-qsa.so
CUDA_QWEN_QSA_BLOCK_SHARED := $(BUILD_DIR)/libsparkserve-qwen-qsa-block.so
CUDA_QWEN_PLE_BLOCK_SHARED := $(BUILD_DIR)/libsparkserve-qwen-ple-block.so
CUDA_QWEN_DECODE_GLUE_SHARED := $(BUILD_DIR)/libsparkserve-qwen-decode-glue.so
CUDA_QWEN_RUNTIME_SHARED := $(BUILD_DIR)/libsparkserve-qwen-runtime.so
CUDA_FABRIC_SHARED := $(BUILD_DIR)/libsparkserve-fabric.so
CUDA_QWEN_EXPERT_PACK_SHARED := $(BUILD_DIR)/libsparkserve-qwen-expert-pack.so
CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST := $(BUILD_DIR)/qwen-expert-pack-fixture-test
CUDA_GGML_QUANT_SHARED := $(BUILD_DIR)/libsparkserve-ggml-quant.so
CUDA_GGML_QUANT_FIXTURE_TEST := $(BUILD_DIR)/ggml-quant-fixture-test
CUDA_GLM_KDA_SHARED := $(BUILD_DIR)/libsparkserve-glm-kda.so
CUDA_GLM_KDA_FIXTURE_TEST := $(BUILD_DIR)/glm-kda-fixture-test
CUDA_GLM_KPOOL_SHARED := $(BUILD_DIR)/libsparkserve-glm-kpool.so
CUDA_GLM_KPOOL_FIXTURE_TEST := $(BUILD_DIR)/glm-kpool-fixture-test
CUDA_GLM_PAGED_MQA_SHARED := $(BUILD_DIR)/libsparkserve-glm-paged-mqa.so
CUDA_GLM_PAGED_MQA_FIXTURE_TEST := $(BUILD_DIR)/glm-paged-mqa-fixture-test
CUDA_GLM_SPARSE_MLA_SHARED := $(BUILD_DIR)/libsparkserve-glm-sparse-mla.so
CUDA_GLM_SPARSE_MLA_FIXTURE_TEST := $(BUILD_DIR)/glm-sparse-mla-fixture-test
DS4_GLM53_ROOT ?= third_party/_deps/ds4-glm53
DS4_GLM53_BUILD_DIR ?= build-spark
DS4_GLM53_ADAPTER_OBJECT := $(DS4_GLM53_BUILD_DIR)/ds4-glm53-adapter.o
DS4_GLM53_STATIC := $(DS4_GLM53_BUILD_DIR)/libsparkserve-ds4-glm53.a
DS4_GLM53_CORE_TARGETS := ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o \
	ds4_cuda.o ds4_layer_pack.o cuda/mmq/ds4_ggml_stubs.o \
	cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o \
	cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
DS4_GLM53_CORE_OBJECTS := $(addprefix $(DS4_GLM53_ROOT)/,$(DS4_GLM53_CORE_TARGETS))
NVCC ?= nvcc
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -O2 -std=c++20 -arch=$(CUDA_ARCH)
FLASHINFER_ROOT ?= third_party/_deps/flashinfer
FLASHINFER_INCLUDE ?= $(FLASHINFER_ROOT)/include
CUTLASS_ROOT ?= $(FLASHINFER_ROOT)/3rdparty/cutlass
# CUDA 13.0 exposes exact sm_121 code generation but does not set the helper
# macro checked by this FlashInfer revision's architecture guard.
FLASHINFER_ARCH_FLAGS ?= -D__CUDA_ARCH_SPECIFIC__ --expt-relaxed-constexpr \
	-diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20015 \
	-diag-suppress 2908
FLASHINFER_INCLUDES := -I$(FLASHINFER_INCLUDE) \
	-I$(CUTLASS_ROOT)/include \
	-I$(CUTLASS_ROOT)/tools/util/include
FLASHINFER_XQA_INCLUDE := -I$(FLASHINFER_ROOT)/csrc/xqa
TILELANG_QSA_SCORE_INCLUDE := -Ithird_party/tilelang-qsa-score/include
GGML_MMVQ_ROOT := third_party/llama-ggml-mmvq
DEEPGEMM_ROOT ?= third_party/_deps/deepgemm
DEEPGEMM_INCLUDES := -I$(DEEPGEMM_ROOT)/deep_gemm/include \
	-I$(DEEPGEMM_ROOT)/third-party/cutlass/include \
	-I$(DEEPGEMM_ROOT)/third-party/cutlass/tools/util/include
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

.PHONY: test test-cpp test-cuda fetch-ds4-glm53 glm53-ds4-static fabric-shared qwen-expert-pack-shared qwen-moe-shared qwen-gdn-shared qwen-qsa-block-shared qwen-ple-block-shared qwen-decode-glue-shared qwen-runtime-shared ggml-quant-shared glm-kda-shared glm-kpool-shared glm-paged-mqa-shared glm-sparse-mla-shared qsa-shared moe-gate-shared shared-expert-shared moe-join-shared mhc-shared test-cuda-fabric test-cuda-qwen-expert-pack-fixture test-cuda-ggml-quant-fixture test-cuda-glm-kda-fixture test-cuda-glm-kpool-fixture test-cuda-glm-paged-mqa-fixture test-cuda-glm-sparse-mla-fixture test-cuda-gdn test-cuda-gdn-fixture test-cuda-gdn-prefill-fixture test-cuda-qwen-gdn-block-fixture test-cuda-nvfp4 test-cuda-nvfp4-fixture test-cuda-grouped-nvfp4 test-cuda-grouped-nvfp4-fixture test-cuda-silu-nvfp4 test-cuda-silu-nvfp4-fixture test-cuda-moe-route test-cuda-moe-gate-fixture test-cuda-shared-expert-fixture test-cuda-moe-join-fixture test-cuda-mhc-fixture test-cuda-ple-gather-fixture test-cuda-qsa-topk-fixture test-cuda-qsa-expand-fixture test-cuda-qsa-score-fixture test-cuda-qsa-index-prep-fixture test-cuda-qsa-kv-pack-fixture test-cuda-qsa-decode-xqa-fixture test-cuda-qwen-moe-fixture test-cuda-qwen-full-layer-fixture docker-flash-next-sm121 clean

test: test-cpp

test-cpp: $(CONTRACT_TEST) $(HEADER_C_TEST)
	$(CONTRACT_TEST)
	$(HEADER_C_TEST)

test-cuda: test-cuda-fabric test-cuda-gdn test-cuda-nvfp4 test-cuda-grouped-nvfp4 test-cuda-moe-route

fetch-ds4-glm53:
	./scripts/fetch-ds4-glm53-sources.sh "$(DS4_GLM53_ROOT)"

glm53-ds4-static: $(DS4_GLM53_STATIC)

$(DS4_GLM53_STATIC): csrc/ds4_glm53_adapter.c csrc/include/sparkserve/ds4_glm53_api.h $(DS4_GLM53_ROOT)/ds4.h third_party/ds4-glm53/source-files.sha256
	(cd "$(DS4_GLM53_ROOT)" && sha256sum -c "$(abspath third_party/ds4-glm53/source-files.sha256)")
	mkdir -p $(DS4_GLM53_BUILD_DIR)
	$(MAKE) -C "$(DS4_GLM53_ROOT)" $(DS4_GLM53_CORE_TARGETS) CUDA_ARCH=$(CUDA_ARCH)
	$(CC) $(CFLAGS) -I csrc/include -I "$(DS4_GLM53_ROOT)" \
		-c csrc/ds4_glm53_adapter.c -o $(DS4_GLM53_ADAPTER_OBJECT)
	$(AR) rcs $(DS4_GLM53_STATIC) $(DS4_GLM53_ADAPTER_OBJECT) $(DS4_GLM53_CORE_OBJECTS)

test-cuda-fabric: $(CUDA_COHERENT_REGION_TEST)
	$(CUDA_COHERENT_REGION_TEST)

test-cuda-qwen-expert-pack-fixture: $(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST)
	$(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST)

test-cuda-ggml-quant-fixture: $(CUDA_GGML_QUANT_FIXTURE_TEST)
	test -n "$(GGML_QUANT_FIXTURE)"
	$(CUDA_GGML_QUANT_FIXTURE_TEST) "$(GGML_QUANT_FIXTURE)"

test-cuda-glm-kda-fixture: $(CUDA_GLM_KDA_FIXTURE_TEST)
	$(CUDA_GLM_KDA_FIXTURE_TEST)

test-cuda-glm-kpool-fixture: $(CUDA_GLM_KPOOL_FIXTURE_TEST)
	$(CUDA_GLM_KPOOL_FIXTURE_TEST)

test-cuda-glm-paged-mqa-fixture: $(CUDA_GLM_PAGED_MQA_FIXTURE_TEST)
	$(CUDA_GLM_PAGED_MQA_FIXTURE_TEST)

test-cuda-glm-sparse-mla-fixture: $(CUDA_GLM_SPARSE_MLA_FIXTURE_TEST)
	$(CUDA_GLM_SPARSE_MLA_FIXTURE_TEST)

fabric-shared: $(CUDA_FABRIC_SHARED)

qwen-expert-pack-shared: $(CUDA_QWEN_EXPERT_PACK_SHARED)

qwen-moe-shared: $(CUDA_QWEN_MOE_SHARED)

qwen-gdn-shared: $(CUDA_QWEN_GDN_SHARED)

qwen-qsa-block-shared: $(CUDA_QWEN_QSA_BLOCK_SHARED)

qwen-ple-block-shared: $(CUDA_QWEN_PLE_BLOCK_SHARED)

qwen-decode-glue-shared: $(CUDA_QWEN_DECODE_GLUE_SHARED)

qwen-runtime-shared: $(CUDA_QWEN_RUNTIME_SHARED)

ggml-quant-shared: $(CUDA_GGML_QUANT_SHARED)

glm-kda-shared: $(CUDA_GLM_KDA_SHARED)

glm-kpool-shared: $(CUDA_GLM_KPOOL_SHARED)

glm-paged-mqa-shared: $(CUDA_GLM_PAGED_MQA_SHARED)

glm-sparse-mla-shared: $(CUDA_GLM_SPARSE_MLA_SHARED)

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

docker-flash-next-sm121:
	docker build -t sparkserve/sglang:qwen38flashnext-sm121 -f docker/flashnext-sm121/Dockerfile .

$(CONTRACT_TEST): csrc/kernel_contract.cc csrc/tests/kernel_contract_test.cc csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Icsrc/include csrc/kernel_contract.cc csrc/tests/kernel_contract_test.cc -o $(CONTRACT_TEST)

$(HEADER_C_TEST): csrc/tests/kernel_header_c_test.c csrc/include/sparkserve/kernel_api.h csrc/include/sparkserve/glm_mqa_api.h csrc/include/sparkserve/glm_sparse_mla_api.h csrc/include/sparkserve/qwen_expert_pack_api.h csrc/include/sparkserve/qwen_gdn_aux_api.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -Icsrc/include csrc/tests/kernel_header_c_test.c -o $(HEADER_C_TEST)

$(CUDA_GDN_TEST): csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu csrc/internal/gdn_decode_backend.h csrc/tests/gdn_decode_cuda_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_CUDA -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu \
		csrc/tests/gdn_decode_cuda_test.cu -o $(CUDA_GDN_TEST)

$(CUDA_GDN_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/internal/gdn_decode_backend.h csrc/internal/gdn_decode_flashinfer_backend.h csrc/tests/gdn_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_CUDA \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_AOT -Icsrc/include -Icsrc \
		-I$(TVM_FFI_ROOT)/include csrc/kernel_contract.cc \
		csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc \
		csrc/tests/gdn_fixture_test.cu "$(GDN_AOT_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_GDN_FIXTURE_TEST)

$(CUDA_GDN_PREFILL_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/internal/gdn_decode_backend.h csrc/internal/gdn_decode_flashinfer_backend.h csrc/tests/gdn_prefill_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(GDN_PREFILL_AOT_OBJECTS)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	for object in $(GDN_PREFILL_AOT_OBJECTS); do test -f "$$object"; done
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_CUDA \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_AOT \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-Icsrc/include -Icsrc -I$(TVM_FFI_ROOT)/include \
		csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu \
		csrc/cuda/gdn_decode_flashinfer_cute.cc \
		csrc/tests/gdn_prefill_fixture_test.cu "$(GDN_AOT_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_GDN_PREFILL_FIXTURE_TEST)

$(CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/cuda/gdn_block_sglang.cu csrc/cuda/mhc_sglang.cu csrc/internal/gdn_decode_backend.h csrc/internal/gdn_decode_flashinfer_backend.h csrc/internal/gdn_block_backend.h csrc/internal/mhc_backend.h csrc/tests/qwen_gdn_block_fixture_test.cc csrc/include/sparkserve/kernel_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(GDN_PREFILL_AOT_OBJECTS)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	for object in $(GDN_PREFILL_AOT_OBJECTS); do test -f "$$object"; done
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_CUDA \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_AOT \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MHC -Icsrc/include -Icsrc \
		-I$(TVM_FFI_ROOT)/include csrc/kernel_contract.cc \
		csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc \
		csrc/cuda/gdn_block_sglang.cu csrc/cuda/mhc_sglang.cu \
		csrc/tests/qwen_gdn_block_fixture_test.cc "$(GDN_AOT_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_GDN_BLOCK_FIXTURE_TEST)

$(CUDA_QWEN_GDN_SHARED): csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/cuda/gdn_block_sglang.cu csrc/cuda/mhc_sglang.cu csrc/cuda/qwen_gdn_aux.cu csrc/internal/gdn_decode_backend.h csrc/internal/gdn_decode_flashinfer_backend.h csrc/internal/gdn_block_backend.h csrc/internal/mhc_backend.h csrc/include/sparkserve/kernel_api.h csrc/include/sparkserve/qwen_gdn_aux_api.h
	test -n "$(GDN_AOT_OBJECT)"
	test -n "$(GDN_PREFILL_AOT_OBJECTS)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(GDN_AOT_OBJECT)"
	for object in $(GDN_PREFILL_AOT_OBJECTS); do test -f "$$object"; done
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_CUDA -DSPARKSERVE_WITH_FLASHINFER_GDN_AOT \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MHC -Icsrc/include -Icsrc \
		-I$(TVM_FFI_ROOT)/include csrc/kernel_contract.cc \
		csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc \
		csrc/cuda/gdn_block_sglang.cu csrc/cuda/mhc_sglang.cu \
		csrc/cuda/qwen_gdn_aux.cu "$(GDN_AOT_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_GDN_SHARED)

$(CUDA_NVFP4_TEST): csrc/kernel_contract.cc csrc/cuda/nvfp4_dense_flashinfer.cu csrc/internal/nvfp4_dense_backend.h csrc/tests/nvfp4_dense_cuda_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DSPARKSERVE_WITH_FLASHINFER_NVFP4 \
		-Icsrc/include -Icsrc $(FLASHINFER_INCLUDES) \
		csrc/kernel_contract.cc csrc/cuda/nvfp4_dense_flashinfer.cu \
		csrc/tests/nvfp4_dense_cuda_test.cu -o $(CUDA_NVFP4_TEST)

$(CUDA_NVFP4_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/nvfp4_dense_flashinfer.cu csrc/internal/nvfp4_dense_backend.h csrc/tests/nvfp4_dense_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DSPARKSERVE_WITH_FLASHINFER_NVFP4 \
		-Icsrc/include -Icsrc $(FLASHINFER_INCLUDES) \
		csrc/kernel_contract.cc csrc/cuda/nvfp4_dense_flashinfer.cu \
		csrc/tests/nvfp4_dense_fixture_test.cu -o $(CUDA_NVFP4_FIXTURE_TEST)

$(CUDA_GROUPED_NVFP4_TEST): csrc/kernel_contract.cc csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/internal/nvfp4_grouped_backend.h csrc/tests/nvfp4_grouped_cuda_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
		-DSPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4 -Icsrc/include -Icsrc \
		$(FLASHINFER_INCLUDES) csrc/kernel_contract.cc \
		csrc/cuda/nvfp4_grouped_flashinfer.cu \
		csrc/tests/nvfp4_grouped_cuda_test.cu -o $(CUDA_GROUPED_NVFP4_TEST)

$(CUDA_GROUPED_NVFP4_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/internal/nvfp4_grouped_backend.h csrc/tests/nvfp4_grouped_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
		-DSPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4 -Icsrc/include -Icsrc \
		$(FLASHINFER_INCLUDES) csrc/kernel_contract.cc \
		csrc/cuda/nvfp4_grouped_flashinfer.cu \
		csrc/tests/nvfp4_grouped_fixture_test.cu \
		-o $(CUDA_GROUPED_NVFP4_FIXTURE_TEST)

$(CUDA_SILU_NVFP4_TEST): csrc/kernel_contract.cc csrc/cuda/nvfp4_silu_cute.cc csrc/internal/nvfp4_silu_backend.h csrc/tests/nvfp4_silu_cute_test.cc csrc/include/sparkserve/kernel_api.h
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-Icsrc/include -Icsrc -I$(TVM_FFI_ROOT)/include \
		csrc/kernel_contract.cc csrc/cuda/nvfp4_silu_cute.cc \
		csrc/tests/nvfp4_silu_cute_test.cc "$(CUTE_NVFP4_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcuda -lcudart -ldl \
		-Xcompiler=-pthread \
		-o $(CUDA_SILU_NVFP4_TEST)

$(CUDA_MOE_ROUTE_TEST): csrc/kernel_contract.cc csrc/cuda/moe_route_flashinfer.cu csrc/internal/moe_route_backend.h csrc/tests/moe_route_cuda_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_FLASHINFER_MOE_ROUTE \
		-Icsrc/include -Icsrc csrc/kernel_contract.cc \
		csrc/cuda/moe_route_flashinfer.cu csrc/tests/moe_route_cuda_test.cu \
		-o $(CUDA_MOE_ROUTE_TEST)

$(CUDA_COHERENT_REGION_TEST): csrc/fabric/coherent_region.cc csrc/fabric/cuda_runtime.cc csrc/tests/coherent_region_cuda_test.cu csrc/include/sparkserve/fabric_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icsrc/include csrc/fabric/coherent_region.cc \
		csrc/fabric/cuda_runtime.cc \
		csrc/tests/coherent_region_cuda_test.cu -lcublas \
		-o $(CUDA_COHERENT_REGION_TEST)

$(CUDA_FABRIC_SHARED): csrc/fabric/coherent_region.cc csrc/fabric/cuda_runtime.cc csrc/include/sparkserve/fabric_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -Icsrc/include \
		csrc/fabric/coherent_region.cc csrc/fabric/cuda_runtime.cc \
		-lcublas -o $(CUDA_FABRIC_SHARED)

$(CUDA_QWEN_EXPERT_PACK_SHARED): csrc/cuda/qwen_expert_pack.cu csrc/include/sparkserve/qwen_expert_pack_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -Icsrc/include \
		csrc/cuda/qwen_expert_pack.cu -o $(CUDA_QWEN_EXPERT_PACK_SHARED)

$(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST): csrc/cuda/qwen_expert_pack.cu csrc/tests/qwen_expert_pack_fixture_test.cu csrc/include/sparkserve/qwen_expert_pack_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icsrc/include csrc/cuda/qwen_expert_pack.cu \
		csrc/tests/qwen_expert_pack_fixture_test.cu \
		-o $(CUDA_QWEN_EXPERT_PACK_FIXTURE_TEST)

$(CUDA_QWEN_MOE_SHARED): csrc/kernel_contract.cc csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu csrc/internal/nvfp4_grouped_backend.h csrc/internal/nvfp4_silu_backend.h csrc/internal/nvfp4_quantize_backend.h csrc/internal/moe_route_backend.h csrc/internal/moe_gate_backend.h csrc/internal/shared_expert_backend.h csrc/internal/moe_join_backend.h csrc/include/sparkserve/kernel_api.h
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	test -f "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(FLASHINFER_ARCH_FLAGS) \
		-diag-suppress 177 -diag-suppress 549 -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DSPARKSERVE_WITH_FLASHINFER_MOE_ROUTE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DSPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN \
		-Icsrc/include -Icsrc $(FLASHINFER_INCLUDES) \
		-I$(TVM_FFI_ROOT)/include csrc/kernel_contract.cc \
		csrc/cuda/nvfp4_grouped_flashinfer.cu \
		csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc \
		csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu \
		csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu \
		"$(CUTE_NVFP4_OBJECT)" "$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_MOE_SHARED)

$(CUDA_GGML_QUANT_SHARED): csrc/cuda/ggml_quant_mmvq.cu csrc/cuda/ggml_runtime_shim.cu csrc/include/sparkserve/ggml_quant_api.h csrc/include/sparkserve/kernel_api.h $(GGML_MMVQ_ROOT)/mmvq.cu $(GGML_MMVQ_ROOT)/mmvq.cuh $(GGML_MMVQ_ROOT)/quantize.cu $(GGML_MMVQ_ROOT)/quantize.cuh $(GGML_MMVQ_ROOT)/common.cuh $(GGML_MMVQ_ROOT)/ggml-common.h $(GGML_MMVQ_ROOT)/vecdotq.cuh
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -Icsrc/include \
		-I$(GGML_MMVQ_ROOT) csrc/cuda/ggml_runtime_shim.cu \
		$(GGML_MMVQ_ROOT)/quantize.cu $(GGML_MMVQ_ROOT)/mmvq.cu \
		csrc/cuda/ggml_quant_mmvq.cu -o $(CUDA_GGML_QUANT_SHARED)

$(CUDA_GGML_QUANT_FIXTURE_TEST): csrc/cuda/ggml_quant_mmvq.cu csrc/cuda/ggml_runtime_shim.cu csrc/tests/ggml_quant_fixture_test.cu csrc/include/sparkserve/ggml_quant_api.h csrc/include/sparkserve/kernel_api.h $(GGML_MMVQ_ROOT)/mmvq.cu $(GGML_MMVQ_ROOT)/mmvq.cuh $(GGML_MMVQ_ROOT)/quantize.cu $(GGML_MMVQ_ROOT)/quantize.cuh $(GGML_MMVQ_ROOT)/common.cuh $(GGML_MMVQ_ROOT)/ggml-common.h $(GGML_MMVQ_ROOT)/vecdotq.cuh
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icsrc/include -I$(GGML_MMVQ_ROOT) \
		csrc/cuda/ggml_runtime_shim.cu $(GGML_MMVQ_ROOT)/quantize.cu \
		$(GGML_MMVQ_ROOT)/mmvq.cu csrc/cuda/ggml_quant_mmvq.cu \
		csrc/tests/ggml_quant_fixture_test.cu \
		-o $(CUDA_GGML_QUANT_FIXTURE_TEST)

$(CUDA_GLM_KDA_SHARED): csrc/cuda/glm_kda_llama.cu csrc/include/sparkserve/glm_kda_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -Icsrc/include \
		csrc/cuda/glm_kda_llama.cu -o $(CUDA_GLM_KDA_SHARED)

$(CUDA_GLM_KDA_FIXTURE_TEST): csrc/cuda/glm_kda_llama.cu csrc/tests/glm_kda_fixture_test.cu csrc/include/sparkserve/glm_kda_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icsrc/include csrc/cuda/glm_kda_llama.cu \
		csrc/tests/glm_kda_fixture_test.cu -o $(CUDA_GLM_KDA_FIXTURE_TEST)

$(CUDA_GLM_KPOOL_SHARED): csrc/cuda/glm_kpool_sglang.cu csrc/include/sparkserve/glm_dsa_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -Icsrc/include \
		csrc/cuda/glm_kpool_sglang.cu -o $(CUDA_GLM_KPOOL_SHARED)

$(CUDA_GLM_KPOOL_FIXTURE_TEST): csrc/cuda/glm_kpool_sglang.cu csrc/tests/glm_kpool_fixture_test.cu csrc/include/sparkserve/glm_dsa_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icsrc/include csrc/cuda/glm_kpool_sglang.cu \
		csrc/tests/glm_kpool_fixture_test.cu -o $(CUDA_GLM_KPOOL_FIXTURE_TEST)

$(CUDA_GLM_PAGED_MQA_SHARED): csrc/cuda/glm_paged_mqa_deepgemm.cu csrc/include/sparkserve/glm_mqa_api.h csrc/include/sparkserve/kernel_api.h
	test -f "$(DEEPGEMM_ROOT)/deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr -shared -Xcompiler=-fPIC \
		-Icsrc/include $(DEEPGEMM_INCLUDES) \
		csrc/cuda/glm_paged_mqa_deepgemm.cu -lcuda \
		-o $(CUDA_GLM_PAGED_MQA_SHARED)

$(CUDA_GLM_PAGED_MQA_FIXTURE_TEST): csrc/cuda/glm_paged_mqa_deepgemm.cu csrc/tests/glm_paged_mqa_fixture_test.cu csrc/include/sparkserve/glm_mqa_api.h csrc/include/sparkserve/kernel_api.h
	test -f "$(DEEPGEMM_ROOT)/deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr -Icsrc/include \
		$(DEEPGEMM_INCLUDES) csrc/cuda/glm_paged_mqa_deepgemm.cu \
		csrc/tests/glm_paged_mqa_fixture_test.cu -lcuda \
		-o $(CUDA_GLM_PAGED_MQA_FIXTURE_TEST)

$(CUDA_GLM_SPARSE_MLA_SHARED): csrc/cuda/glm_sparse_mla_flashinfer.cu csrc/include/sparkserve/glm_sparse_mla_api.h csrc/include/sparkserve/kernel_api.h
	test -f "$(FLASHINFER_INCLUDE)/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -shared -Xcompiler=-fPIC \
		-Icsrc/include $(FLASHINFER_INCLUDES) \
		csrc/cuda/glm_sparse_mla_flashinfer.cu \
		-o $(CUDA_GLM_SPARSE_MLA_SHARED)

$(CUDA_GLM_SPARSE_MLA_FIXTURE_TEST): csrc/cuda/glm_sparse_mla_flashinfer.cu csrc/tests/glm_sparse_mla_fixture_test.cu csrc/include/sparkserve/glm_sparse_mla_api.h csrc/include/sparkserve/kernel_api.h
	test -f "$(FLASHINFER_INCLUDE)/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) \
		-Icsrc/include $(FLASHINFER_INCLUDES) \
		csrc/cuda/glm_sparse_mla_flashinfer.cu \
		csrc/tests/glm_sparse_mla_fixture_test.cu \
		-o $(CUDA_GLM_SPARSE_MLA_FIXTURE_TEST)

$(CUDA_MOE_GATE_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/moe_gate_sglang.cu csrc/internal/moe_gate_backend.h csrc/tests/moe_gate_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/moe_gate_sglang.cu \
		csrc/tests/moe_gate_fixture_test.cu -lcublas \
		-o $(CUDA_MOE_GATE_FIXTURE_TEST)

$(CUDA_MOE_GATE_SHARED): csrc/kernel_contract.cc csrc/cuda/moe_gate_sglang.cu csrc/internal/moe_gate_backend.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/moe_gate_sglang.cu -lcublas \
		-o $(CUDA_MOE_GATE_SHARED)

$(CUDA_SHARED_EXPERT_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/shared_expert_sglang.cu csrc/internal/shared_expert_backend.h csrc/tests/shared_expert_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/shared_expert_sglang.cu \
		csrc/tests/shared_expert_fixture_test.cu -lcublas \
		-o $(CUDA_SHARED_EXPERT_FIXTURE_TEST)

$(CUDA_SHARED_EXPERT_SHARED): csrc/kernel_contract.cc csrc/cuda/shared_expert_sglang.cu csrc/internal/shared_expert_backend.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/shared_expert_sglang.cu -lcublas \
		-o $(CUDA_SHARED_EXPERT_SHARED)

$(CUDA_MOE_JOIN_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/moe_join_sglang.cu csrc/internal/moe_join_backend.h csrc/tests/moe_join_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math \
		-DSPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/moe_join_sglang.cu \
		csrc/tests/moe_join_fixture_test.cu -o $(CUDA_MOE_JOIN_FIXTURE_TEST)

$(CUDA_MOE_JOIN_SHARED): csrc/kernel_contract.cc csrc/cuda/moe_join_sglang.cu csrc/internal/moe_join_backend.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/moe_join_sglang.cu \
		-o $(CUDA_MOE_JOIN_SHARED)

$(CUDA_MHC_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/mhc_sglang.cu csrc/internal/mhc_backend.h csrc/tests/mhc_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MHC -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/mhc_sglang.cu \
		csrc/tests/mhc_fixture_test.cu -lcublas -o $(CUDA_MHC_FIXTURE_TEST)

$(CUDA_MHC_SHARED): csrc/kernel_contract.cc csrc/cuda/mhc_sglang.cu csrc/internal/mhc_backend.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MHC -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/mhc_sglang.cu -lcublas \
		-o $(CUDA_MHC_SHARED)

$(CUDA_PLE_GATHER_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/ple_gather.cu csrc/fabric/coherent_region.cc csrc/internal/ple_gather_backend.h csrc/tests/ple_gather_fixture_test.cu csrc/include/sparkserve/kernel_api.h csrc/include/sparkserve/fabric_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_SGLANG_PLE_GATHER \
		-Icsrc/include -Icsrc csrc/kernel_contract.cc csrc/cuda/ple_gather.cu \
		csrc/fabric/coherent_region.cc \
		csrc/tests/ple_gather_fixture_test.cu -o $(CUDA_PLE_GATHER_FIXTURE_TEST)

$(CUDA_QSA_TOPK_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/qsa_topk_sglang.cu csrc/internal/qsa_topk_backend.h csrc/tests/qsa_topk_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_SGLANG_QSA_TOPK \
		-Icsrc/include -Icsrc csrc/kernel_contract.cc csrc/cuda/qsa_topk_sglang.cu \
		csrc/tests/qsa_topk_fixture_test.cu -o $(CUDA_QSA_TOPK_FIXTURE_TEST)

$(CUDA_QSA_EXPAND_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/qsa_expand_sglang.cu csrc/internal/qsa_expand_backend.h csrc/tests/qsa_expand_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_SGLANG_QSA_EXPAND \
		-Icsrc/include -Icsrc csrc/kernel_contract.cc csrc/cuda/qsa_expand_sglang.cu \
		csrc/tests/qsa_expand_fixture_test.cu -o $(CUDA_QSA_EXPAND_FIXTURE_TEST)

$(CUDA_QSA_SCORE_OBJECT): csrc/cuda/qsa_score_tilelang.cu csrc/internal/qsa_score_backend.h csrc/include/sparkserve/kernel_api.h third_party/tilelang-qsa-score/generated/device_kernel.cu $(wildcard third_party/tilelang-qsa-score/include/tl_templates/cuda/*.h) $(wildcard third_party/tilelang-qsa-score/include/tl_templates/cuda/instruction/*.h)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -DENABLE_BF16 -Xcompiler=-fPIC \
		-I. -Icsrc/include -Icsrc $(TILELANG_QSA_SCORE_INCLUDE) \
		-I$(CUTLASS_ROOT)/include -c csrc/cuda/qsa_score_tilelang.cu \
		-o $(CUDA_QSA_SCORE_OBJECT)

$(CUDA_QSA_SCORE_FIXTURE_TEST): $(CUDA_QSA_SCORE_OBJECT) csrc/kernel_contract.cc csrc/tests/qsa_score_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_TILELANG_QSA_SCORE \
		-Icsrc/include -Icsrc csrc/kernel_contract.cc \
		csrc/tests/qsa_score_fixture_test.cu $(CUDA_QSA_SCORE_OBJECT) \
		-o $(CUDA_QSA_SCORE_FIXTURE_TEST)

$(CUDA_QSA_INDEX_PREP_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/qsa_index_prep_sglang.cu csrc/internal/qsa_index_prep_backend.h csrc/tests/qsa_index_prep_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_SGLANG_QSA_INDEX_PREP \
		-Icsrc/include -Icsrc csrc/kernel_contract.cc csrc/cuda/qsa_index_prep_sglang.cu \
		csrc/tests/qsa_index_prep_fixture_test.cu -o $(CUDA_QSA_INDEX_PREP_FIXTURE_TEST)

$(CUDA_QSA_KV_PACK_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/qsa_kv_pack_sglang.cu csrc/internal/qsa_kv_pack_backend.h csrc/tests/qsa_kv_pack_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_SGLANG_QSA_KV_PACK \
		-Icsrc/include -Icsrc csrc/kernel_contract.cc csrc/cuda/qsa_kv_pack_sglang.cu \
		csrc/tests/qsa_kv_pack_fixture_test.cu -o $(CUDA_QSA_KV_PACK_FIXTURE_TEST)

$(CUDA_QSA_DECODE_XQA_MHA_OBJECT): $(FLASHINFER_ROOT)/csrc/xqa/mha.cu $(wildcard $(FLASHINFER_ROOT)/csrc/xqa/*.h) $(wildcard $(FLASHINFER_ROOT)/csrc/xqa/*.cuh)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) -DNDEBUG=1 -Xcompiler=-fPIC \
		$(FLASHINFER_XQA_INCLUDE) -c $(FLASHINFER_ROOT)/csrc/xqa/mha.cu \
		-o $(CUDA_QSA_DECODE_XQA_MHA_OBJECT)

$(CUDA_QSA_SHARED): $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) csrc/kernel_contract.cc csrc/cuda/qsa_index_prep_sglang.cu csrc/cuda/qsa_topk_sglang.cu csrc/cuda/qsa_expand_sglang.cu csrc/cuda/qsa_kv_pack_sglang.cu csrc/cuda/qsa_decode_xqa_flashinfer.cu csrc/internal/qsa_index_prep_backend.h csrc/internal/qsa_topk_backend.h csrc/internal/qsa_expand_backend.h csrc/internal/qsa_score_backend.h csrc/internal/qsa_kv_pack_backend.h csrc/internal/qsa_decode_backend.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_SGLANG_QSA_INDEX_PREP \
		-DSPARKSERVE_WITH_SGLANG_QSA_TOPK \
		-DSPARKSERVE_WITH_SGLANG_QSA_EXPAND \
		-DSPARKSERVE_WITH_TILELANG_QSA_SCORE \
		-DSPARKSERVE_WITH_SGLANG_QSA_KV_PACK \
		-DSPARKSERVE_WITH_FLASHINFER_XQA_DECODE -Icsrc/include -Icsrc \
		$(FLASHINFER_XQA_INCLUDE) csrc/kernel_contract.cc \
		csrc/cuda/qsa_index_prep_sglang.cu \
		csrc/cuda/qsa_topk_sglang.cu \
		csrc/cuda/qsa_expand_sglang.cu \
		csrc/cuda/qsa_kv_pack_sglang.cu \
		csrc/cuda/qsa_decode_xqa_flashinfer.cu \
		$(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) -lcuda -o $(CUDA_QSA_SHARED)

$(CUDA_QWEN_QSA_BLOCK_SHARED): csrc/cuda/qwen_qsa_block.cu csrc/include/sparkserve/qwen_qsa_block_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -Icsrc/include \
		csrc/cuda/qwen_qsa_block.cu -lcublas -lcudart \
		-o $(CUDA_QWEN_QSA_BLOCK_SHARED)

$(CUDA_QWEN_PLE_BLOCK_SHARED): csrc/kernel_contract.cc csrc/cuda/ple_gather.cu csrc/cuda/qwen_ple_block.cu csrc/internal/ple_gather_backend.h csrc/include/sparkserve/qwen_ple_block_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC \
		-DSPARKSERVE_WITH_SGLANG_PLE_GATHER -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/ple_gather.cu \
		csrc/cuda/qwen_ple_block.cu -lcublas -lcudart \
		-o $(CUDA_QWEN_PLE_BLOCK_SHARED)

$(CUDA_QWEN_DECODE_GLUE_SHARED): csrc/cuda/qwen_decode_glue.cu csrc/include/sparkserve/qwen_decode_glue_api.h csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler=-fPIC -Icsrc/include \
		csrc/cuda/qwen_decode_glue.cu -lcublas -lcudart \
		-o $(CUDA_QWEN_DECODE_GLUE_SHARED)

$(CUDA_QWEN_RUNTIME_SHARED): $(GDN_AOT_OBJECT) $(GDN_PREFILL_AOT_OBJECTS) $(CUTE_NVFP4_OBJECT) $(CUTE_NVFP4_QUANTIZE_OBJECT) $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) csrc/kernel_contract.cc csrc/qwen_runtime_direct.cc csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/cuda/gdn_block_sglang.cu csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu csrc/cuda/mhc_sglang.cu csrc/cuda/ple_gather.cu csrc/cuda/qsa_index_prep_sglang.cu csrc/cuda/qsa_topk_sglang.cu csrc/cuda/qsa_expand_sglang.cu csrc/cuda/qsa_kv_pack_sglang.cu csrc/cuda/qsa_decode_xqa_flashinfer.cu csrc/cuda/qwen_expert_pack.cu csrc/cuda/qwen_gdn_aux.cu csrc/cuda/qwen_qsa_block.cu csrc/cuda/qwen_ple_block.cu csrc/cuda/qwen_decode_glue.cu csrc/include/sparkserve/kernel_api.h csrc/include/sparkserve/qwen_runtime_api.h
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
		-DSPARKSERVE_WITH_CUDA \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_AOT \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_PREFILL_AOT \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DSPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DSPARKSERVE_WITH_FLASHINFER_MOE_ROUTE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DSPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MHC \
		-DSPARKSERVE_WITH_SGLANG_PLE_GATHER \
		-DSPARKSERVE_WITH_SGLANG_QSA_INDEX_PREP \
		-DSPARKSERVE_WITH_SGLANG_QSA_TOPK \
		-DSPARKSERVE_WITH_SGLANG_QSA_EXPAND \
		-DSPARKSERVE_WITH_TILELANG_QSA_SCORE \
		-DSPARKSERVE_WITH_SGLANG_QSA_KV_PACK \
		-DSPARKSERVE_WITH_FLASHINFER_XQA_DECODE \
		-Icsrc/include -Icsrc $(FLASHINFER_INCLUDES) $(FLASHINFER_XQA_INCLUDE) -I$(TVM_FFI_ROOT)/include \
		csrc/kernel_contract.cc csrc/qwen_runtime_direct.cc \
		csrc/cuda/gdn_decode.cu \
		csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/cuda/gdn_block_sglang.cu \
		csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/cuda/nvfp4_silu_cute.cc \
		csrc/cuda/nvfp4_quantize_cute.cc csrc/cuda/moe_route_flashinfer.cu \
		csrc/cuda/moe_gate_sglang.cu csrc/cuda/shared_expert_sglang.cu \
		csrc/cuda/moe_join_sglang.cu csrc/cuda/mhc_sglang.cu \
		csrc/cuda/ple_gather.cu csrc/cuda/qsa_index_prep_sglang.cu \
		csrc/cuda/qsa_topk_sglang.cu csrc/cuda/qsa_expand_sglang.cu \
		csrc/cuda/qsa_kv_pack_sglang.cu csrc/cuda/qsa_decode_xqa_flashinfer.cu \
		csrc/cuda/qwen_expert_pack.cu \
		csrc/cuda/qwen_gdn_aux.cu csrc/cuda/qwen_qsa_block.cu \
		csrc/cuda/qwen_ple_block.cu csrc/cuda/qwen_decode_glue.cu \
		"$(GDN_AOT_OBJECT)" "$(CUTE_NVFP4_OBJECT)" \
		$(GDN_PREFILL_AOT_OBJECTS) \
		$(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SCORE_OBJECT) \
		"$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_RUNTIME_SHARED)

$(CUDA_QSA_DECODE_XQA_FIXTURE_TEST): $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) csrc/kernel_contract.cc csrc/cuda/qsa_decode_xqa_flashinfer.cu csrc/internal/qsa_decode_backend.h csrc/tests/qsa_decode_xqa_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) \
		-DSPARKSERVE_WITH_FLASHINFER_XQA_DECODE -Icsrc/include -Icsrc \
		$(FLASHINFER_XQA_INCLUDE) csrc/kernel_contract.cc \
		csrc/cuda/qsa_decode_xqa_flashinfer.cu \
		csrc/tests/qsa_decode_xqa_fixture_test.cu \
		$(CUDA_QSA_DECODE_XQA_MHA_OBJECT) -lcuda \
		-o $(CUDA_QSA_DECODE_XQA_FIXTURE_TEST)

$(CUDA_QWEN_MOE_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu csrc/cuda/mhc_sglang.cu csrc/internal/nvfp4_grouped_backend.h csrc/internal/nvfp4_silu_backend.h csrc/internal/nvfp4_quantize_backend.h csrc/internal/moe_route_backend.h csrc/internal/moe_gate_backend.h csrc/internal/shared_expert_backend.h csrc/internal/moe_join_backend.h csrc/internal/mhc_backend.h csrc/tests/qwen_moe_fixture_test.cc csrc/include/sparkserve/kernel_api.h
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	test -f "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(FLASHINFER_ARCH_FLAGS) -diag-suppress 177 \
		-diag-suppress 549 -DSPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DSPARKSERVE_WITH_FLASHINFER_MOE_ROUTE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DSPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MHC \
		-Icsrc/include -Icsrc $(FLASHINFER_INCLUDES) -I$(TVM_FFI_ROOT)/include \
		csrc/kernel_contract.cc csrc/cuda/nvfp4_grouped_flashinfer.cu \
		csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc \
		csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu \
		csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu \
		csrc/cuda/mhc_sglang.cu \
		csrc/tests/qwen_moe_fixture_test.cc \
		"$(CUTE_NVFP4_OBJECT)" "$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_MOE_FIXTURE_TEST)

$(CUDA_QWEN_FULL_LAYER_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/cuda/gdn_block_sglang.cu csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu csrc/cuda/mhc_sglang.cu csrc/tests/qwen_gdn_block_fixture_test.cc csrc/tests/qwen_moe_fixture_test.cc csrc/tests/qwen_full_layer_fixture_test.cc csrc/include/sparkserve/kernel_api.h
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
		-diag-suppress 549 -DSPARKSERVE_FIXTURE_LIBRARY \
		-DSPARKSERVE_WITH_FLASHINFER_GDN_AOT \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_GDN_BLOCK \
		-DSPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DSPARKSERVE_WITH_FLASHINFER_MOE_ROUTE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DSPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MHC \
		-Icsrc/include -Icsrc $(FLASHINFER_INCLUDES) -I$(TVM_FFI_ROOT)/include \
		csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu \
		csrc/cuda/gdn_decode_flashinfer_cute.cc csrc/cuda/gdn_block_sglang.cu \
		csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/cuda/nvfp4_silu_cute.cc \
		csrc/cuda/nvfp4_quantize_cute.cc csrc/cuda/moe_route_flashinfer.cu \
		csrc/cuda/moe_gate_sglang.cu csrc/cuda/shared_expert_sglang.cu \
		csrc/cuda/moe_join_sglang.cu csrc/cuda/mhc_sglang.cu \
		csrc/tests/qwen_gdn_block_fixture_test.cc \
		csrc/tests/qwen_moe_fixture_test.cc \
		csrc/tests/qwen_full_layer_fixture_test.cc \
		"$(GDN_AOT_OBJECT)" "$(CUTE_NVFP4_OBJECT)" \
		"$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_FULL_LAYER_FIXTURE_TEST)

clean:
	rm -f $(CONTRACT_TEST) $(HEADER_C_TEST) $(DS4_GLM53_ADAPTER_OBJECT) $(DS4_GLM53_STATIC) $(CUDA_COHERENT_REGION_TEST) $(CUDA_PLE_GATHER_FIXTURE_TEST) $(CUDA_QSA_TOPK_FIXTURE_TEST) $(CUDA_QSA_EXPAND_FIXTURE_TEST) $(CUDA_QSA_SCORE_FIXTURE_TEST) $(CUDA_QSA_SCORE_OBJECT) $(CUDA_QSA_INDEX_PREP_FIXTURE_TEST) $(CUDA_QSA_KV_PACK_FIXTURE_TEST) $(CUDA_QSA_DECODE_XQA_FIXTURE_TEST) $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SHARED) $(CUDA_QWEN_QSA_BLOCK_SHARED) $(CUDA_FABRIC_SHARED) $(CUDA_GGML_QUANT_SHARED) $(CUDA_GGML_QUANT_FIXTURE_TEST) $(CUDA_GLM_KDA_SHARED) $(CUDA_GLM_KDA_FIXTURE_TEST) $(CUDA_GLM_KPOOL_SHARED) $(CUDA_GLM_KPOOL_FIXTURE_TEST) $(CUDA_GLM_PAGED_MQA_SHARED) $(CUDA_GLM_PAGED_MQA_FIXTURE_TEST) $(CUDA_GDN_TEST) $(CUDA_GDN_FIXTURE_TEST) $(CUDA_NVFP4_TEST) $(CUDA_NVFP4_FIXTURE_TEST) $(CUDA_GROUPED_NVFP4_TEST) $(CUDA_GROUPED_NVFP4_FIXTURE_TEST) $(CUDA_SILU_NVFP4_TEST) $(CUDA_MOE_ROUTE_TEST) $(CUDA_QWEN_MOE_FIXTURE_TEST) $(CUDA_QWEN_FULL_LAYER_FIXTURE_TEST)

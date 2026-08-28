CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Werror
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror

BUILD_DIR := build
CONTRACT_TEST := $(BUILD_DIR)/kernel-contract-test
HEADER_C_TEST := $(BUILD_DIR)/kernel-header-c-test
CUDA_GDN_TEST := $(BUILD_DIR)/gdn-decode-cuda-test
CUDA_NVFP4_TEST := $(BUILD_DIR)/nvfp4-dense-cuda-test
CUDA_NVFP4_FIXTURE_TEST := $(BUILD_DIR)/nvfp4-dense-fixture-test
CUDA_GROUPED_NVFP4_TEST := $(BUILD_DIR)/nvfp4-grouped-cuda-test
CUDA_GROUPED_NVFP4_FIXTURE_TEST := $(BUILD_DIR)/nvfp4-grouped-fixture-test
CUDA_SILU_NVFP4_TEST := $(BUILD_DIR)/nvfp4-silu-cute-test
CUDA_QWEN_MOE_FIXTURE_TEST := $(BUILD_DIR)/qwen-moe-fixture-test
CUDA_MOE_ROUTE_TEST := $(BUILD_DIR)/moe-route-cuda-test
CUDA_MOE_GATE_FIXTURE_TEST := $(BUILD_DIR)/moe-gate-fixture-test
CUDA_MOE_GATE_SHARED := $(BUILD_DIR)/libsparkserve-moe-gate.so
CUDA_SHARED_EXPERT_FIXTURE_TEST := $(BUILD_DIR)/shared-expert-fixture-test
CUDA_SHARED_EXPERT_SHARED := $(BUILD_DIR)/libsparkserve-shared-expert.so
CUDA_MOE_JOIN_FIXTURE_TEST := $(BUILD_DIR)/moe-join-fixture-test
CUDA_MOE_JOIN_SHARED := $(BUILD_DIR)/libsparkserve-moe-join.so
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
CUDA_FABRIC_SHARED := $(BUILD_DIR)/libsparkserve-fabric.so
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
QWEN_XQA_FLAGS := --expt-relaxed-constexpr \
	-DBEAM_WIDTH=1 -DUSE_INPUT_KV=0 -DUSE_CUSTOM_BARRIER=1 \
	-DTOKENS_PER_PAGE=64 -DHEAD_ELEMS=256 -DINPUT_FP16=0 \
	-DDTYPE=__nv_bfloat16 -DCACHE_ELEM_ENUM=0 -DHEAD_GRP_SIZE=12 \
	-DSLIDING_WINDOW=0 -DLOW_PREC_OUTPUT=0 -DSPEC_DEC=0 \
	-DMLA_WRAPPER=0 -DUSE_SM90_MHA=0
CUTE_NVFP4_OBJECT ?=
CUTE_NVFP4_QUANTIZE_OBJECT ?=
TVM_FFI_ROOT ?=
CUTE_DSL_ROOT ?=

.PHONY: test test-cpp test-cuda fabric-shared qsa-shared moe-gate-shared shared-expert-shared moe-join-shared test-cuda-fabric test-cuda-gdn test-cuda-nvfp4 test-cuda-nvfp4-fixture test-cuda-grouped-nvfp4 test-cuda-grouped-nvfp4-fixture test-cuda-silu-nvfp4 test-cuda-silu-nvfp4-fixture test-cuda-moe-route test-cuda-moe-gate-fixture test-cuda-shared-expert-fixture test-cuda-moe-join-fixture test-cuda-ple-gather-fixture test-cuda-qsa-topk-fixture test-cuda-qsa-expand-fixture test-cuda-qsa-score-fixture test-cuda-qsa-index-prep-fixture test-cuda-qsa-kv-pack-fixture test-cuda-qsa-decode-xqa-fixture test-cuda-qwen-moe-fixture docker-flash-next-sm121 clean

test: test-cpp

test-cpp: $(CONTRACT_TEST) $(HEADER_C_TEST)
	$(CONTRACT_TEST)
	$(HEADER_C_TEST)

test-cuda: test-cuda-fabric test-cuda-gdn test-cuda-nvfp4 test-cuda-grouped-nvfp4 test-cuda-moe-route

test-cuda-fabric: $(CUDA_COHERENT_REGION_TEST)
	$(CUDA_COHERENT_REGION_TEST)

fabric-shared: $(CUDA_FABRIC_SHARED)

qsa-shared: $(CUDA_QSA_SHARED)

moe-gate-shared: $(CUDA_MOE_GATE_SHARED)

shared-expert-shared: $(CUDA_SHARED_EXPERT_SHARED)

moe-join-shared: $(CUDA_MOE_JOIN_SHARED)

test-cuda-gdn: $(CUDA_GDN_TEST)
	$(CUDA_GDN_TEST)

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

docker-flash-next-sm121:
	docker build -t sparkserve/sglang:qwen38flashnext-sm121 -f docker/flashnext-sm121/Dockerfile .

$(CONTRACT_TEST): csrc/kernel_contract.cc csrc/tests/kernel_contract_test.cc csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Icsrc/include csrc/kernel_contract.cc csrc/tests/kernel_contract_test.cc -o $(CONTRACT_TEST)

$(HEADER_C_TEST): csrc/tests/kernel_header_c_test.c csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -Icsrc/include csrc/tests/kernel_header_c_test.c -o $(HEADER_C_TEST)

$(CUDA_GDN_TEST): csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu csrc/internal/gdn_decode_backend.h csrc/tests/gdn_decode_cuda_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DSPARKSERVE_WITH_CUDA -Icsrc/include -Icsrc \
		csrc/kernel_contract.cc csrc/cuda/gdn_decode.cu \
		csrc/tests/gdn_decode_cuda_test.cu -o $(CUDA_GDN_TEST)

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

$(CUDA_QSA_DECODE_XQA_FIXTURE_TEST): $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) csrc/kernel_contract.cc csrc/cuda/qsa_decode_xqa_flashinfer.cu csrc/internal/qsa_decode_backend.h csrc/tests/qsa_decode_xqa_fixture_test.cu csrc/include/sparkserve/kernel_api.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(QWEN_XQA_FLAGS) \
		-DSPARKSERVE_WITH_FLASHINFER_XQA_DECODE -Icsrc/include -Icsrc \
		$(FLASHINFER_XQA_INCLUDE) csrc/kernel_contract.cc \
		csrc/cuda/qsa_decode_xqa_flashinfer.cu \
		csrc/tests/qsa_decode_xqa_fixture_test.cu \
		$(CUDA_QSA_DECODE_XQA_MHA_OBJECT) -lcuda \
		-o $(CUDA_QSA_DECODE_XQA_FIXTURE_TEST)

$(CUDA_QWEN_MOE_FIXTURE_TEST): csrc/kernel_contract.cc csrc/cuda/nvfp4_grouped_flashinfer.cu csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu csrc/internal/nvfp4_grouped_backend.h csrc/internal/nvfp4_silu_backend.h csrc/internal/nvfp4_quantize_backend.h csrc/internal/moe_route_backend.h csrc/internal/moe_gate_backend.h csrc/internal/shared_expert_backend.h csrc/internal/moe_join_backend.h csrc/tests/qwen_moe_fixture_test.cc csrc/include/sparkserve/kernel_api.h
	test -n "$(CUTE_NVFP4_OBJECT)"
	test -n "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	test -n "$(TVM_FFI_ROOT)"
	test -n "$(CUTE_DSL_ROOT)"
	test -f "$(CUTE_NVFP4_OBJECT)"
	test -f "$(CUTE_NVFP4_QUANTIZE_OBJECT)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASHINFER_ARCH_FLAGS) -diag-suppress 177 \
		-diag-suppress 549 -DSPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4 \
		-DSPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE \
		-DSPARKSERVE_WITH_FLASHINFER_MOE_ROUTE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE \
		-DSPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT \
		-DSPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN \
		-Icsrc/include -Icsrc $(FLASHINFER_INCLUDES) -I$(TVM_FFI_ROOT)/include \
		csrc/kernel_contract.cc csrc/cuda/nvfp4_grouped_flashinfer.cu \
		csrc/cuda/nvfp4_silu_cute.cc csrc/cuda/nvfp4_quantize_cute.cc \
		csrc/cuda/moe_route_flashinfer.cu csrc/cuda/moe_gate_sglang.cu \
		csrc/cuda/shared_expert_sglang.cu csrc/cuda/moe_join_sglang.cu \
		csrc/tests/qwen_moe_fixture_test.cc \
		"$(CUTE_NVFP4_OBJECT)" "$(CUTE_NVFP4_QUANTIZE_OBJECT)" \
		$(CUTE_DSL_ROOT)/lib/libcuda_dialect_runtime_static.a \
		-L$(TVM_FFI_ROOT)/lib -ltvm_ffi -lcublas -lcuda -lcudart -ldl \
		-Xcompiler=-pthread -o $(CUDA_QWEN_MOE_FIXTURE_TEST)

clean:
	rm -f $(CONTRACT_TEST) $(HEADER_C_TEST) $(CUDA_COHERENT_REGION_TEST) $(CUDA_PLE_GATHER_FIXTURE_TEST) $(CUDA_QSA_TOPK_FIXTURE_TEST) $(CUDA_QSA_EXPAND_FIXTURE_TEST) $(CUDA_QSA_SCORE_FIXTURE_TEST) $(CUDA_QSA_SCORE_OBJECT) $(CUDA_QSA_INDEX_PREP_FIXTURE_TEST) $(CUDA_QSA_KV_PACK_FIXTURE_TEST) $(CUDA_QSA_DECODE_XQA_FIXTURE_TEST) $(CUDA_QSA_DECODE_XQA_MHA_OBJECT) $(CUDA_QSA_SHARED) $(CUDA_FABRIC_SHARED) $(CUDA_GDN_TEST) $(CUDA_NVFP4_TEST) $(CUDA_NVFP4_FIXTURE_TEST) $(CUDA_GROUPED_NVFP4_TEST) $(CUDA_GROUPED_NVFP4_FIXTURE_TEST) $(CUDA_SILU_NVFP4_TEST) $(CUDA_MOE_ROUTE_TEST) $(CUDA_QWEN_MOE_FIXTURE_TEST)

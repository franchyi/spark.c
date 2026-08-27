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

.PHONY: test test-cpp test-cuda test-cuda-gdn test-cuda-nvfp4 test-cuda-nvfp4-fixture test-cuda-grouped-nvfp4 test-cuda-grouped-nvfp4-fixture docker-flash-next-sm121 clean

test: test-cpp

test-cpp: $(CONTRACT_TEST) $(HEADER_C_TEST)
	$(CONTRACT_TEST)
	$(HEADER_C_TEST)

test-cuda: test-cuda-gdn test-cuda-nvfp4 test-cuda-grouped-nvfp4

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

clean:
	rm -f $(CONTRACT_TEST) $(HEADER_C_TEST) $(CUDA_GDN_TEST) $(CUDA_NVFP4_TEST) $(CUDA_NVFP4_FIXTURE_TEST) $(CUDA_GROUPED_NVFP4_TEST) $(CUDA_GROUPED_NVFP4_FIXTURE_TEST)

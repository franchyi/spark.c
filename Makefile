CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Werror
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror

BUILD_DIR := build
CONTRACT_TEST := $(BUILD_DIR)/kernel-contract-test
HEADER_C_TEST := $(BUILD_DIR)/kernel-header-c-test
CUDA_GDN_TEST := $(BUILD_DIR)/gdn-decode-cuda-test
NVCC ?= nvcc
CUDA_ARCH ?= sm_121a
NVCCFLAGS ?= -O2 -std=c++20 -arch=$(CUDA_ARCH)

.PHONY: test test-cpp test-cuda test-cuda-gdn docker-flash-next-sm121 clean

test: test-cpp

test-cpp: $(CONTRACT_TEST) $(HEADER_C_TEST)
	$(CONTRACT_TEST)
	$(HEADER_C_TEST)

test-cuda: test-cuda-gdn

test-cuda-gdn: $(CUDA_GDN_TEST)
	$(CUDA_GDN_TEST)

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

clean:
	rm -f $(CONTRACT_TEST) $(HEADER_C_TEST) $(CUDA_GDN_TEST)

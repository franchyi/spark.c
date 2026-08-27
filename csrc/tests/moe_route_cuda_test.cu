#include "sparkserve/kernel_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

template <class T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CudaOk(cudaMalloc(&device, host.size() * sizeof(T)));
  CudaOk(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
  return device;
}

template <class T>
T* Allocate(size_t count) {
  T* device = nullptr;
  CudaOk(cudaMalloc(&device, count * sizeof(T)));
  return device;
}

}  // namespace

int main() {
  constexpr uint32_t kTokens = 2;
  constexpr uint32_t kTopK = 2;
  constexpr uint64_t kHidden = 16;
  constexpr uint64_t kRows = 8;
  std::vector<__nv_bfloat16> token_input(kTokens * kHidden);
  for (uint32_t token = 0; token < kTokens; ++token) {
    for (uint64_t column = 0; column < kHidden; ++column) {
      token_input[token * kHidden + column] =
          __float2bfloat16(static_cast<float>(token * 100 + column));
    }
  }
  const std::vector<uint32_t> route_map = {4, 0, 1, 5};
  auto* token_input_device = Upload(token_input);
  auto* route_map_device = Upload(route_map);
  auto* packed_device = Allocate<__nv_bfloat16>(kRows * kHidden);

  SparkServeMoeRoutePlan plan = {
      sizeof(SparkServeMoeRoutePlan), SPARKSERVE_KERNEL_ABI_VERSION,
      kTokens,                         kTopK,
      4,                               0,
      kHidden,                         kRows,
  };
  SparkServeMoeRouteArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.token_input = token_input_device;
  args.route_to_packed_row = route_map_device;
  args.packed_input = packed_device;
  args.token_input_row_stride_bytes = kHidden * 2;
  args.packed_row_stride_bytes = kHidden * 2;
  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeStatus status = sparkserve_moe_route_dispatch(&caps, &args);
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> packed(kRows * kHidden);
  CudaOk(cudaMemcpy(packed.data(), packed_device, packed.size() * sizeof(packed[0]),
                    cudaMemcpyDeviceToHost));
  for (uint32_t route = 0; route < route_map.size(); ++route) {
    const uint32_t token = route / kTopK;
    const uint32_t row = route_map[route];
    for (uint64_t column = 0; column < kHidden; ++column) {
      assert(__bfloat162float(packed[row * kHidden + column]) ==
             __bfloat162float(token_input[token * kHidden + column]));
    }
  }
  for (uint32_t row : {2u, 3u, 6u, 7u}) {
    for (uint64_t column = 0; column < kHidden; ++column) {
      assert(__bfloat162float(packed[row * kHidden + column]) == 0.0f);
    }
  }

  std::vector<__nv_bfloat16> expert_output(kRows * kHidden);
  for (uint32_t route = 0; route < route_map.size(); ++route) {
    const uint32_t row = route_map[route];
    for (uint64_t column = 0; column < kHidden; ++column) {
      expert_output[row * kHidden + column] =
          __float2bfloat16(static_cast<float>((route + 1) * 4 + column));
    }
  }
  const std::vector<float> weights = {0.25f, 0.75f, 0.5f, 0.5f};
  auto* expert_output_device = Upload(expert_output);
  auto* weights_device = Upload(weights);
  auto* token_output_device = Allocate<__nv_bfloat16>(kTokens * kHidden);
  args.route_weights = weights_device;
  args.packed_expert_output = expert_output_device;
  args.token_output = token_output_device;
  args.expert_output_row_stride_bytes = kHidden * 2;
  status = sparkserve_moe_route_finalize(&caps, &args);
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> token_output(kTokens * kHidden);
  CudaOk(cudaMemcpy(token_output.data(), token_output_device,
                    token_output.size() * sizeof(token_output[0]),
                    cudaMemcpyDeviceToHost));
  for (uint32_t token = 0; token < kTokens; ++token) {
    for (uint64_t column = 0; column < kHidden; ++column) {
      float expected = 0.0f;
      for (uint32_t rank = 0; rank < kTopK; ++rank) {
        const uint32_t route = token * kTopK + rank;
        expected += weights[route] * static_cast<float>((route + 1) * 4 + column);
      }
      assert(__bfloat162float(token_output[token * kHidden + column]) == expected);
    }
  }

  CudaOk(cudaFree(token_output_device));
  CudaOk(cudaFree(weights_device));
  CudaOk(cudaFree(expert_output_device));
  CudaOk(cudaFree(packed_device));
  CudaOk(cudaFree(route_map_device));
  CudaOk(cudaFree(token_input_device));
  std::cout << "FlashInfer-derived MoE dispatch/finalize passed\n";
  return 0;
}

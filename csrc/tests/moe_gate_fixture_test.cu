#include "sparkserve/kernel_api.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

constexpr uint32_t kTokens = 8;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kExperts = 512;
constexpr uint32_t kTopK = 10;

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }
void CublasOk(cublasStatus_t status) { assert(status == CUBLAS_STATUS_SUCCESS); }

template <typename T>
std::vector<T> Read(const std::filesystem::path& path, size_t elements) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  assert(stream.good());
  assert(static_cast<size_t>(stream.tellg()) == elements * sizeof(T));
  stream.seekg(0);
  std::vector<T> data(elements);
  stream.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
  assert(stream.good());
  return data;
}

template <typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CudaOk(cudaMalloc(&device, host.size() * sizeof(T)));
  CudaOk(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
  return device;
}

float Bf16ToFloat(uint16_t bits) {
  union {
    uint32_t bits;
    float value;
  } converted = {static_cast<uint32_t>(bits) << 16};
  return converted.value;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  const auto hidden = Read<uint16_t>(fixture / "hidden_bf16.bin",
                                     kTokens * kHidden);
  const auto weight = Read<uint16_t>(fixture / "router_weight_bf16.bin",
                                     kExperts * kHidden);
  const auto expected_logits = Read<uint16_t>(fixture / "logits_bf16.bin",
                                              kTokens * kExperts);
  const auto expected_weights = Read<float>(fixture / "topk_weights_f32.bin",
                                            kTokens * kTopK);
  const auto expected_ids = Read<int32_t>(fixture / "topk_ids_i32.bin",
                                          kTokens * kTopK);

  uint16_t* hidden_device = Upload(hidden);
  uint16_t* weight_device = Upload(weight);
  uint16_t* logits_device = nullptr;
  float* topk_weights_device = nullptr;
  int32_t* topk_ids_device = nullptr;
  CudaOk(cudaMalloc(&logits_device, expected_logits.size() * sizeof(uint16_t)));
  CudaOk(cudaMalloc(&topk_weights_device,
                    expected_weights.size() * sizeof(float)));
  CudaOk(cudaMalloc(&topk_ids_device, expected_ids.size() * sizeof(int32_t)));

  cublasHandle_t blas = nullptr;
  CublasOk(cublasCreate(&blas));
  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeMoeGatePlan plan = {
      sizeof(SparkServeMoeGatePlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kTokens,
      kHidden,
      kExperts,
      kTopK,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_SGLANG_CUBLAS_MOE_GATE,
      1,
      0,
  };
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_moe_gate_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.available == 1);

  SparkServeMoeGateArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.hidden_states = hidden_device;
  args.router_weight = weight_device;
  args.router_logits = logits_device;
  args.topk_weights = topk_weights_device;
  args.topk_ids = topk_ids_device;
  args.cublas_handle = blas;

  SparkServeStatus status = sparkserve_moe_gate_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint16_t> actual_logits(expected_logits.size());
  std::vector<float> actual_weights(expected_weights.size());
  std::vector<int32_t> actual_ids(expected_ids.size());
  CudaOk(cudaMemcpy(actual_logits.data(), logits_device,
                    actual_logits.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_weights.data(), topk_weights_device,
                    actual_weights.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_ids.data(), topk_ids_device,
                    actual_ids.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost));

  float max_logit_error = 0.0F;
  for (size_t index = 0; index < actual_logits.size(); ++index) {
    max_logit_error =
        std::max(max_logit_error,
                 std::abs(Bf16ToFloat(actual_logits[index]) -
                          Bf16ToFloat(expected_logits[index])));
  }
  float max_weight_error = 0.0F;
  for (size_t index = 0; index < actual_weights.size(); ++index) {
    max_weight_error =
        std::max(max_weight_error,
                 std::abs(actual_weights[index] - expected_weights[index]));
  }
  const size_t id_mismatches =
      std::inner_product(actual_ids.begin(), actual_ids.end(),
                         expected_ids.begin(), size_t{0}, std::plus<size_t>(),
                         std::not_equal_to<int32_t>());
  std::cout << "Qwen router max BF16 logit error: " << max_logit_error << '\n';
  std::cout << "SGLang top-k id mismatches: " << id_mismatches << '\n';
  std::cout << "SGLang top-k max weight error: " << max_weight_error << '\n';
  assert(max_logit_error <= 0.125F);
  assert(id_mismatches == 0);
  assert(max_weight_error <= 1.0e-5F);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 100;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = sparkserve_moe_gate_launch(&caps, &args);
    assert(status.code == SPARKSERVE_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "Qwen BF16 router + normalized top-10 mean: "
            << elapsed_ms * 1000.0F / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CublasOk(cublasDestroy(blas));
  CudaOk(cudaFree(topk_ids_device));
  CudaOk(cudaFree(topk_weights_device));
  CudaOk(cudaFree(logits_device));
  CudaOk(cudaFree(weight_device));
  CudaOk(cudaFree(hidden_device));
  return 0;
}

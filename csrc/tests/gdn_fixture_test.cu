#include "sparkserve/kernel_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t kQkHeads = 16;
constexpr uint32_t kValueHeads = 48;
constexpr uint32_t kHeadDim = 128;
constexpr uint64_t kQkWidth = kQkHeads * kHeadDim;
constexpr uint64_t kValueWidth = kValueHeads * kHeadDim;
constexpr uint64_t kConvWidth = 2 * kQkWidth + kValueWidth;

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

std::vector<uint8_t> Read(const std::filesystem::path& path, size_t bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  assert(stream.good());
  assert(static_cast<size_t>(stream.tellg()) == bytes);
  stream.seekg(0);
  std::vector<uint8_t> data(bytes);
  stream.read(reinterpret_cast<char*>(data.data()), data.size());
  assert(stream.good());
  return data;
}

void* Upload(const std::vector<uint8_t>& host) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, host.size()));
  CudaOk(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
  return device;
}

void* Allocate(size_t bytes) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, bytes));
  CudaOk(cudaMemset(device, 0, bytes));
  return device;
}

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void Report(const void* device, const std::vector<uint8_t>& expected,
            const char* stage) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost));
  size_t mismatched_bytes = 0;
  float max_error = 0.0F;
  const auto* actual_bf16 = reinterpret_cast<const uint16_t*>(actual.data());
  const auto* expected_bf16 =
      reinterpret_cast<const uint16_t*>(expected.data());
  for (size_t index = 0; index < expected.size(); ++index)
    mismatched_bytes += actual[index] != expected[index];
  for (size_t index = 0; index < expected.size() / 2; ++index) {
    max_error = std::max(
        max_error, std::abs(Bf16ToFloat(actual_bf16[index]) -
                            Bf16ToFloat(expected_bf16[index])));
  }
  std::cout << stage << " mismatched bytes: " << mismatched_bytes
            << ", max BF16 absolute error: " << max_error << '\n';
  assert(mismatched_bytes == 0);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  const auto qkv = Read(fixture / "convolved_qkv_bf16.bin", kConvWidth * 2);
  const auto a = Read(fixture / "projected_a_bf16.bin", kValueHeads * 2);
  const auto b = Read(fixture / "projected_b_bf16.bin", kValueHeads * 2);
  const auto a_log = Read(fixture / "a_log_f32.bin", kValueHeads * 4);
  const auto dt_bias = Read(fixture / "dt_bias_f32.bin", kValueHeads * 4);
  const auto state_before =
      Read(fixture / "temporal_state_before_bf16.bin",
           kValueHeads * kHeadDim * kHeadDim * 2);
  const auto state_expected =
      Read(fixture / "temporal_state_after_bf16.bin", state_before.size());
  const auto output_expected =
      Read(fixture / "gdn_core_output_bf16.bin", kValueWidth * 2);
  const auto state_indices =
      Read(fixture / "state_indices_i32.bin", sizeof(int32_t));

  void* qkv_device = Upload(qkv);
  void* a_device = Upload(a);
  void* b_device = Upload(b);
  void* a_log_device = Upload(a_log);
  void* dt_bias_device = Upload(dt_bias);
  void* state_device = Upload(state_before);
  void* state_seed_device = Upload(state_before);
  void* output_device = Allocate(output_expected.size());
  void* state_indices_device = Upload(state_indices);

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeGdnDecodePlan plan = {
      sizeof(SparkServeGdnDecodePlan), SPARKSERVE_KERNEL_ABI_VERSION,
      1,                                kQkHeads,
      kValueHeads,                      kHeadDim,
      kHeadDim,                         1,
      SPARKSERVE_DTYPE_BF16,            SPARKSERVE_GDN_BACKEND_FLASHINFER};
  auto* qkv_bytes = static_cast<uint8_t*>(qkv_device);
  SparkServeGdnDecodeArgs args = {
      sizeof(SparkServeGdnDecodeArgs), SPARKSERVE_KERNEL_ABI_VERSION,
      plan,                             qkv_bytes,
      qkv_bytes + kQkWidth * 2,         qkv_bytes + 2 * kQkWidth * 2,
      a_device,                         b_device,
      static_cast<const float*>(a_log_device),
      static_cast<const float*>(dt_bias_device),
      state_device,
      static_cast<const int32_t*>(state_indices_device),
      output_device,
      1.0F / std::sqrt(static_cast<float>(kHeadDim)),
      0,
      nullptr};
  SparkServeStatus status = sparkserve_gdn_decode_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  Report(state_device, state_expected, "real GDN temporal state");
  Report(output_device, output_expected, "real GDN core output");

  constexpr int kBenchmarkIterations = 30;
  float elapsed_ms = 0.0F;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CudaOk(cudaEventCreate(&start));
  CudaOk(cudaEventCreate(&stop));
  for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
    CudaOk(cudaMemcpyAsync(state_device, state_seed_device, state_before.size(),
                           cudaMemcpyDeviceToDevice));
    CudaOk(cudaEventRecord(start));
    status = sparkserve_gdn_decode_launch(&caps, &args);
    assert(status.code == SPARKSERVE_STATUS_OK);
    CudaOk(cudaEventRecord(stop));
    CudaOk(cudaEventSynchronize(stop));
    float iteration_ms = 0.0F;
    CudaOk(cudaEventElapsedTime(&iteration_ms, start, stop));
    elapsed_ms += iteration_ms;
  }
  std::cout << "borrowed FlashInfer GDN decode: "
            << elapsed_ms * 1000.0F / kBenchmarkIterations << " us/token\n";
  CudaOk(cudaEventDestroy(stop));
  CudaOk(cudaEventDestroy(start));

  CudaOk(cudaFree(state_indices_device));
  CudaOk(cudaFree(output_device));
  CudaOk(cudaFree(state_seed_device));
  CudaOk(cudaFree(state_device));
  CudaOk(cudaFree(dt_bias_device));
  CudaOk(cudaFree(a_log_device));
  CudaOk(cudaFree(b_device));
  CudaOk(cudaFree(a_device));
  CudaOk(cudaFree(qkv_device));
  return 0;
}

#include "flash/kernel_api.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t kTokens = 1;
constexpr uint32_t kHc = 4;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kWidth = kHc * kHidden;
constexpr uint32_t kLowrank = 320;

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }
void CublasOk(cublasStatus_t status) { assert(status == CUBLAS_STATUS_SUCCESS); }

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

size_t ExpectBytes(const void* device, const std::vector<uint8_t>& expected,
                   const char* stage) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != expected[index];
  }
  std::cout << stage << " mismatched bytes: " << mismatches << '\n';
  return mismatches;
}

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void ExpectTolerance(const void* device, const std::vector<uint8_t>& expected,
                     float tolerance, const char* stage) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost));
  const auto* actual_values = reinterpret_cast<const uint16_t*>(actual.data());
  const auto* expected_values =
      reinterpret_cast<const uint16_t*>(expected.data());
  float max_error = 0.0F;
  for (size_t index = 0; index < expected.size() / 2; ++index) {
    max_error = fmaxf(max_error,
                      fabsf(Bf16ToFloat(actual_values[index]) -
                            Bf16ToFloat(expected_values[index])));
  }
  std::cout << stage << " max BF16 absolute error: " << max_error << '\n';
  assert(max_error <= tolerance);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  const auto hyper_input =
      Read(fixture / "hyper_input_bf16.bin", kTokens * kWidth * 2);
  const auto block_output =
      Read(fixture / "block_output_bf16.bin", kTokens * kHidden * 2);
  const auto norm_weight = Read(fixture / "norm_weight_bf16.bin", kWidth * 2);
  const auto down_weight =
      Read(fixture / "down_weight_bf16.bin", kLowrank * kWidth * 2);
  const auto up_weight =
      Read(fixture / "up_weight_bf16.bin", kWidth * kLowrank * 2);
  const auto inject_weight =
      Read(fixture / "inject_weight_bf16.bin", kHc * kWidth * 2);
  const auto expected_normed =
      Read(fixture / "normed_bf16.bin", kTokens * kWidth * 2);
  const auto expected_down =
      Read(fixture / "down_bf16.bin", kTokens * kLowrank * 2);
  const auto expected_activated =
      Read(fixture / "activated_bf16.bin", kTokens * kLowrank * 2);
  const auto expected_up =
      Read(fixture / "up_bf16.bin", kTokens * kWidth * 2);
  const auto expected_reference_mix =
      Read(fixture / "reference_mix_bf16.bin", kTokens * kHidden * 2);
  const auto expected_fused_mix =
      Read(fixture / "fused_mix_first_bf16.bin", kTokens * kHidden * 2);
  const auto expected_combined =
      Read(fixture / "combined_bf16.bin", kTokens * kWidth * 2);

  void* hyper_input_device = Upload(hyper_input);
  void* block_output_device = Upload(block_output);
  void* norm_weight_device = Upload(norm_weight);
  void* down_weight_device = Upload(down_weight);
  void* up_weight_device = Upload(up_weight);
  void* inject_weight_device = Upload(inject_weight);
  void* normed_device = Allocate(expected_normed.size());
  void* down_device = Allocate(expected_down.size());
  void* activated_device = Allocate(expected_activated.size());
  void* up_device = Allocate(expected_up.size());
  void* mixed_device = Allocate(expected_reference_mix.size());
  void* combined_device = Allocate(expected_combined.size());
  cublasHandle_t blas = nullptr;
  CublasOk(cublasCreate(&blas));

  FlashDeviceCaps caps = {
      sizeof(FlashDeviceCaps), FLASH_KERNEL_ABI_VERSION, 121, 1, 0};
  FlashMhcArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = {sizeof(FlashMhcPlan),
               FLASH_KERNEL_ABI_VERSION,
               kTokens,
               kHc,
               kHidden,
               kLowrank,
               FLASH_DTYPE_BF16,
               FLASH_BACKEND_SGLANG_CUBLAS_MHC,
               1.0e-6F,
               0,
               0,
               0};
  args.hyper_input = hyper_input_device;
  args.norm_weight = norm_weight_device;
  args.mix_down_weight = down_weight_device;
  args.mix_up_weight = up_weight_device;
  args.inject_weight = inject_weight_device;
  args.block_output = block_output_device;
  args.normed = normed_device;
  args.mix_down = down_device;
  args.mix_activated = activated_device;
  args.mix_up = up_device;
  args.mixed_output = mixed_device;
  args.combined_output = combined_device;
  args.cublas_handle = blas;

  FlashStatus status = flash_mhc_mix_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  assert(ExpectBytes(normed_device, expected_normed, "mHC grouped RMSNorm") == 0);
  assert(ExpectBytes(down_device, expected_down, "mHC down projection") == 0);
  assert(ExpectBytes(activated_device, expected_activated, "mHC scaled SiLU") ==
         0);
  assert(ExpectBytes(up_device, expected_up, "mHC up projection") == 0);
  assert(ExpectBytes(mixed_device, expected_reference_mix,
                     "mHC deterministic mix") == 0);
  ExpectTolerance(mixed_device, expected_fused_mix, 0.015625F,
                  "mHC deployed persistent mix");

  status = flash_mhc_combine_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  assert(ExpectBytes(combined_device, expected_combined, "mHC combine") == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 100;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    assert(flash_mhc_mix_launch(&caps, &args).code ==
           FLASH_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float mix_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&mix_ms, begin, end));
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    assert(flash_mhc_combine_launch(&caps, &args).code ==
           FLASH_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float combine_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&combine_ms, begin, end));
  std::cout << "mHC deterministic mix mean: "
            << mix_ms * 1000.0F / kIterations << " us\n";
  std::cout << "mHC combine mean: "
            << combine_ms * 1000.0F / kIterations << " us\n";
  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));

  CublasOk(cublasDestroy(blas));
  CudaOk(cudaFree(combined_device));
  CudaOk(cudaFree(mixed_device));
  CudaOk(cudaFree(up_device));
  CudaOk(cudaFree(activated_device));
  CudaOk(cudaFree(down_device));
  CudaOk(cudaFree(normed_device));
  CudaOk(cudaFree(inject_weight_device));
  CudaOk(cudaFree(up_weight_device));
  CudaOk(cudaFree(down_weight_device));
  CudaOk(cudaFree(norm_weight_device));
  CudaOk(cudaFree(block_output_device));
  CudaOk(cudaFree(hyper_input_device));
  return 0;
}

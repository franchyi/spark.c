#include "flash/kernel_api.h"

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
constexpr uint32_t kHidden = 2560;

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

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void ExpectExact(const void* device, const std::vector<uint8_t>& expected) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  float max_abs_error = 0.0F;
  const auto* actual_bf16 = reinterpret_cast<const uint16_t*>(actual.data());
  const auto* expected_bf16 =
      reinterpret_cast<const uint16_t*>(expected.data());
  for (size_t index = 0; index < expected.size() / 2; ++index) {
    mismatches += actual_bf16[index] != expected_bf16[index];
    max_abs_error = fmaxf(
        max_abs_error,
        fabsf(Bf16ToFloat(actual_bf16[index]) -
              Bf16ToFloat(expected_bf16[index])));
  }
  std::cout << "joined MoE mismatched BF16 values: " << mismatches << '\n';
  std::cout << "joined MoE max BF16 absolute error: " << max_abs_error << '\n';
  assert(mismatches == 0);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  const auto hidden = Read(fixture / "hidden_bf16.bin", kTokens * kHidden * 2);
  const auto gate_weight =
      Read(fixture / "shared_gate_weight_bf16.bin", kHidden * 2);
  const auto shared =
      Read(fixture / "shared_ungated_bf16.bin", kTokens * kHidden * 2);
  const auto routed =
      Read(fixture / "routed_output_bf16.bin", kTokens * kHidden * 2);
  const auto expected =
      Read(fixture / "joined_output_bf16.bin", kTokens * kHidden * 2);

  void* hidden_device = Upload(hidden);
  void* gate_weight_device = Upload(gate_weight);
  void* shared_device = Upload(shared);
  void* routed_device = Upload(routed);

  FlashDeviceCaps caps = {
      sizeof(FlashDeviceCaps), FLASH_KERNEL_ABI_VERSION, 121, 1, 0};
  FlashMoeJoinArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = {sizeof(FlashMoeJoinPlan),
               FLASH_KERNEL_ABI_VERSION,
               kTokens,
               kHidden,
               FLASH_DTYPE_BF16,
               FLASH_DTYPE_BF16,
               FLASH_BACKEND_SGLANG_FUSED_MOE_JOIN,
               0};
  args.hidden_states = hidden_device;
  args.shared_gate_weight = gate_weight_device;
  args.shared_output = shared_device;
  args.routed_output = routed_device;
  FlashStatus status = flash_moe_join_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectExact(routed_device, expected);

  CudaOk(cudaFree(routed_device));
  CudaOk(cudaFree(shared_device));
  CudaOk(cudaFree(gate_weight_device));
  CudaOk(cudaFree(hidden_device));
  return 0;
}

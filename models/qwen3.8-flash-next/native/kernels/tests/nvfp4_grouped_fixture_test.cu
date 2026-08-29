#include "flash/kernel_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

std::vector<uint8_t> Read(const std::filesystem::path& path,
                          size_t expected_bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  assert(stream.good());
  assert(static_cast<size_t>(stream.tellg()) == expected_bytes);
  stream.seekg(0);
  std::vector<uint8_t> data(expected_bytes);
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

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  constexpr uint32_t kGroups = 2;
  constexpr uint64_t kRows = 8;
  constexpr uint64_t kScaleRows = 256;
  constexpr uint64_t kN = 640;
  constexpr uint64_t kK = 2560;
  constexpr size_t kWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;
  constexpr size_t kOutputBytes = kRows * kN * sizeof(__nv_bfloat16);

  auto input_host = Read(fixture / "input_fp4.bin", kRows * kK / 2);
  auto input_scales_host =
      Read(fixture / "input_scales.bin", kScaleRows * kK / 16);
  auto weight_host =
      Read(fixture / "weight_fp4.bin", kGroups * kN * kK / 2);
  auto weight_scales_host =
      Read(fixture / "weight_scales.bin", kGroups * kN * kK / 16);
  auto alpha_host = Read(fixture / "alpha_f32.bin", kGroups * sizeof(float));
  auto indptr_host =
      Read(fixture / "m_indptr_i32.bin", (kGroups + 1) * sizeof(int32_t));
  auto expected = Read(fixture / "output_bf16.bin", kOutputBytes);

  void* input = Upload(input_host);
  void* input_scales = Upload(input_scales_host);
  void* weights = Upload(weight_host);
  void* weight_scales = Upload(weight_scales_host);
  void* alpha = Upload(alpha_host);
  void* indptr = Upload(indptr_host);
  void* output = nullptr;
  void* int_workspace = nullptr;
  void* float_workspace = nullptr;
  CudaOk(cudaMalloc(&output, kOutputBytes));
  CudaOk(cudaMalloc(&int_workspace, kWorkspaceBytes));
  CudaOk(cudaMalloc(&float_workspace, kWorkspaceBytes));

  FlashGroupedNvfp4Plan plan = {
      sizeof(FlashGroupedNvfp4Plan),
      FLASH_KERNEL_ABI_VERSION,
      kGroups,
      16,
      kRows,
      kScaleRows,
      kN,
      kK,
      128,
      128,
      256,
      0,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_FLASHINFER_GROUP_MM_FP4,
  };
  FlashDeviceCaps caps = {
      sizeof(FlashDeviceCaps), FLASH_KERNEL_ABI_VERSION, 121, 1, 0};
  FlashGroupedNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.input = {input, input_scales, kK / 2, kK / 16};
  args.weights = {weights, weight_scales, kN * kK / 2, kN * kK / 16};
  args.m_indptr = static_cast<const int32_t*>(indptr);
  args.alpha_device = static_cast<const float*>(alpha);
  args.output = output;
  args.output_row_stride_bytes = kN * sizeof(__nv_bfloat16);
  args.int_workspace = int_workspace;
  args.int_workspace_bytes = kWorkspaceBytes;
  args.float_workspace = float_workspace;
  args.float_workspace_bytes = kWorkspaceBytes;
  FlashStatus status = flash_grouped_nvfp4_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> actual(kOutputBytes);
  CudaOk(cudaMemcpy(actual.data(), output, kOutputBytes, cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); index += sizeof(uint16_t)) {
    uint16_t got;
    uint16_t want;
    std::memcpy(&got, &actual[index], sizeof(got));
    std::memcpy(&want, &expected[index], sizeof(want));
    mismatches += got != want;
  }
  std::cout << "real Qwen grouped NVFP4 BF16 mismatches: " << mismatches << '\n';
  assert(mismatches == 0);

  CudaOk(cudaFree(float_workspace));
  CudaOk(cudaFree(int_workspace));
  CudaOk(cudaFree(output));
  CudaOk(cudaFree(indptr));
  CudaOk(cudaFree(alpha));
  CudaOk(cudaFree(weight_scales));
  CudaOk(cudaFree(weights));
  CudaOk(cudaFree(input_scales));
  CudaOk(cudaFree(input));
  return 0;
}

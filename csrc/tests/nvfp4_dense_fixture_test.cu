#include "sparkserve/kernel_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
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
  constexpr uint64_t kM = 1;
  constexpr uint64_t kN = 640;
  constexpr uint64_t kK = 2560;
  constexpr size_t kInputBytes = kM * kK / 2;
  constexpr size_t kWeightBytes = kN * kK / 2;
  constexpr size_t kInputScaleBytes = 20'480;
  constexpr size_t kWeightScaleBytes = kN * (kK / 16);
  constexpr size_t kOutputBytes = kM * kN * sizeof(__nv_bfloat16);

  auto input_host = Read(fixture / "input_fp4.bin", kInputBytes);
  auto input_scales_host =
      Read(fixture / "input_scales.bin", kInputScaleBytes);
  auto weight_host = Read(fixture / "weight_fp4.bin", kWeightBytes);
  auto weight_scales_host =
      Read(fixture / "weight_scales.bin", kWeightScaleBytes);
  auto alpha_host = Read(fixture / "alpha_f32.bin", sizeof(float));
  auto expected = Read(fixture / "output_bf16.bin", kOutputBytes);

  void* input = Upload(input_host);
  void* input_scales = Upload(input_scales_host);
  void* weight = Upload(weight_host);
  void* weight_scales = Upload(weight_scales_host);
  void* alpha = Upload(alpha_host);
  void* output = nullptr;
  CudaOk(cudaMalloc(&output, kOutputBytes));

  SparkServeDenseNvfp4Plan plan = {
      sizeof(SparkServeDenseNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kM,
      kN,
      kK,
      kN,
      kK,
      kN,
      16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_FLASHINFER_MM_FP4,
      0,
  };
  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_dense_nvfp4_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.available == 1);
  void* workspace = nullptr;
  if (info.workspace_bytes != 0) CudaOk(cudaMalloc(&workspace, info.workspace_bytes));

  SparkServeDenseNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.input = {input, input_scales, kK / 2, kK / 16};
  args.weight = {weight, weight_scales, kK / 2, kK / 16};
  args.output = output;
  args.output_row_stride_bytes = kN * sizeof(__nv_bfloat16);
  args.alpha = *reinterpret_cast<const float*>(alpha_host.data());
  args.workspace = workspace;
  args.workspace_bytes = info.workspace_bytes;
  args.alpha_device = static_cast<const float*>(alpha);
  SparkServeStatus status = sparkserve_dense_nvfp4_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> actual(kOutputBytes);
  CudaOk(cudaMemcpy(actual.data(), output, actual.size(), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); index += sizeof(uint16_t)) {
    const uint16_t got = *reinterpret_cast<const uint16_t*>(&actual[index]);
    const uint16_t want = *reinterpret_cast<const uint16_t*>(&expected[index]);
    mismatches += got != want;
  }
  std::cout << "real Qwen expert NVFP4 BF16 mismatches: " << mismatches << '\n';
  assert(mismatches == 0);

  if (workspace != nullptr) CudaOk(cudaFree(workspace));
  CudaOk(cudaFree(output));
  CudaOk(cudaFree(alpha));
  CudaOk(cudaFree(weight_scales));
  CudaOk(cudaFree(weight));
  CudaOk(cudaFree(input_scales));
  CudaOk(cudaFree(input));
  return 0;
}

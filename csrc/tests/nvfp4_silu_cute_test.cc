#include "sparkserve/kernel_api.h"

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

void* Zeros(size_t bytes) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, bytes));
  CudaOk(cudaMemset(device, 0, bytes));
  return device;
}

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

void Upload(void* device, const std::vector<uint8_t>& host) {
  CudaOk(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 1 || argc == 2);
  constexpr uint32_t kExperts = 2;
  constexpr uint32_t kRowsPerExpert = 4;
  constexpr uint64_t kHidden = 640;
  constexpr size_t kInputExpertBytes = kRowsPerExpert * kHidden * 4;
  constexpr size_t kOutputExpertBytes = kRowsPerExpert * kHidden / 2;
  constexpr size_t kScaleExpertBytes = 128 * (kHidden / 16);

  constexpr size_t kInputBytes = kExperts * kInputExpertBytes;
  constexpr size_t kOutputBytes = kExperts * kOutputExpertBytes;
  constexpr size_t kScaleBytes = kExperts * kScaleExpertBytes;
  void* input = Zeros(kInputBytes);
  void* output = Zeros(kExperts * kOutputExpertBytes);
  void* output_scales = Zeros(kExperts * kScaleExpertBytes);
  std::vector<uint8_t> host_scale_values(sizeof(float) * kExperts);
  const float default_scales[] = {1.0F, 2.0F};
  std::memcpy(host_scale_values.data(), default_scales,
              sizeof(default_scales));
  int32_t active_rows[] = {4, 2};
  std::vector<uint8_t> expected_output;
  std::vector<uint8_t> expected_scales;
  if (argc == 2) {
    const std::filesystem::path fixture(argv[1]);
    Upload(input, Read(fixture / "input_bf16.bin", kInputBytes));
    host_scale_values =
        Read(fixture / "global_scales_f32.bin", sizeof(float) * kExperts);
    auto active =
        Read(fixture / "active_rows_i32.bin", sizeof(int32_t) * kExperts);
    std::memcpy(active_rows, active.data(), active.size());
    expected_output = Read(fixture / "output_fp4.bin", kOutputBytes);
    expected_scales = Read(fixture / "output_scales.bin", kScaleBytes);
  }
  float* global_scales = nullptr;
  CudaOk(cudaMalloc(&global_scales, host_scale_values.size()));
  CudaOk(cudaMemcpy(global_scales, host_scale_values.data(),
                    host_scale_values.size(), cudaMemcpyHostToDevice));

  SparkServeSiluNvfp4Plan plan = {
      sizeof(SparkServeSiluNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kExperts,
      kRowsPerExpert,
      kHidden,
      16,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4,
      0,
  };
  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_silu_nvfp4_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.available == 1);

  SparkServeSiluNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.input = input;
  args.input_global_scales = global_scales;
  args.active_rows = active_rows;
  args.packed_output = output;
  args.output_scales = output_scales;
  args.input_expert_stride_bytes = kInputExpertBytes;
  args.output_expert_stride_bytes = kOutputExpertBytes;
  args.scale_expert_stride_bytes = kScaleExpertBytes;
  SparkServeStatus status = sparkserve_silu_nvfp4_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> host_output(kOutputBytes);
  std::vector<uint8_t> host_scales(kScaleBytes);
  CudaOk(cudaMemcpy(host_output.data(), output, host_output.size(),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(host_scales.data(), output_scales, host_scales.size(),
                    cudaMemcpyDeviceToHost));
  if (argc == 1) {
    for (uint8_t value : host_output) assert(value == 0);
    for (uint8_t value : host_scales) assert(value == 0);
    std::cout << "FlashInfer CuTe fused SiLU NVFP4 zero smoke passed\n";
  } else {
    size_t output_mismatches = 0;
    size_t scale_mismatches = 0;
    for (size_t index = 0; index < host_output.size(); ++index) {
      output_mismatches += host_output[index] != expected_output[index];
    }
    for (size_t index = 0; index < host_scales.size(); ++index) {
      scale_mismatches += host_scales[index] != expected_scales[index];
    }
    std::cout << "FlashInfer CuTe fused SiLU NVFP4 mismatches: values="
              << output_mismatches << " scales=" << scale_mismatches << '\n';
    assert(output_mismatches == 0);
    assert(scale_mismatches == 0);
  }

  CudaOk(cudaMemset(output, 0, kOutputBytes));
  CudaOk(cudaMemset(output_scales, 0, kScaleBytes));
  const int32_t m_indptr[] = {0, 4, 8};
  const uint64_t scale_row_offsets[] = {0, 128};
  SparkServeSegmentedSiluNvfp4Plan segmented_plan = {
      sizeof(SparkServeSegmentedSiluNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kExperts,
      16,
      8,
      256,
      kHidden,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4,
      0,
  };
  SparkServeSegmentedSiluNvfp4Args segmented_args = {};
  segmented_args.struct_size = sizeof(segmented_args);
  segmented_args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  segmented_args.plan = segmented_plan;
  segmented_args.input = input;
  segmented_args.input_global_scales = global_scales;
  segmented_args.active_rows_host = active_rows;
  segmented_args.m_indptr_host = m_indptr;
  segmented_args.scale_row_offsets_host = scale_row_offsets;
  segmented_args.packed_output = output;
  segmented_args.output_scales = output_scales;
  segmented_args.input_row_stride_bytes = kHidden * 4;
  segmented_args.output_row_stride_bytes = kHidden / 2;
  segmented_args.scale_row_stride_bytes = kHidden / 16;
  status = sparkserve_segmented_silu_nvfp4_launch(&caps, &segmented_args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  CudaOk(cudaMemcpy(host_output.data(), output, host_output.size(),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(host_scales.data(), output_scales, host_scales.size(),
                    cudaMemcpyDeviceToHost));
  if (argc == 1) {
    for (uint8_t value : host_output) assert(value == 0);
    for (uint8_t value : host_scales) assert(value == 0);
  } else {
    assert(host_output == expected_output);
    assert(host_scales == expected_scales);
  }
  std::cout << "FlashInfer CuTe segmented SiLU NVFP4 parity passed\n";

  CudaOk(cudaFree(global_scales));
  CudaOk(cudaFree(output_scales));
  CudaOk(cudaFree(output));
  CudaOk(cudaFree(input));
  return 0;
}

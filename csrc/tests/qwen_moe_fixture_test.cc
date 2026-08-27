#include "sparkserve/kernel_api.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t kExperts = 2;
constexpr uint64_t kRows = 8;
constexpr uint64_t kScaleRows = 256;
constexpr uint64_t kHidden = 2560;
constexpr uint64_t kMoe = 640;
constexpr size_t kWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

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

void* Allocate(size_t bytes) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, bytes));
  CudaOk(cudaMemset(device, 0, bytes));
  return device;
}

void ExpectBytes(void* device, const std::vector<uint8_t>& expected,
                 const char* stage) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(),
                    cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != expected[index];
  }
  std::cout << stage << " mismatched bytes: " << mismatches << '\n';
  assert(mismatches == 0);
}

SparkServeGroupedNvfp4Args GroupedArgs(
    uint64_t n, uint64_t k, const void* input, const void* input_scales,
    const void* weights, const void* weight_scales, const int32_t* m_indptr,
    const float* alpha, void* output, void* int_workspace,
    void* float_workspace) {
  SparkServeGroupedNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = {
      sizeof(SparkServeGroupedNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kExperts,
      16,
      kRows,
      kScaleRows,
      n,
      k,
      128,
      128,
      256,
      0,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_FLASHINFER_GROUP_MM_FP4,
  };
  args.input = {input, input_scales, k / 2, k / 16};
  args.weights = {weights, weight_scales, n * k / 2, n * k / 16};
  args.m_indptr = m_indptr;
  args.alpha_device = alpha;
  args.output = output;
  args.output_row_stride_bytes = n * 2;
  args.int_workspace = int_workspace;
  args.int_workspace_bytes = kWorkspaceBytes;
  args.float_workspace = float_workspace;
  args.float_workspace_bytes = kWorkspaceBytes;
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  const auto input_host = Read(fixture / "input_fp4.bin", kRows * kHidden / 2);
  const auto input_scales_host =
      Read(fixture / "input_scales.bin", kScaleRows * kHidden / 16);
  const auto w13_host =
      Read(fixture / "w13_fp4.bin", kExperts * 2 * kMoe * kHidden / 2);
  const auto w13_scales_host = Read(
      fixture / "w13_scales.bin", kExperts * 2 * kMoe * kHidden / 16);
  const auto w13_alpha_host =
      Read(fixture / "w13_alpha_f32.bin", kExperts * sizeof(float));
  const auto m_indptr_host =
      Read(fixture / "m_indptr_i32.bin", (kExperts + 1) * sizeof(int32_t));
  const auto gateup_expected =
      Read(fixture / "gateup_bf16.bin", kRows * 2 * kMoe * 2);
  const auto down_global_host = Read(fixture / "down_global_scales_f32.bin",
                                     kExperts * sizeof(float));
  const auto down_input_expected =
      Read(fixture / "down_input_fp4.bin", kRows * kMoe / 2);
  const auto down_scales_expected =
      Read(fixture / "down_input_scales.bin", kScaleRows * kMoe / 16);
  const auto w2_host =
      Read(fixture / "w2_fp4.bin", kExperts * kHidden * kMoe / 2);
  const auto w2_scales_host =
      Read(fixture / "w2_scales.bin", kExperts * kHidden * kMoe / 16);
  const auto w2_alpha_host =
      Read(fixture / "w2_alpha_f32.bin", kExperts * sizeof(float));
  const auto output_expected =
      Read(fixture / "output_bf16.bin", kRows * kHidden * 2);

  void* input = Upload(input_host);
  void* input_scales = Upload(input_scales_host);
  void* w13 = Upload(w13_host);
  void* w13_scales = Upload(w13_scales_host);
  void* w13_alpha = Upload(w13_alpha_host);
  void* m_indptr = Upload(m_indptr_host);
  void* gateup = Allocate(gateup_expected.size());
  void* down_global = Upload(down_global_host);
  void* down_input = Allocate(down_input_expected.size());
  void* down_scales = Allocate(down_scales_expected.size());
  void* w2 = Upload(w2_host);
  void* w2_scales = Upload(w2_scales_host);
  void* w2_alpha = Upload(w2_alpha_host);
  void* output = Allocate(output_expected.size());
  void* int_workspace = Allocate(kWorkspaceBytes);
  void* float_workspace = Allocate(kWorkspaceBytes);

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  auto gateup_args = GroupedArgs(
      2 * kMoe, kHidden, input, input_scales, w13, w13_scales,
      static_cast<const int32_t*>(m_indptr), static_cast<const float*>(w13_alpha),
      gateup, int_workspace, float_workspace);
  SparkServeStatus status = sparkserve_grouped_nvfp4_launch(&caps, &gateup_args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(gateup, gateup_expected, "grouped gate/up");

  const int32_t active_rows[] = {4, 4};
  const int32_t m_indptr_cpu[] = {0, 4, 8};
  const uint64_t scale_offsets[] = {0, 128};
  SparkServeSegmentedSiluNvfp4Args activation = {};
  activation.struct_size = sizeof(activation);
  activation.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  activation.plan = {
      sizeof(SparkServeSegmentedSiluNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kExperts,
      16,
      kRows,
      kScaleRows,
      kMoe,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4,
      0,
  };
  activation.input = gateup;
  activation.input_global_scales = static_cast<const float*>(down_global);
  activation.active_rows_host = active_rows;
  activation.m_indptr_host = m_indptr_cpu;
  activation.scale_row_offsets_host = scale_offsets;
  activation.packed_output = down_input;
  activation.output_scales = down_scales;
  activation.input_row_stride_bytes = kMoe * 4;
  activation.output_row_stride_bytes = kMoe / 2;
  activation.scale_row_stride_bytes = kMoe / 16;
  status = sparkserve_segmented_silu_nvfp4_launch(&caps, &activation);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(down_input, down_input_expected, "fused activation values");
  ExpectBytes(down_scales, down_scales_expected, "fused activation scales");

  auto down_args = GroupedArgs(
      kHidden, kMoe, down_input, down_scales, w2, w2_scales,
      static_cast<const int32_t*>(m_indptr), static_cast<const float*>(w2_alpha),
      output, int_workspace, float_workspace);
  status = sparkserve_grouped_nvfp4_launch(&caps, &down_args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(output, output_expected, "grouped down");

  CudaOk(cudaFree(float_workspace));
  CudaOk(cudaFree(int_workspace));
  CudaOk(cudaFree(output));
  CudaOk(cudaFree(w2_alpha));
  CudaOk(cudaFree(w2_scales));
  CudaOk(cudaFree(w2));
  CudaOk(cudaFree(down_scales));
  CudaOk(cudaFree(down_input));
  CudaOk(cudaFree(down_global));
  CudaOk(cudaFree(gateup));
  CudaOk(cudaFree(m_indptr));
  CudaOk(cudaFree(w13_alpha));
  CudaOk(cudaFree(w13_scales));
  CudaOk(cudaFree(w13));
  CudaOk(cudaFree(input_scales));
  CudaOk(cudaFree(input));
  return 0;
}

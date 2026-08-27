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
  const auto token_input_host =
      Read(fixture / "token_input_bf16.bin", 2 * kHidden * 2);
  const auto route_map_host =
      Read(fixture / "route_to_packed_u32.bin", 4 * sizeof(uint32_t));
  const auto route_weights_host =
      Read(fixture / "route_weights_f32.bin", 4 * sizeof(float));
  const auto packed_input_expected =
      Read(fixture / "packed_input_bf16.bin", kRows * kHidden * 2);
  const auto input_host = Read(fixture / "input_fp4.bin", kRows * kHidden / 2);
  const auto input_scales_host =
      Read(fixture / "input_scales.bin", kScaleRows * kHidden / 16);
  const auto w13_global_host =
      Read(fixture / "w13_global_scales_f32.bin", kExperts * sizeof(float));
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
  const auto final_output_expected =
      Read(fixture / "final_output_bf16.bin", 2 * kHidden * 2);

  void* token_input = Upload(token_input_host);
  void* route_map = Upload(route_map_host);
  void* route_weights = Upload(route_weights_host);
  void* packed_input_bf16 = Allocate(packed_input_expected.size());
  void* input = Allocate(input_host.size());
  void* input_scales = Allocate(input_scales_host.size());
  void* w13_global = Upload(w13_global_host);
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
  void* final_output = Allocate(final_output_expected.size());
  void* int_workspace = Allocate(kWorkspaceBytes);
  void* float_workspace = Allocate(kWorkspaceBytes);

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeMoeRouteArgs route = {};
  route.struct_size = sizeof(route);
  route.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  route.plan = {
      sizeof(SparkServeMoeRoutePlan), SPARKSERVE_KERNEL_ABI_VERSION,
      2,                               2,
      kExperts,                        0,
      kHidden,                         kRows,
  };
  route.token_input = token_input;
  route.route_to_packed_row = static_cast<const uint32_t*>(route_map);
  route.packed_input = packed_input_bf16;
  route.token_input_row_stride_bytes = kHidden * 2;
  route.packed_row_stride_bytes = kHidden * 2;
  SparkServeStatus status = sparkserve_moe_route_dispatch(&caps, &route);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(packed_input_bf16, packed_input_expected, "route dispatch");

  const int32_t active_rows[] = {2, 2};
  const int32_t m_indptr_cpu[] = {0, 4, 8};
  const uint64_t scale_offsets[] = {0, 128};
  SparkServeSegmentedNvfp4QuantizeArgs quantize = {};
  quantize.struct_size = sizeof(quantize);
  quantize.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  quantize.plan = {
      sizeof(SparkServeSegmentedNvfp4QuantizePlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kExperts,
      16,
      kRows,
      kScaleRows,
      kHidden,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_BACKEND_FLASHINFER_CUTE_NVFP4_QUANTIZE,
      0,
  };
  quantize.input = packed_input_bf16;
  quantize.input_global_scales = static_cast<const float*>(w13_global);
  quantize.active_rows_host = active_rows;
  quantize.m_indptr_host = m_indptr_cpu;
  quantize.scale_row_offsets_host = scale_offsets;
  quantize.packed_output = input;
  quantize.output_scales = input_scales;
  quantize.input_row_stride_bytes = kHidden * 2;
  quantize.output_row_stride_bytes = kHidden / 2;
  quantize.scale_row_stride_bytes = kHidden / 16;
  status = sparkserve_segmented_nvfp4_quantize_launch(&caps, &quantize);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(input, input_host, "routed input values");
  ExpectBytes(input_scales, input_scales_host, "routed input scales");

  auto gateup_args = GroupedArgs(
      2 * kMoe, kHidden, input, input_scales, w13, w13_scales,
      static_cast<const int32_t*>(m_indptr), static_cast<const float*>(w13_alpha),
      gateup, int_workspace, float_workspace);
  status = sparkserve_grouped_nvfp4_launch(&caps, &gateup_args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(gateup, gateup_expected, "grouped gate/up");

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

  route.route_weights = static_cast<const float*>(route_weights);
  route.packed_expert_output = output;
  route.token_output = final_output;
  route.expert_output_row_stride_bytes = kHidden * 2;
  status = sparkserve_moe_route_finalize(&caps, &route);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(final_output, final_output_expected, "weighted route finalize");

  CudaOk(cudaFree(float_workspace));
  CudaOk(cudaFree(int_workspace));
  CudaOk(cudaFree(final_output));
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
  CudaOk(cudaFree(w13_global));
  CudaOk(cudaFree(packed_input_bf16));
  CudaOk(cudaFree(route_weights));
  CudaOk(cudaFree(route_map));
  CudaOk(cudaFree(token_input));
  return 0;
}

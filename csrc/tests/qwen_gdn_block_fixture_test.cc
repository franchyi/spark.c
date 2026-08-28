#include "sparkserve/kernel_api.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kHc = 4;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kLowrank = 320;
constexpr uint32_t kQkHeads = 16;
constexpr uint32_t kValueHeads = 48;
constexpr uint32_t kHeadDim = 128;
constexpr uint64_t kQkWidth = kQkHeads * kHeadDim;
constexpr uint64_t kValueWidth = kValueHeads * kHeadDim;
constexpr uint64_t kConvWidth = 2 * kQkWidth + kValueWidth;
constexpr uint64_t kConvKernel = 4;
constexpr uint64_t kConvStateWidth = kConvKernel - 1;

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

void ExpectBytes(const void* device, const std::vector<uint8_t>& expected,
                 const char* stage) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index)
    mismatches += actual[index] != expected[index];
  std::cout << stage << " mismatched bytes: " << mismatches << '\n';
  if (mismatches != 0 && expected.size() % 2 == 0) {
    const auto* actual_bf16 =
        reinterpret_cast<const uint16_t*>(actual.data());
    const auto* expected_bf16 =
        reinterpret_cast<const uint16_t*>(expected.data());
    float max_error = 0.0F;
    size_t shown = 0;
    for (size_t index = 0; index < expected.size() / 2; ++index) {
      uint32_t actual_bits = static_cast<uint32_t>(actual_bf16[index]) << 16;
      uint32_t expected_bits =
          static_cast<uint32_t>(expected_bf16[index]) << 16;
      float actual_value = 0.0F;
      float expected_value = 0.0F;
      std::memcpy(&actual_value, &actual_bits, sizeof(actual_value));
      std::memcpy(&expected_value, &expected_bits, sizeof(expected_value));
      max_error = std::max(max_error, std::abs(actual_value - expected_value));
      if (actual_bf16[index] != expected_bf16[index] && shown < 5) {
        std::cout << "  [" << index << "] actual=" << actual_value
                  << " expected=" << expected_value << '\n';
        ++shown;
      }
    }
    std::cout << "  max BF16 absolute error: " << max_error << '\n';
  }
  std::cout.flush();
  assert(mismatches == 0);
}

void Check(SparkServeStatus status) {
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);

  const auto hyper_input =
      Read(fixture / "mhc_hyper_input_bf16.bin", kHc * kHidden * 2);
  const auto mhc_norm_weight =
      Read(fixture / "mhc_norm_weight_bf16.bin", kHc * kHidden * 2);
  const auto mhc_down_weight = Read(fixture / "mhc_down_weight_bf16.bin",
                                    kLowrank * kHc * kHidden * 2);
  const auto mhc_up_weight = Read(fixture / "mhc_up_weight_bf16.bin",
                                  kHc * kHidden * kLowrank * 2);
  const auto mhc_inject_weight = Read(fixture / "mhc_inject_weight_bf16.bin",
                                      kHc * kHc * kHidden * 2);
  const auto mhc_normed_expected =
      Read(fixture / "mhc_normed_bf16.bin", kHc * kHidden * 2);
  const auto mhc_down_expected =
      Read(fixture / "mhc_down_bf16.bin", kLowrank * 2);
  const auto mhc_activated_expected =
      Read(fixture / "mhc_activated_bf16.bin", kLowrank * 2);
  const auto mhc_up_expected =
      Read(fixture / "mhc_up_bf16.bin", kHc * kHidden * 2);
  const auto mixed_expected =
      Read(fixture / "mixed_hidden_bf16.bin", kHidden * 2);
  const auto combined_expected =
      Read(fixture / "mhc_combined_bf16.bin", kHc * kHidden * 2);

  const auto qkv_weight = Read(fixture / "in_proj_qkv_weight_bf16.bin",
                               kConvWidth * kHidden * 2);
  const auto z_weight = Read(fixture / "in_proj_z_weight_bf16.bin",
                             kValueWidth * kHidden * 2);
  const auto b_weight = Read(fixture / "in_proj_b_weight_bf16.bin",
                             kValueHeads * kHidden * 2);
  const auto a_weight = Read(fixture / "in_proj_a_weight_bf16.bin",
                             kValueHeads * kHidden * 2);
  const auto conv_weight = Read(fixture / "conv_weight_view_bf16.bin",
                                kConvWidth * kConvKernel * 2);
  const auto norm_weight =
      Read(fixture / "gated_norm_weight_bf16.bin", kHeadDim * 2);
  const auto out_weight = Read(fixture / "out_proj_weight_bf16.bin",
                               kHidden * kValueWidth * 2);
  const auto a_log = Read(fixture / "a_log_f32.bin", kValueHeads * 4);
  const auto dt_bias = Read(fixture / "dt_bias_f32.bin", kValueHeads * 4);
  const auto indices = Read(fixture / "state_indices_i32.bin", sizeof(int32_t));

  const auto qkv_expected =
      Read(fixture / "projected_qkv_bf16.bin", kConvWidth * 2);
  const auto z_expected =
      Read(fixture / "projected_z_bf16.bin", kValueWidth * 2);
  const auto b_expected =
      Read(fixture / "projected_b_bf16.bin", kValueHeads * 2);
  const auto a_expected =
      Read(fixture / "projected_a_bf16.bin", kValueHeads * 2);
  const auto conv_before = Read(fixture / "conv_state_before_bf16.bin",
                                kConvWidth * kConvStateWidth * 2);
  const auto conv_after = Read(fixture / "conv_state_after_bf16.bin",
                               kConvWidth * kConvStateWidth * 2);
  const auto convolved_expected =
      Read(fixture / "convolved_qkv_bf16.bin", kConvWidth * 2);
  const auto temporal_before = Read(
      fixture / "temporal_state_before_bf16.bin",
      kValueHeads * kHeadDim * kHeadDim * 2);
  const auto temporal_after = Read(
      fixture / "temporal_state_after_bf16.bin", temporal_before.size());
  const auto core_expected =
      Read(fixture / "gdn_core_output_bf16.bin", kValueWidth * 2);
  const auto gated_expected =
      Read(fixture / "gated_norm_bf16.bin", kValueWidth * 2);
  const auto attention_expected =
      Read(fixture / "attention_output_bf16.bin", kHidden * 2);

  void* hyper_input_d = Upload(hyper_input);
  void* mhc_norm_weight_d = Upload(mhc_norm_weight);
  void* mhc_down_weight_d = Upload(mhc_down_weight);
  void* mhc_up_weight_d = Upload(mhc_up_weight);
  void* mhc_inject_weight_d = Upload(mhc_inject_weight);
  void* mhc_normed_d = Allocate(mhc_normed_expected.size());
  void* mhc_down_d = Allocate(mhc_down_expected.size());
  void* mhc_activated_d = Allocate(mhc_activated_expected.size());
  void* mhc_up_d = Allocate(mhc_up_expected.size());
  void* mixed_d = Allocate(mixed_expected.size());
  void* combined_d = Allocate(combined_expected.size());

  void* qkv_weight_d = Upload(qkv_weight);
  void* z_weight_d = Upload(z_weight);
  void* b_weight_d = Upload(b_weight);
  void* a_weight_d = Upload(a_weight);
  void* conv_weight_d = Upload(conv_weight);
  void* norm_weight_d = Upload(norm_weight);
  void* out_weight_d = Upload(out_weight);
  void* a_log_d = Upload(a_log);
  void* dt_bias_d = Upload(dt_bias);
  void* indices_d = Upload(indices);
  void* qkv_d = Allocate(qkv_expected.size());
  void* z_d = Allocate(z_expected.size());
  void* b_d = Allocate(b_expected.size());
  void* a_d = Allocate(a_expected.size());
  void* conv_state_d = Upload(conv_before);
  void* conv_seed_d = Upload(conv_before);
  void* convolved_d = Allocate(convolved_expected.size());
  void* temporal_state_d = Upload(temporal_before);
  void* temporal_seed_d = Upload(temporal_before);
  void* core_d = Allocate(core_expected.size());
  void* gated_d = Allocate(gated_expected.size());
  void* attention_d = Allocate(attention_expected.size());

  cublasHandle_t blas = nullptr;
  CublasOk(cublasCreate(&blas));
  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeMhcArgs mhc = {};
  mhc.struct_size = sizeof(mhc);
  mhc.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  mhc.plan = {sizeof(SparkServeMhcPlan),
              SPARKSERVE_KERNEL_ABI_VERSION,
              1,
              kHc,
              kHidden,
              kLowrank,
              SPARKSERVE_DTYPE_BF16,
              SPARKSERVE_BACKEND_SGLANG_CUBLAS_MHC,
              1.0e-6F,
              0,
              0,
              0};
  mhc.hyper_input = hyper_input_d;
  mhc.norm_weight = mhc_norm_weight_d;
  mhc.mix_down_weight = mhc_down_weight_d;
  mhc.mix_up_weight = mhc_up_weight_d;
  mhc.inject_weight = mhc_inject_weight_d;
  mhc.normed = mhc_normed_d;
  mhc.mix_down = mhc_down_d;
  mhc.mix_activated = mhc_activated_d;
  mhc.mix_up = mhc_up_d;
  mhc.mixed_output = mixed_d;
  mhc.block_output = attention_d;
  mhc.combined_output = combined_d;
  mhc.cublas_handle = blas;

  SparkServeGdnBlockArgs block = {};
  block.struct_size = sizeof(block);
  block.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  block.plan = {sizeof(SparkServeGdnBlockPlan),
                SPARKSERVE_KERNEL_ABI_VERSION,
                1,
                kHidden,
                kQkHeads,
                kValueHeads,
                kHeadDim,
                kConvKernel,
                SPARKSERVE_DTYPE_BF16,
                SPARKSERVE_BACKEND_SGLANG_CUBLAS_GDN_BLOCK,
                1.0e-6F,
                0};
  block.hidden_states = mixed_d;
  block.in_proj_qkv_weight = qkv_weight_d;
  block.in_proj_z_weight = z_weight_d;
  block.in_proj_b_weight = b_weight_d;
  block.in_proj_a_weight = a_weight_d;
  block.conv_weight = conv_weight_d;
  block.gated_norm_weight = norm_weight_d;
  block.out_proj_weight = out_weight_d;
  block.conv_state_pool = conv_state_d;
  block.state_indices = static_cast<const int32_t*>(indices_d);
  block.projected_qkv = qkv_d;
  block.projected_z = z_d;
  block.projected_b = b_d;
  block.projected_a = a_d;
  block.convolved_qkv = convolved_d;
  block.gdn_core_output = core_d;
  block.gated_norm_output = gated_d;
  block.attention_output = attention_d;
  block.cublas_handle = blas;

  SparkServeGdnDecodeArgs recurrence = {};
  recurrence.struct_size = sizeof(recurrence);
  recurrence.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  recurrence.plan = {sizeof(SparkServeGdnDecodePlan),
                     SPARKSERVE_KERNEL_ABI_VERSION,
                     1,
                     kQkHeads,
                     kValueHeads,
                     kHeadDim,
                     kHeadDim,
                     1,
                     SPARKSERVE_DTYPE_BF16,
                     SPARKSERVE_GDN_BACKEND_FLASHINFER};
  auto* convolved_bytes = static_cast<uint8_t*>(convolved_d);
  recurrence.q = convolved_bytes;
  recurrence.k = convolved_bytes + kQkWidth * 2;
  recurrence.v = convolved_bytes + 2 * kQkWidth * 2;
  recurrence.a = a_d;
  recurrence.b = b_d;
  recurrence.a_log = static_cast<const float*>(a_log_d);
  recurrence.dt_bias = static_cast<const float*>(dt_bias_d);
  recurrence.state_pool = temporal_state_d;
  recurrence.state_indices = static_cast<const int32_t*>(indices_d);
  recurrence.output = core_d;
  recurrence.scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

  Check(sparkserve_mhc_mix_launch(&caps, &mhc));
  Check(sparkserve_gdn_block_prepare_launch(&caps, &block));
  Check(sparkserve_gdn_decode_launch(&caps, &recurrence));
  Check(sparkserve_gdn_block_finish_launch(&caps, &block));
  Check(sparkserve_mhc_combine_launch(&caps, &mhc));
  CudaOk(cudaDeviceSynchronize());

  ExpectBytes(mhc_normed_d, mhc_normed_expected, "attention mHC norm");
  ExpectBytes(mhc_down_d, mhc_down_expected, "attention mHC down");
  ExpectBytes(mhc_activated_d, mhc_activated_expected,
              "attention mHC activation");
  ExpectBytes(mhc_up_d, mhc_up_expected, "attention mHC up");
  ExpectBytes(mixed_d, mixed_expected, "attention mixed hidden");
  ExpectBytes(qkv_d, qkv_expected, "GDN QKV projection");
  ExpectBytes(z_d, z_expected, "GDN Z projection");
  ExpectBytes(b_d, b_expected, "GDN B projection");
  ExpectBytes(a_d, a_expected, "GDN A projection");
  ExpectBytes(conv_state_d, conv_after, "GDN convolution state");
  ExpectBytes(convolved_d, convolved_expected, "GDN causal convolution");
  ExpectBytes(temporal_state_d, temporal_after, "GDN temporal state");
  ExpectBytes(core_d, core_expected, "GDN recurrent output");
  ExpectBytes(gated_d, gated_expected, "GDN gated RMSNorm");
  ExpectBytes(attention_d, attention_expected, "GDN output projection");
  ExpectBytes(combined_d, combined_expected, "attention mHC combine");

  constexpr int kIterations = 20;
  float total_ms = 0.0F;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CudaOk(cudaEventCreate(&start));
  CudaOk(cudaEventCreate(&stop));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    CudaOk(cudaMemcpyAsync(conv_state_d, conv_seed_d, conv_before.size(),
                           cudaMemcpyDeviceToDevice));
    CudaOk(cudaMemcpyAsync(temporal_state_d, temporal_seed_d,
                           temporal_before.size(), cudaMemcpyDeviceToDevice));
    CudaOk(cudaEventRecord(start));
    Check(sparkserve_mhc_mix_launch(&caps, &mhc));
    Check(sparkserve_gdn_block_prepare_launch(&caps, &block));
    Check(sparkserve_gdn_decode_launch(&caps, &recurrence));
    Check(sparkserve_gdn_block_finish_launch(&caps, &block));
    Check(sparkserve_mhc_combine_launch(&caps, &mhc));
    CudaOk(cudaEventRecord(stop));
    CudaOk(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    CudaOk(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_ms += elapsed_ms;
  }
  std::cout << "full Qwen GDN attention half-layer: "
            << total_ms * 1000.0F / kIterations << " us/token\n";
  CudaOk(cudaEventDestroy(stop));
  CudaOk(cudaEventDestroy(start));

  CublasOk(cublasDestroy(blas));
  for (void* pointer :
       {attention_d, gated_d, core_d, temporal_seed_d, temporal_state_d,
        convolved_d, conv_seed_d, conv_state_d, a_d, b_d, z_d, qkv_d,
        indices_d, dt_bias_d, a_log_d, out_weight_d, norm_weight_d,
        conv_weight_d, a_weight_d, b_weight_d, z_weight_d, qkv_weight_d,
        combined_d, mixed_d, mhc_up_d, mhc_activated_d, mhc_down_d,
        mhc_normed_d, mhc_inject_weight_d, mhc_up_weight_d,
        mhc_down_weight_d, mhc_norm_weight_d, hyper_input_d}) {
    CudaOk(cudaFree(pointer));
  }
  return 0;
}

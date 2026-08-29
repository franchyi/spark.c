#include "q27_gdn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}

void GdnOk(q27_gdn_status status) {
  if (status.code != Q27_GDN_OK) {
    std::fprintf(stderr, "q27 GDN: %s\n", status.message);
    std::abort();
  }
}

std::vector<uint8_t> Read(const std::filesystem::path& path, size_t bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  assert(input.good());
  assert(static_cast<size_t>(input.tellg()) == bytes);
  input.seekg(0);
  std::vector<uint8_t> data(bytes);
  input.read(reinterpret_cast<char*>(data.data()), data.size());
  assert(input.good());
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

void ExpectExact(const char* label, const void* device,
                 const std::vector<uint8_t>& expected) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(),
                    cudaMemcpyDeviceToHost));
  size_t mismatch = 0;
  for (size_t index = 0; index < actual.size(); ++index)
    mismatch += actual[index] != expected[index];
  std::printf("q27 GDN %-17s mismatched_bytes=%zu\n", label, mismatch);
  assert(mismatch == 0);
}

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void CheckParameterConversion(const std::vector<uint8_t>& a_log,
                              const std::vector<uint8_t>& dt_bias) {
  const float* source_a = reinterpret_cast<const float*>(a_log.data());
  const float* source_dt = reinterpret_cast<const float*>(dt_bias.data());
  std::vector<uint16_t> bf16_a(Q27_GDN_VALUE_HEADS);
  std::vector<uint16_t> bf16_dt(Q27_GDN_VALUE_HEADS);
  for (int index = 0; index < Q27_GDN_VALUE_HEADS; ++index) {
    bf16_a[index] = FloatToBf16(source_a[index]);
    bf16_dt[index] = FloatToBf16(source_dt[index]);
  }
  std::vector<uint8_t> a_bytes(bf16_a.size() * sizeof(uint16_t));
  std::vector<uint8_t> dt_bytes(bf16_dt.size() * sizeof(uint16_t));
  std::memcpy(a_bytes.data(), bf16_a.data(), a_bytes.size());
  std::memcpy(dt_bytes.data(), bf16_dt.data(), dt_bytes.size());
  void* a_bf16_d = Upload(a_bytes);
  void* dt_bf16_d = Upload(dt_bytes);
  float* a_f32_d = static_cast<float*>(Allocate(Q27_GDN_VALUE_HEADS * 4));
  float* dt_f32_d = static_cast<float*>(Allocate(Q27_GDN_VALUE_HEADS * 4));
  q27_gdn_convert_parameters_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_GDN_ABI_VERSION;
  args.a_log_bf16 = a_bf16_d;
  args.dt_bias_bf16 = dt_bf16_d;
  args.a_log_f32 = a_f32_d;
  args.dt_bias_f32 = dt_f32_d;
  GdnOk(q27_gdn_convert_parameters(&args));
  std::vector<float> actual_a(Q27_GDN_VALUE_HEADS);
  std::vector<float> actual_dt(Q27_GDN_VALUE_HEADS);
  CudaOk(cudaMemcpy(actual_a.data(), a_f32_d, actual_a.size() * 4,
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_dt.data(), dt_f32_d, actual_dt.size() * 4,
                    cudaMemcpyDeviceToHost));
  for (int index = 0; index < Q27_GDN_VALUE_HEADS; ++index) {
    assert(actual_a[index] == Bf16ToFloat(bf16_a[index]));
    assert(actual_dt[index] == Bf16ToFloat(bf16_dt[index]));
  }
  std::printf("q27 GDN BF16 parameter conversion exact\n");
  CudaOk(cudaFree(dt_f32_d));
  CudaOk(cudaFree(a_f32_d));
  CudaOk(cudaFree(dt_bf16_d));
  CudaOk(cudaFree(a_bf16_d));
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path root(argv[1]);
  constexpr size_t kQkvBytes = Q27_GDN_CONV_WIDTH * 2;
  constexpr size_t kValueBytes = Q27_GDN_VALUE_WIDTH * 2;
  constexpr size_t kGateBytes = Q27_GDN_VALUE_HEADS * 2;
  constexpr size_t kConvWeightBytes =
      Q27_GDN_CONV_WIDTH * Q27_GDN_CONV_KERNEL * 2;
  constexpr size_t kNormBytes = Q27_GDN_HEAD_DIM * 2;
  constexpr size_t kConvStateBytes =
      Q27_GDN_CONV_WIDTH * Q27_GDN_CONV_HISTORY * 2;
  constexpr size_t kRecurrentStateBytes =
      Q27_GDN_VALUE_HEADS * Q27_GDN_HEAD_DIM * Q27_GDN_HEAD_DIM * 2;

  q27_gdn_layout layout = {};
  layout.struct_size = sizeof(layout);
  layout.abi_version = Q27_GDN_ABI_VERSION;
  GdnOk(q27_gdn_query_layout(&layout));
  assert(layout.convolution_state_bytes_per_slot == kConvStateBytes);
  assert(layout.recurrent_state_bytes_per_slot == kRecurrentStateBytes);
  assert(layout.projected_qkv_bytes == kQkvBytes);
  assert(layout.recurrent_output_bytes == kValueBytes);

  const auto qkv = Read(root / "projected_qkv_bf16.bin", kQkvBytes);
  const auto z = Read(root / "projected_z_bf16.bin", kValueBytes);
  const auto a = Read(root / "projected_a_bf16.bin", kGateBytes);
  const auto b = Read(root / "projected_b_bf16.bin", kGateBytes);
  const auto conv_weight =
      Read(root / "conv_weight_view_bf16.bin", kConvWeightBytes);
  const auto norm_weight = Read(root / "gated_norm_weight_bf16.bin", kNormBytes);
  const auto a_log = Read(root / "a_log_f32.bin", Q27_GDN_VALUE_HEADS * 4);
  const auto dt_bias = Read(root / "dt_bias_f32.bin", Q27_GDN_VALUE_HEADS * 4);
  const auto indices = Read(root / "state_indices_i32.bin", sizeof(int32_t));
  const auto conv_before =
      Read(root / "conv_state_before_bf16.bin", kConvStateBytes);
  const auto conv_after =
      Read(root / "conv_state_after_bf16.bin", kConvStateBytes);
  const auto convolved = Read(root / "convolved_qkv_bf16.bin", kQkvBytes);
  const auto recurrent_before =
      Read(root / "temporal_state_before_bf16.bin", kRecurrentStateBytes);
  const auto recurrent_after =
      Read(root / "temporal_state_after_bf16.bin", kRecurrentStateBytes);
  const auto recurrent_output =
      Read(root / "gdn_core_output_bf16.bin", kValueBytes);
  const auto normalized = Read(root / "gated_norm_bf16.bin", kValueBytes);

  CheckParameterConversion(a_log, dt_bias);

  void* qkv_d = Upload(qkv);
  void* z_d = Upload(z);
  void* a_d = Upload(a);
  void* b_d = Upload(b);
  void* conv_weight_d = Upload(conv_weight);
  void* norm_weight_d = Upload(norm_weight);
  void* a_log_d = Upload(a_log);
  void* dt_bias_d = Upload(dt_bias);
  void* indices_d = Upload(indices);
  void* conv_state_d = Upload(conv_before);
  void* conv_seed_d = Upload(conv_before);
  void* recurrent_state_d = Upload(recurrent_before);
  void* recurrent_seed_d = Upload(recurrent_before);
  void* convolved_d = Allocate(kQkvBytes);
  void* recurrent_output_d = Allocate(kValueBytes);
  void* normalized_d = Allocate(kValueBytes);

  q27_gdn_decode_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_GDN_ABI_VERSION;
  args.state_slots = 1;
  args.projected_qkv_bf16 = qkv_d;
  args.projected_z_bf16 = z_d;
  args.projected_a_bf16 = a_d;
  args.projected_b_bf16 = b_d;
  args.conv_weight_bf16 = conv_weight_d;
  args.norm_weight_bf16 = norm_weight_d;
  args.a_log_f32 = static_cast<const float*>(a_log_d);
  args.dt_bias_f32 = static_cast<const float*>(dt_bias_d);
  args.convolution_state_bf16 = conv_state_d;
  args.recurrent_state_bf16 = recurrent_state_d;
  args.state_indices_i32 = static_cast<const int32_t*>(indices_d);
  args.convolved_qkv_bf16 = convolved_d;
  args.recurrent_output_bf16 = recurrent_output_d;
  args.normalized_output_bf16 = normalized_d;
  GdnOk(q27_gdn_decode(&args));
  CudaOk(cudaDeviceSynchronize());

  ExpectExact("convolution state", conv_state_d, conv_after);
  ExpectExact("convolved QKV", convolved_d, convolved);
  ExpectExact("recurrent state", recurrent_state_d, recurrent_after);
  ExpectExact("recurrent output", recurrent_output_d, recurrent_output);
  ExpectExact("gated RMSNorm", normalized_d, normalized);

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 100;
  float elapsed_ms = 0.0F;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    CudaOk(cudaMemcpyAsync(conv_state_d, conv_seed_d, kConvStateBytes,
                           cudaMemcpyDeviceToDevice));
    CudaOk(cudaMemcpyAsync(recurrent_state_d, recurrent_seed_d,
                           kRecurrentStateBytes, cudaMemcpyDeviceToDevice));
    CudaOk(cudaEventRecord(begin));
    GdnOk(q27_gdn_decode(&args));
    CudaOk(cudaEventRecord(end));
    CudaOk(cudaEventSynchronize(end));
    float iteration_ms = 0.0F;
    CudaOk(cudaEventElapsedTime(&iteration_ms, begin, end));
    elapsed_ms += iteration_ms;
  }
  std::printf("q27 GDN conv+recurrent+norm mean_us=%.3f\n",
              elapsed_ms * 1000.0F / kIterations);
  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));

  for (void* pointer : {normalized_d, recurrent_output_d, convolved_d,
                        recurrent_seed_d, recurrent_state_d, conv_seed_d,
                        conv_state_d, indices_d, dt_bias_d, a_log_d,
                        norm_weight_d, conv_weight_d, b_d, a_d, z_d, qkv_d})
    CudaOk(cudaFree(pointer));
  return 0;
}

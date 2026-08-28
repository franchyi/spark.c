#include "sparkserve/ggml_quant_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kExperts = 3;
constexpr uint64_t kRows = 32;
constexpr uint64_t kK = 512;
constexpr uint64_t kTokens = 2;
constexpr uint64_t kTopK = 2;

struct QuantCase {
  const char* slug;
  uint32_t quant_type;
  uint64_t block_elements;
  uint64_t block_bytes;
};

constexpr QuantCase kCases[] = {
    {"q8_0", SPARKSERVE_GGML_QUANT_Q8_0, 32, 34},
    {"q2_k", SPARKSERVE_GGML_QUANT_Q2_K, 256, 84},
    {"q3_k", SPARKSERVE_GGML_QUANT_Q3_K, 256, 110},
    {"q6_k", SPARKSERVE_GGML_QUANT_Q6_K, 256, 210},
    {"iq3_xxs", SPARKSERVE_GGML_QUANT_IQ3_XXS, 256, 98},
    {"iq3_s", SPARKSERVE_GGML_QUANT_IQ3_S, 256, 110},
    {"iq2_s", SPARKSERVE_GGML_QUANT_IQ2_S, 256, 82},
    {"iq4_xs", SPARKSERVE_GGML_QUANT_IQ4_XS, 256, 136},
};

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) std::cerr << cudaGetErrorString(error) << '\n';
  assert(error == cudaSuccess);
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

void* Upload(const std::vector<uint8_t>& host) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, host.size()));
  CudaOk(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
  return device;
}

void CheckClose(const std::vector<float>& actual,
                const std::vector<uint8_t>& expected_bytes,
                const std::string& label) {
  assert(expected_bytes.size() == actual.size() * sizeof(float));
  const auto* expected = reinterpret_cast<const float*>(expected_bytes.data());
  float max_absolute = 0.0F;
  float max_relative = 0.0F;
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float absolute = std::abs(actual[index] - expected[index]);
    const float relative = absolute / std::max(1.0F, std::abs(expected[index]));
    max_absolute = std::max(max_absolute, absolute);
    max_relative = std::max(max_relative, relative);
    mismatches += absolute > 0.03F + 0.0075F * std::abs(expected[index]);
  }
  std::cout << label << " max abs " << max_absolute << ", max rel "
            << max_relative << ", outliers " << mismatches << '\n';
  assert(mismatches == 0);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  constexpr size_t kInputBytes = kTokens * kK * sizeof(float);
  constexpr size_t kIdsBytes = kTokens * kTopK * sizeof(int32_t);
  constexpr size_t kDenseOutputBytes = kTokens * kRows * sizeof(float);
  constexpr size_t kRoutedOutputBytes =
      kTokens * kTopK * kRows * sizeof(float);

  auto input_host = Read(fixture / "input_f32.bin", kInputBytes);
  auto ids_host = Read(fixture / "expert_ids_i32.bin", kIdsBytes);
  void* input = Upload(input_host);
  void* ids = Upload(ids_host);
  void* dense_output = nullptr;
  void* routed_output = nullptr;
  CudaOk(cudaMalloc(&dense_output, kDenseOutputBytes));
  CudaOk(cudaMalloc(&routed_output, kRoutedOutputBytes));

  uint64_t scratch_bytes = 0;
  SparkServeStatus status = sparkserve_ggml_quant_q8_scratch_bytes(
      kTokens, kK, &scratch_bytes);
  assert(status.code == SPARKSERVE_STATUS_OK);
  assert(scratch_bytes == kTokens * (kK / 32) * 36);
  void* scratch = nullptr;
  CudaOk(cudaMalloc(&scratch, scratch_bytes));

  for (const QuantCase& quant : kCases) {
    const size_t weight_slice_bytes =
        kRows * (kK / quant.block_elements) * quant.block_bytes;
    auto weights_host =
        Read(fixture / (std::string("weights_") + quant.slug + ".bin"),
             kExperts * weight_slice_bytes);
    auto dense_expected = Read(
        fixture / (std::string("dense_") + quant.slug + "_f32.bin"),
        kDenseOutputBytes);
    auto routed_expected = Read(
        fixture / (std::string("routed_") + quant.slug + "_f32.bin"),
        kRoutedOutputBytes);
    void* weights = Upload(weights_host);

    SparkServeGgmlQuantDenseArgs dense_args{};
    dense_args.struct_size = sizeof(dense_args);
    dense_args.abi_version = SPARKSERVE_GGML_QUANT_ABI_VERSION;
    dense_args.quant_type = quant.quant_type;
    dense_args.input = static_cast<const float*>(input);
    dense_args.weights = weights;
    dense_args.output = static_cast<float*>(dense_output);
    dense_args.q8_scratch = scratch;
    dense_args.q8_scratch_bytes = scratch_bytes;
    dense_args.vectors = kTokens;
    dense_args.rows = kRows;
    dense_args.k = kK;
    status = sparkserve_ggml_quant_dense_launch(&dense_args);
    if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
    assert(status.code == SPARKSERVE_STATUS_OK);
    CudaOk(cudaDeviceSynchronize());
    std::vector<float> dense_actual(kTokens * kRows);
    CudaOk(cudaMemcpy(dense_actual.data(), dense_output, kDenseOutputBytes,
                      cudaMemcpyDeviceToHost));
    CheckClose(dense_actual, dense_expected,
               std::string(quant.slug) + " dense");

    SparkServeGgmlQuantRoutedArgs routed_args{};
    routed_args.struct_size = sizeof(routed_args);
    routed_args.abi_version = SPARKSERVE_GGML_QUANT_ABI_VERSION;
    routed_args.quant_type = quant.quant_type;
    routed_args.input = static_cast<const float*>(input);
    routed_args.weights = weights;
    routed_args.expert_ids = static_cast<const int32_t*>(ids);
    routed_args.output = static_cast<float*>(routed_output);
    routed_args.q8_scratch = scratch;
    routed_args.q8_scratch_bytes = scratch_bytes;
    routed_args.tokens = kTokens;
    routed_args.top_k = kTopK;
    routed_args.experts = kExperts;
    routed_args.rows = kRows;
    routed_args.k = kK;
    routed_args.weight_slot_stride_bytes = weight_slice_bytes;
    status = sparkserve_ggml_quant_routed_launch(&routed_args);
    if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
    assert(status.code == SPARKSERVE_STATUS_OK);
    CudaOk(cudaDeviceSynchronize());
    std::vector<float> routed_actual(kTokens * kTopK * kRows);
    CudaOk(cudaMemcpy(routed_actual.data(), routed_output, kRoutedOutputBytes,
                      cudaMemcpyDeviceToHost));
    CheckClose(routed_actual, routed_expected,
               std::string(quant.slug) + " routed");

    CudaOk(cudaFree(weights));
  }

  CudaOk(cudaFree(scratch));
  CudaOk(cudaFree(routed_output));
  CudaOk(cudaFree(dense_output));
  CudaOk(cudaFree(ids));
  CudaOk(cudaFree(input));
  return 0;
}

#include "q27_prefill_fp8.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kM = 128;

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

std::vector<uint8_t> Read(const std::string& path, uint64_t expected_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open " + path);
  const auto bytes = input.tellg();
  if (bytes < 0 || static_cast<uint64_t>(bytes) != expected_bytes)
    throw std::runtime_error("byte-size mismatch for " + path);
  std::vector<uint8_t> data(expected_bytes);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(data.data()), bytes);
  if (!input) throw std::runtime_error("cannot read " + path);
  return data;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes) {
    Cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  void* data() const { return data_; }
  uint64_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  uint64_t bytes_ = 0;
};

float Bf16(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::runtime_error(
          "usage: q27-prefill-fp8-parity DIR [qkvz|gdn-out]");
    const std::string root = argv[1];
    const std::string projection = argc == 3 ? argv[2] : "qkvz";
    const uint32_t n = projection == "qkvz" ? 16384 : 5120;
    const uint32_t k = projection == "qkvz" ? 5120 : 6144;
    if (projection != "qkvz" && projection != "gdn-out")
      throw std::runtime_error("projection must be qkvz or gdn-out");
    q27_prefill_fp8_shape shape{sizeof(shape), Q27_PREFILL_FP8_ABI_VERSION};
    q27_prefill_fp8_status status =
        q27_prefill_fp8_query(kM, n, k, &shape);
    if (status.code != Q27_PREFILL_FP8_OK)
      throw std::runtime_error(status.message);

    const auto input = Read(root + "/input.bf16", shape.input_bf16_bytes);
    const auto expected_quantized =
        Read(root + "/quantized_input.fp8", shape.quantized_input_bytes);
    const auto weight = Read(root + "/weight.fp8", shape.packed_weight_bytes);
    const auto expected = Read(root + "/expected.bf16", shape.output_bf16_bytes);
    const auto scales = Read(root + "/scales.f32le", 2 * sizeof(float));

    DeviceBuffer d_input(input.size());
    DeviceBuffer d_quantized(expected_quantized.size());
    DeviceBuffer d_weight(weight.size());
    DeviceBuffer d_output(expected.size());
    DeviceBuffer d_scales(scales.size());
    DeviceBuffer d_workspace(shape.workspace_bytes);
    Cuda(cudaMemcpy(d_input.data(), input.data(), input.size(), cudaMemcpyHostToDevice),
         "copy input");
    Cuda(cudaMemcpy(d_weight.data(), weight.data(), weight.size(), cudaMemcpyHostToDevice),
         "copy weight");
    Cuda(cudaMemcpy(d_scales.data(), scales.data(), scales.size(), cudaMemcpyHostToDevice),
         "copy scales");

    q27_prefill_fp8_plan_config config{};
    config.struct_size = sizeof(config);
    config.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
    config.m = kM;
    config.n = n;
    config.k = k;
    config.workspace_bytes = shape.workspace_bytes;
    q27_prefill_fp8_plan* plan = nullptr;
    status = q27_prefill_fp8_plan_create(&config, &plan);
    if (status.code != Q27_PREFILL_FP8_OK)
      throw std::runtime_error(status.message);

    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    q27_prefill_fp8_project_args args{};
    args.struct_size = sizeof(args);
    args.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
    args.input_bf16 = d_input.data();
    args.input_bf16_bytes = d_input.bytes();
    args.input_scale = static_cast<const float*>(d_scales.data());
    args.weight_fp8_e4m3 = d_weight.data();
    args.packed_weight_bytes = d_weight.bytes();
    args.weight_scale = static_cast<const float*>(d_scales.data()) + 1;
    args.quantized_input_fp8_e4m3 = d_quantized.data();
    args.quantized_input_bytes = d_quantized.bytes();
    args.output_bf16 = d_output.data();
    args.output_bf16_bytes = d_output.bytes();
    args.workspace = d_workspace.data();
    args.workspace_bytes = d_workspace.bytes();
    args.cuda_stream = stream;
    status = q27_prefill_fp8_project(plan, &args);
    if (status.code != Q27_PREFILL_FP8_OK)
      throw std::runtime_error(status.message);
    Cuda(cudaStreamSynchronize(stream), "project synchronize");

    std::vector<uint8_t> actual_quantized(expected_quantized.size());
    std::vector<uint8_t> actual(expected.size());
    Cuda(cudaMemcpy(actual_quantized.data(), d_quantized.data(),
                    actual_quantized.size(), cudaMemcpyDeviceToHost),
         "copy quantized input");
    Cuda(cudaMemcpy(actual.data(), d_output.data(), actual.size(),
                    cudaMemcpyDeviceToHost),
         "copy output");

    uint64_t quantized_mismatch = 0;
    for (size_t index = 0; index < actual_quantized.size(); ++index)
      quantized_mismatch += actual_quantized[index] != expected_quantized[index];
    uint64_t output_mismatch = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double dot = 0.0;
    double actual_square = 0.0;
    double expected_square = 0.0;
    const auto* actual_bf16 = reinterpret_cast<const uint16_t*>(actual.data());
    const auto* expected_bf16 = reinterpret_cast<const uint16_t*>(expected.data());
    const uint64_t elements = shape.output_bf16_bytes / 2;
    for (uint64_t index = 0; index < elements; ++index) {
      output_mismatch += actual_bf16[index] != expected_bf16[index];
      const double a = Bf16(actual_bf16[index]);
      const double e = Bf16(expected_bf16[index]);
      const double error = std::abs(a - e);
      max_abs = std::max(max_abs, error);
      mean_abs += error;
      dot += a * e;
      actual_square += a * a;
      expected_square += e * e;
    }
    mean_abs /= elements;
    const double cosine = dot / std::sqrt(actual_square * expected_square);
    const bool passed = quantized_mismatch == 0 && max_abs <= 0.0625 &&
                        mean_abs <= 0.002 && cosine >= 0.99999;
    std::cout << "q27_prefill_fp8_parity projection=" << projection
              << " m=" << kM
              << " quantized_mismatch=" << quantized_mismatch
              << " output_mismatch=" << output_mismatch << '/' << elements
              << " max_abs=" << max_abs << " mean_abs=" << mean_abs
              << " cosine=" << cosine << " pass=" << (passed ? "true" : "false")
              << '\n';

    cudaStreamDestroy(stream);
    q27_prefill_fp8_plan_destroy(plan);
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "q27_prefill_fp8_parity: FAIL: " << error.what() << '\n';
    return 1;
  }
}

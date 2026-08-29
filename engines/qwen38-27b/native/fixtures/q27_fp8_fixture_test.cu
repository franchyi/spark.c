#include "q27_kernels.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct Header {
  char magic[8];
  uint32_t version;
  uint32_t reserved;
  uint64_t n;
  uint64_t k;
  float input_scale;
  float weight_scale;
  uint64_t input_bytes;
  uint64_t quantized_bytes;
  uint64_t weight_bytes;
  uint64_t output_bytes;
};
#pragma pack(pop)

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}

std::vector<uint8_t> Read(std::ifstream& input, uint64_t bytes) {
  std::vector<uint8_t> value(bytes);
  input.read(reinterpret_cast<char*>(value.data()), bytes);
  assert(input.good());
  return value;
}

void* DeviceCopy(const void* source, size_t bytes) {
  void* output = nullptr;
  CudaOk(cudaMalloc(&output, bytes));
  CudaOk(cudaMemcpy(output, source, bytes, cudaMemcpyHostToDevice));
  return output;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  std::ifstream fixture(argv[1], std::ios::binary);
  assert(fixture.good());
  Header header = {};
  fixture.read(reinterpret_cast<char*>(&header), sizeof(header));
  assert(std::string(header.magic, 8) == "Q27FP8V1");
  assert(header.version == 1 && header.n == 10240 && header.k == 5120);
  auto input = Read(fixture, header.input_bytes);
  auto expected_quantized = Read(fixture, header.quantized_bytes);
  auto weight = Read(fixture, header.weight_bytes);
  auto expected_output = Read(fixture, header.output_bytes);

  void* device_input = DeviceCopy(input.data(), input.size());
  void* device_weight = DeviceCopy(weight.data(), weight.size());
  float* device_input_scale =
      static_cast<float*>(DeviceCopy(&header.input_scale, sizeof(float)));
  float* device_weight_scale =
      static_cast<float*>(DeviceCopy(&header.weight_scale, sizeof(float)));
  void* device_quantized = nullptr;
  void* device_output = nullptr;
  CudaOk(cudaMalloc(&device_quantized, expected_quantized.size()));
  CudaOk(cudaMalloc(&device_output, expected_output.size()));

  q27_fp8_project_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_KERNEL_ABI_VERSION;
  args.n = header.n;
  args.k = header.k;
  args.input_bf16 = device_input;
  args.weight_fp8_e4m3 = device_weight;
  args.input_scale = device_input_scale;
  args.weight_scale = device_weight_scale;
  args.quantized_input_fp8_e4m3 = device_quantized;
  args.output_bf16 = device_output;
  q27_kernel_status status = q27_fp8_project(&args);
  if (status.code != Q27_KERNEL_OK) {
    std::fprintf(stderr, "q27_fp8_project: %s\n", status.message);
    return 1;
  }
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> actual_quantized(expected_quantized.size());
  std::vector<uint8_t> actual_output(expected_output.size());
  CudaOk(cudaMemcpy(actual_quantized.data(), device_quantized,
                    actual_quantized.size(), cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_output.data(), device_output, actual_output.size(),
                    cudaMemcpyDeviceToHost));
  assert(actual_quantized == expected_quantized);
  assert(actual_output == expected_output);
  std::printf("q27 FP8 layer0 QKV fixture: exact (%lu x %lu)\n", header.n,
              header.k);

  for (int iteration = 0; iteration < 10; ++iteration) {
    status = q27_fp8_project(&args);
    assert(status.code == Q27_KERNEL_OK);
  }
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  CudaOk(cudaEventRecord(begin));
  constexpr int kIterations = 100;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = q27_fp8_project(&args);
    assert(status.code == Q27_KERNEL_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  const double mean_us = elapsed_ms * 1000.0 / kIterations;
  const double weight_gb_s =
      static_cast<double>(header.weight_bytes) / mean_us / 1000.0;
  std::printf("q27 FP8 layer0 QKV mean_us=%.3f weight_gb_s=%.2f\n", mean_us,
              weight_gb_s);
  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));

  CudaOk(cudaFree(device_output));
  CudaOk(cudaFree(device_quantized));
  CudaOk(cudaFree(device_weight_scale));
  CudaOk(cudaFree(device_input_scale));
  CudaOk(cudaFree(device_weight));
  CudaOk(cudaFree(device_input));
  return 0;
}

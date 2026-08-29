#include "q27_bf16_ab.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  uint64_t input_bytes;
  uint64_t weight_a_bytes;
  uint64_t weight_b_bytes;
  uint64_t output_a_bytes;
  uint64_t output_b_bytes;
};
#pragma pack(pop)

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}

void CublasOk(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS: %s\n", cublasGetStatusString(status));
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

void AssertExact(const char* name, const std::vector<uint8_t>& expected,
                 const void* device) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(),
                    cudaMemcpyDeviceToHost));
  if (actual != expected) {
    size_t mismatches = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
      mismatches += actual[index] != expected[index];
    }
    std::fprintf(stderr, "%s is not byte-exact: %zu/%zu byte mismatches\n",
                 name, mismatches, actual.size());
    std::abort();
  }
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  std::ifstream fixture(argv[1], std::ios::binary);
  assert(fixture.good());
  Header header = {};
  fixture.read(reinterpret_cast<char*>(&header), sizeof(header));
  assert(std::string(header.magic, 8) == std::string("Q27ABV1\0", 8));
  assert(header.version == 1 && header.n == 48 && header.k == 5120);
  auto input = Read(fixture, header.input_bytes);
  auto weight_a = Read(fixture, header.weight_a_bytes);
  auto weight_b = Read(fixture, header.weight_b_bytes);
  auto expected_a = Read(fixture, header.output_a_bytes);
  auto expected_b = Read(fixture, header.output_b_bytes);

  void* device_input = DeviceCopy(input.data(), input.size());
  void* device_weight_a = DeviceCopy(weight_a.data(), weight_a.size());
  void* device_weight_b = DeviceCopy(weight_b.data(), weight_b.size());
  void* device_output_a = nullptr;
  void* device_output_b = nullptr;
  CudaOk(cudaMalloc(&device_output_a, expected_a.size()));
  CudaOk(cudaMalloc(&device_output_b, expected_b.size()));
  cudaStream_t stream = nullptr;
  CudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle = nullptr;
  CublasOk(cublasCreate(&handle));

  q27_bf16_ab_project_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_KERNEL_ABI_VERSION;
  args.hidden_size = header.k;
  args.value_heads = header.n;
  args.input_bf16 = device_input;
  args.weight_a_bf16 = device_weight_a;
  args.weight_b_bf16 = device_weight_b;
  args.output_a_bf16 = device_output_a;
  args.output_b_bf16 = device_output_b;
  args.cublas_handle = handle;
  args.cuda_stream = stream;
  q27_kernel_status status = q27_bf16_ab_project(&args);
  if (status.code != Q27_KERNEL_OK) {
    std::fprintf(stderr, "q27_bf16_ab_project: %s\n", status.message);
    return 1;
  }
  CudaOk(cudaDeviceSynchronize());
  AssertExact("in_proj_a", expected_a, device_output_a);
  AssertExact("in_proj_b", expected_b, device_output_b);
  std::printf("q27 BF16 layer0 a/b fixture: byte-exact (48 x 5120, twice)\n");

  for (int iteration = 0; iteration < 20; ++iteration) {
    status = q27_bf16_ab_project(&args);
    assert(status.code == Q27_KERNEL_OK);
  }
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  CudaOk(cudaEventRecord(begin, stream));
  constexpr int kIterations = 1000;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = q27_bf16_ab_project(&args);
    assert(status.code == Q27_KERNEL_OK);
  }
  CudaOk(cudaEventRecord(end, stream));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  const double mean_us = elapsed_ms * 1000.0 / kIterations;
  const double weight_bytes = header.weight_a_bytes + header.weight_b_bytes;
  std::printf("q27 BF16 layer0 a/b mean_us=%.3f weight_gb_s=%.2f\n", mean_us,
              weight_bytes / mean_us / 1000.0);

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CublasOk(cublasDestroy(handle));
  CudaOk(cudaStreamDestroy(stream));
  CudaOk(cudaFree(device_output_b));
  CudaOk(cudaFree(device_output_a));
  CudaOk(cudaFree(device_weight_b));
  CudaOk(cudaFree(device_weight_a));
  CudaOk(cudaFree(device_input));
  return 0;
}

#include "sparkserve/kernel_api.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

template <typename T>
std::vector<T> Read(const std::filesystem::path& path, size_t elements) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  assert(stream.good());
  assert(static_cast<size_t>(stream.tellg()) == elements * sizeof(T));
  stream.seekg(0);
  std::vector<T> data(elements);
  stream.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
  assert(stream.good());
  return data;
}

template <typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CudaOk(cudaMalloc(&device, host.size() * sizeof(T)));
  CudaOk(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
  return device;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  constexpr uint32_t kRows = 6;
  constexpr uint32_t kBlockTopk = 512;
  constexpr uint32_t kFinalTopk = 2051;
  const auto blocks =
      Read<int32_t>(fixture / "block_indices_i32.bin", kRows * kBlockTopk);
  const auto positions =
      Read<int64_t>(fixture / "query_positions_i64.bin", kRows);
  const auto lengths =
      Read<int32_t>(fixture / "sequence_lengths_i32.bin", kRows);
  const auto expected =
      Read<int32_t>(fixture / "logical_indices_i32.bin", kRows * kFinalTopk);

  int32_t* blocks_device = Upload(blocks);
  int64_t* positions_device = Upload(positions);
  int32_t* lengths_device = Upload(lengths);
  int32_t* output_device = nullptr;
  CudaOk(cudaMalloc(&output_device, expected.size() * sizeof(int32_t)));

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeQsaExpandPlan plan = {
      sizeof(SparkServeQsaExpandPlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kRows,
      kBlockTopk,
      4,
      2048,
      kFinalTopk,
      SPARKSERVE_DTYPE_INT32,
      SPARKSERVE_BACKEND_SGLANG_QSA_EXPAND,
      0,
  };
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_qsa_expand_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.available == 1);

  SparkServeQsaExpandArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.block_indices = blocks_device;
  args.query_positions = positions_device;
  args.sequence_lengths = lengths_device;
  args.logical_indices = output_device;
  SparkServeStatus status = sparkserve_qsa_expand_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<int32_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), output_device,
                    actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != expected[index];
  }
  std::cout << "SGLang QSA block expansion mismatches: " << mismatches << '\n';
  assert(mismatches == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 1000;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = sparkserve_qsa_expand_launch(&caps, &args);
    assert(status.code == SPARKSERVE_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "QSA block expansion 6x2051 mean: "
            << elapsed_ms * 1000.0f / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaFree(output_device));
  CudaOk(cudaFree(lengths_device));
  CudaOk(cudaFree(positions_device));
  CudaOk(cudaFree(blocks_device));
  return 0;
}

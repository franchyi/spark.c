#include "sparkserve/kernel_api.h"

#include <cuda_runtime.h>

#include <algorithm>
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
  constexpr uint32_t kRows = 4;
  constexpr uint32_t kColumns = 65'536;
  constexpr uint32_t kTopK = 512;
  const auto scores = Read<float>(fixture / "scores_f32.bin", kRows * kColumns);
  const auto row_starts = Read<int32_t>(fixture / "row_starts_i32.bin", kRows);
  const auto lengths = Read<int32_t>(fixture / "lengths_i32.bin", kRows);
  const auto expected =
      Read<int32_t>(fixture / "indices_i32.bin", kRows * kTopK);

  float* scores_device = Upload(scores);
  int32_t* starts_device = Upload(row_starts);
  int32_t* lengths_device = Upload(lengths);
  int32_t* indices_device = nullptr;
  CudaOk(cudaMalloc(&indices_device, expected.size() * sizeof(int32_t)));

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeQsaTopkPlan plan = {
      sizeof(SparkServeQsaTopkPlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kRows,
      kColumns,
      kTopK,
      SPARKSERVE_DTYPE_F32,
      SPARKSERVE_DTYPE_INT32,
      SPARKSERVE_BACKEND_SGLANG_QSA_TOPK,
      kColumns,
  };
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_qsa_topk_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.available == 1);

  SparkServeQsaTopkArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.scores = scores_device;
  args.row_starts = starts_device;
  args.lengths = lengths_device;
  args.indices = indices_device;
  SparkServeStatus status = sparkserve_qsa_topk_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<int32_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), indices_device,
                    actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  size_t mismatched_rows = 0;
  for (uint32_t row = 0; row < kRows; ++row) {
    auto actual_begin = actual.begin() + row * kTopK;
    auto expected_begin = expected.begin() + row * kTopK;
    std::sort(actual_begin, actual_begin + kTopK);
    std::vector<int32_t> sorted_expected(expected_begin,
                                         expected_begin + kTopK);
    std::sort(sorted_expected.begin(), sorted_expected.end());
    mismatched_rows += !std::equal(actual_begin, actual_begin + kTopK,
                                   sorted_expected.begin());
  }
  std::cout << "SGLang QSA top-k selected-set mismatched rows: "
            << mismatched_rows << '\n';
  assert(mismatched_rows == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 100;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = sparkserve_qsa_topk_launch(&caps, &args);
    assert(status.code == SPARKSERVE_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "QSA radix top-k 4x65536 mean: "
            << elapsed_ms * 1000.0f / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaFree(indices_device));
  CudaOk(cudaFree(lengths_device));
  CudaOk(cudaFree(starts_device));
  CudaOk(cudaFree(scores_device));
  return 0;
}

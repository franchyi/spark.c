#include "flash/kernel_api.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
  constexpr uint32_t kBatch = 3;
  constexpr uint32_t kHeads = 8;
  constexpr uint32_t kHeadDim = 128;
  constexpr uint32_t kPages = 41;
  constexpr uint32_t kPageSize = 16;
  constexpr uint32_t kMaxPages = 17;
  constexpr uint32_t kMaxModelLen = kMaxPages * kPageSize;

  const auto query =
      Read<uint16_t>(fixture / "q_bf16.bin", kBatch * kHeads * kHeadDim);
  const auto key_cache = Read<uint16_t>(
      fixture / "k_cache_bf16.bin", kPages * kPageSize * kHeadDim);
  const auto page_table =
      Read<int32_t>(fixture / "page_table_i32.bin", kBatch * kMaxPages);
  const auto context_lengths =
      Read<int32_t>(fixture / "context_lens_i32.bin", kBatch);
  const auto expected =
      Read<float>(fixture / "logits_f32.bin", kBatch * kMaxModelLen);

  uint16_t* query_device = Upload(query);
  uint16_t* key_cache_device = Upload(key_cache);
  int32_t* page_table_device = Upload(page_table);
  int32_t* context_lengths_device = Upload(context_lengths);
  std::vector<float> sentinel(expected.size(),
                              std::numeric_limits<float>::quiet_NaN());
  float* logits_device = Upload(sentinel);

  FlashDeviceCaps caps = {
      sizeof(FlashDeviceCaps), FLASH_KERNEL_ABI_VERSION, 121, 1, 0};
  FlashQsaScorePlan plan = {
      sizeof(FlashQsaScorePlan),
      FLASH_KERNEL_ABI_VERSION,
      kBatch,
      kPages,
      kMaxPages,
      kMaxModelLen,
      kHeads,
      kHeadDim,
      kPageSize,
      FLASH_DTYPE_BF16,
      FLASH_DTYPE_F32,
      FLASH_BACKEND_TILELANG_QSA_SCORE,
  };
  FlashKernelInfo info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_score_query(&caps, &plan, &info).code ==
         FLASH_STATUS_OK);
  assert(info.available == 1);

  FlashQsaScoreArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.query = query_device;
  args.key_cache = key_cache_device;
  args.page_table = page_table_device;
  args.context_lengths = context_lengths_device;
  args.logits = logits_device;
  args.score_scale = std::sqrt(static_cast<float>(kHeadDim));
  FlashStatus status = flash_qsa_score_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<float> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), logits_device,
                    actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  size_t compared = 0;
  for (uint32_t row = 0; row < kBatch; ++row) {
    for (int32_t column = 0; column < context_lengths[row]; ++column) {
      const size_t index = static_cast<size_t>(row) * kMaxModelLen + column;
      mismatches += actual[index] != expected[index];
      ++compared;
    }
  }
  std::cout << "SGLang/TileLang QSA score mismatches: " << mismatches << '/'
            << compared << '\n';
  assert(mismatches == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 1000;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = flash_qsa_score_launch(&caps, &args);
    assert(status.code == FLASH_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "QSA score 3x272 mean: "
            << elapsed_ms * 1000.0f / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaFree(logits_device));
  CudaOk(cudaFree(context_lengths_device));
  CudaOk(cudaFree(page_table_device));
  CudaOk(cudaFree(key_cache_device));
  CudaOk(cudaFree(query_device));
  return 0;
}

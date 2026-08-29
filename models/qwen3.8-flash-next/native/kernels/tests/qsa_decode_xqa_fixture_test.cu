#include "flash/kernel_api.h"

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
  constexpr uint32_t kBatch = 1;
  constexpr uint32_t kSmCount = 48;
  constexpr uint32_t kQueryHeads = 24;
  constexpr uint32_t kKvHeads = 2;
  constexpr uint32_t kHeadDim = 256;
  constexpr uint32_t kPageSize = 64;
  constexpr uint32_t kPagesPerRow = 33;
  constexpr uint32_t kPackedRowStride = kPageSize * kPagesPerRow;
  constexpr uint64_t kWorkspaceBytes = 128ULL * 1024 * 1024;
  const size_t query_elements =
      static_cast<size_t>(kBatch) * kQueryHeads * kHeadDim;
  const size_t packed_elements =
      static_cast<size_t>(kBatch) * kPackedRowStride * kKvHeads * kHeadDim;

  const auto query =
      Read<uint16_t>(fixture / "query_bf16.bin", query_elements);
  const auto packed_key =
      Read<uint16_t>(fixture / "packed_key_bf16.bin", packed_elements);
  const auto packed_value =
      Read<uint16_t>(fixture / "packed_value_bf16.bin", packed_elements);
  const auto block_tables = Read<int32_t>(
      fixture / "block_tables_i32.bin", kBatch * kPagesPerRow);
  const auto sequence_lengths =
      Read<int32_t>(fixture / "sequence_lengths_i32.bin", kBatch);
  const auto expected =
      Read<uint16_t>(fixture / "output_bf16.bin", query_elements);

  uint16_t* query_device = Upload(query);
  uint16_t* key_device = Upload(packed_key);
  uint16_t* value_device = Upload(packed_value);
  int32_t* block_tables_device = Upload(block_tables);
  int32_t* sequence_lengths_device = Upload(sequence_lengths);
  uint16_t* output_device = nullptr;
  void* workspace_device = nullptr;
  CudaOk(cudaMalloc(&output_device, query_elements * sizeof(uint16_t)));
  CudaOk(cudaMalloc(&workspace_device, kWorkspaceBytes));
  CudaOk(cudaMemset(workspace_device, 0, kWorkspaceBytes));

  FlashDeviceCaps caps = {sizeof(FlashDeviceCaps),
                               FLASH_KERNEL_ABI_VERSION,
                               121,
                               1,
                               kWorkspaceBytes};
  FlashQsaDecodePlan plan = {
      sizeof(FlashQsaDecodePlan),
      FLASH_KERNEL_ABI_VERSION,
      kBatch,
      kSmCount,
      kQueryHeads,
      kKvHeads,
      kHeadDim,
      kPageSize,
      kPagesPerRow,
      kPackedRowStride,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_FLASHINFER_XQA_DECODE,
      1,
      0,
  };
  FlashKernelInfo info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_decode_query(&caps, &plan, &info).code ==
         FLASH_STATUS_OK);
  assert(info.available == 1);
  assert(info.workspace_bytes == kWorkspaceBytes);

  FlashQsaDecodeArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.query = query_device;
  args.packed_key = key_device;
  args.packed_value = value_device;
  args.block_tables = block_tables_device;
  args.sequence_lengths = sequence_lengths_device;
  args.output = output_device;
  args.workspace = workspace_device;
  args.workspace_bytes = kWorkspaceBytes;
  args.bmm1_scale = 0.0625f;
  args.bmm2_scale = 1.0f;
  FlashStatus status = flash_qsa_decode_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint16_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), output_device,
                    actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != expected[index];
  }
  std::cout << "FlashInfer XQA QSA BF16 mismatches: " << mismatches << '\n';
  assert(mismatches == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 200;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = flash_qsa_decode_launch(&caps, &args);
    assert(status.code == FLASH_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "FlashInfer XQA QSA batch-1 mean: "
            << elapsed_ms * 1000.0f / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaFree(workspace_device));
  CudaOk(cudaFree(output_device));
  CudaOk(cudaFree(sequence_lengths_device));
  CudaOk(cudaFree(block_tables_device));
  CudaOk(cudaFree(value_device));
  CudaOk(cudaFree(key_device));
  CudaOk(cudaFree(query_device));
  return 0;
}

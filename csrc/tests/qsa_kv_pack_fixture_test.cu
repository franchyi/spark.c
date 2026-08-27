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

template <typename T>
size_t Mismatches(const T* device, const std::vector<T>& expected) {
  std::vector<T> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size() * sizeof(T),
                    cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != expected[index];
  }
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  constexpr uint32_t kBatch = 4;
  constexpr uint32_t kSlotCapacity = 8192;
  constexpr uint32_t kRequestCapacity = 6;
  constexpr uint32_t kRequestStride = 4096;
  constexpr uint32_t kTopk = 2051;
  constexpr uint32_t kPackedRowStride = 2112;
  constexpr uint32_t kKvHeads = 2;
  constexpr uint32_t kHeadDim = 256;
  const size_t state_elements =
      static_cast<size_t>(kSlotCapacity) * kKvHeads * kHeadDim;
  const size_t packed_elements =
      static_cast<size_t>(kBatch) * kPackedRowStride * kKvHeads * kHeadDim;

  const auto key_state =
      Read<uint16_t>(fixture / "key_state_bf16.bin", state_elements);
  const auto value_state =
      Read<uint16_t>(fixture / "value_state_bf16.bin", state_elements);
  const auto req_to_token = Read<int32_t>(
      fixture / "req_to_token_i32.bin", kRequestCapacity * kRequestStride);
  const auto request_indices =
      Read<int32_t>(fixture / "request_indices_i32.bin", kBatch);
  const auto logical_indices =
      Read<int32_t>(fixture / "logical_indices_i32.bin", kBatch * kTopk);
  const auto sequence_lengths =
      Read<int32_t>(fixture / "sequence_lengths_i32.bin", kBatch);
  const auto expected_counts =
      Read<int32_t>(fixture / "valid_counts_i32.bin", kBatch);
  const auto expected_key =
      Read<uint16_t>(fixture / "packed_key_bf16.bin", packed_elements);
  const auto expected_value =
      Read<uint16_t>(fixture / "packed_value_bf16.bin", packed_elements);

  uint16_t* key_device = Upload(key_state);
  uint16_t* value_device = Upload(value_state);
  int32_t* req_to_token_device = Upload(req_to_token);
  int32_t* request_indices_device = Upload(request_indices);
  int32_t* logical_indices_device = Upload(logical_indices);
  int32_t* sequence_lengths_device = Upload(sequence_lengths);
  int32_t* counts_device = nullptr;
  uint16_t* packed_key_device = nullptr;
  uint16_t* packed_value_device = nullptr;
  CudaOk(cudaMalloc(&counts_device, kBatch * sizeof(int32_t)));
  CudaOk(cudaMalloc(&packed_key_device, packed_elements * sizeof(uint16_t)));
  CudaOk(cudaMalloc(&packed_value_device, packed_elements * sizeof(uint16_t)));
  CudaOk(cudaMemset(packed_key_device, 0, packed_elements * sizeof(uint16_t)));
  CudaOk(cudaMemset(packed_value_device, 0, packed_elements * sizeof(uint16_t)));

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeQsaKvPackPlan plan = {
      sizeof(SparkServeQsaKvPackPlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kBatch,
      kSlotCapacity,
      kRequestCapacity,
      kRequestStride,
      kTopk,
      kPackedRowStride,
      kKvHeads,
      kHeadDim,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_SGLANG_QSA_KV_PACK,
  };
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_qsa_kv_pack_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.available == 1);

  SparkServeQsaKvPackArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.key_state = key_device;
  args.value_state = value_device;
  args.req_to_token = req_to_token_device;
  args.request_indices = request_indices_device;
  args.logical_indices = logical_indices_device;
  args.sequence_lengths = sequence_lengths_device;
  args.valid_counts = counts_device;
  args.packed_key = packed_key_device;
  args.packed_value = packed_value_device;
  SparkServeStatus status = sparkserve_qsa_kv_pack_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  const size_t count_mismatches = Mismatches(counts_device, expected_counts);
  const size_t key_mismatches = Mismatches(packed_key_device, expected_key);
  const size_t value_mismatches =
      Mismatches(packed_value_device, expected_value);
  std::cout << "QSA K/V-pack mismatches counts/key/value: "
            << count_mismatches << '/' << key_mismatches << '/'
            << value_mismatches << '\n';
  assert(count_mismatches == 0);
  assert(key_mismatches == 0);
  assert(value_mismatches == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 100;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = sparkserve_qsa_kv_pack_launch(&caps, &args);
    assert(status.code == SPARKSERVE_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "QSA valid-count + 4-row BF16 K/V pack mean: "
            << elapsed_ms * 1000.0f / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaFree(packed_value_device));
  CudaOk(cudaFree(packed_key_device));
  CudaOk(cudaFree(counts_device));
  CudaOk(cudaFree(sequence_lengths_device));
  CudaOk(cudaFree(logical_indices_device));
  CudaOk(cudaFree(request_indices_device));
  CudaOk(cudaFree(req_to_token_device));
  CudaOk(cudaFree(value_device));
  CudaOk(cudaFree(key_device));
  return 0;
}

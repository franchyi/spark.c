#include "sparkserve/kernel_api.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstring>
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
  constexpr uint32_t kTokens = 37;
  constexpr uint32_t kGroups = 9;
  constexpr uint32_t kStateSlots = 128;
  constexpr uint32_t kCompressedSlots = 64;
  constexpr uint32_t kHeadDim = 128;
  constexpr uint32_t kQueryHeads = 4;
  constexpr uint32_t kQueryHeadsPadded = 8;
  constexpr uint32_t kPositionRows = 512;

  const auto qk = Read<uint16_t>(fixture / "qk_bf16.bin",
                                 kTokens * (kQueryHeads + 1) * kHeadDim);
  const auto q_weight =
      Read<uint16_t>(fixture / "q_weight_bf16.bin", kHeadDim);
  const auto k_weight =
      Read<uint16_t>(fixture / "k_weight_bf16.bin", kHeadDim);
  const auto cos_sin = Read<float>(fixture / "cos_sin_f32.bin",
                                   kPositionRows * kHeadDim);
  const auto axis_map = Read<int32_t>(fixture / "axis_map_i32.bin", kHeadDim / 2);
  const auto positions = Read<int64_t>(fixture / "positions_i64.bin", kTokens);
  const auto cache_locs = Read<int64_t>(fixture / "cache_locs_i64.bin", kTokens);
  const auto group_locs =
      Read<int32_t>(fixture / "group_locs_i32.bin", kGroups * 4);
  const auto write_locs = Read<int32_t>(fixture / "write_locs_i32.bin", kGroups);
  const auto expected_q = Read<uint16_t>(fixture / "q_output_bf16.bin",
                                         kTokens * kQueryHeadsPadded * kHeadDim);
  const auto expected_state = Read<uint16_t>(fixture / "key_state_bf16.bin",
                                             kStateSlots * kHeadDim);
  const auto expected_rope =
      Read<int64_t>(fixture / "rope_positions_i64.bin", kStateSlots * 3);
  const auto expected_compressed = Read<uint16_t>(
      fixture / "compressed_bf16.bin", kCompressedSlots * kHeadDim);

  uint16_t* qk_device = Upload(qk);
  uint16_t* q_weight_device = Upload(q_weight);
  uint16_t* k_weight_device = Upload(k_weight);
  float* cos_sin_device = Upload(cos_sin);
  int32_t* axis_map_device = Upload(axis_map);
  int64_t* positions_device = Upload(positions);
  int64_t* cache_locs_device = Upload(cache_locs);
  int32_t* group_locs_device = Upload(group_locs);
  int32_t* write_locs_device = Upload(write_locs);
  uint16_t* q_output_device = nullptr;
  uint16_t* state_device = nullptr;
  int64_t* rope_device = nullptr;
  uint16_t* compressed_device = nullptr;
  CudaOk(cudaMalloc(&q_output_device, expected_q.size() * sizeof(uint16_t)));
  CudaOk(cudaMalloc(&state_device, expected_state.size() * sizeof(uint16_t)));
  CudaOk(cudaMalloc(&rope_device, expected_rope.size() * sizeof(int64_t)));
  CudaOk(cudaMalloc(&compressed_device,
                    expected_compressed.size() * sizeof(uint16_t)));
  CudaOk(cudaMemset(state_device, 0, expected_state.size() * sizeof(uint16_t)));
  CudaOk(cudaMemset(rope_device, 0, expected_rope.size() * sizeof(int64_t)));
  CudaOk(cudaMemset(compressed_device, 0,
                    expected_compressed.size() * sizeof(uint16_t)));

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeQsaIndexPrepPlan plan = {
      sizeof(SparkServeQsaIndexPrepPlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kTokens,
      kGroups,
      kStateSlots,
      kCompressedSlots,
      kQueryHeads,
      kHeadDim,
      kHeadDim,
      4,
      1,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_SGLANG_QSA_INDEX_PREP,
      kQueryHeadsPadded,
  };
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_qsa_index_prep_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.available == 1);

  SparkServeQsaIndexPrepArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.qk = qk_device;
  args.q_output = q_output_device;
  args.q_norm_weight = q_weight_device;
  args.k_norm_weight = k_weight_device;
  args.cos_sin_cache = cos_sin_device;
  args.cos_sin_rows = kPositionRows;
  args.axis_map = axis_map_device;
  args.positions = positions_device;
  args.positions_stride = kTokens;
  args.cache_locs = cache_locs_device;
  args.key_state = state_device;
  args.rope_positions = rope_device;
  args.group_locs = group_locs_device;
  args.write_locs = write_locs_device;
  args.compressed_keys = compressed_device;
  args.eps = 1.0e-6f;
  SparkServeStatus status = sparkserve_qsa_index_prep_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  const size_t q_mismatches = Mismatches(q_output_device, expected_q);
  const size_t state_mismatches = Mismatches(state_device, expected_state);
  const size_t rope_mismatches = Mismatches(rope_device, expected_rope);
  const size_t compressed_mismatches =
      Mismatches(compressed_device, expected_compressed);
  std::cout << "QSA prep mismatches q/state/rope/compressed: " << q_mismatches
            << '/' << state_mismatches << '/' << rope_mismatches << '/'
            << compressed_mismatches << '\n';
  assert(q_mismatches == 0);
  assert(state_mismatches == 0);
  assert(rope_mismatches == 0);
  assert(compressed_mismatches == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 1000;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = sparkserve_qsa_index_prep_launch(&caps, &args);
    assert(status.code == SPARKSERVE_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "QSA prep 37 tokens + 9 groups mean: "
            << elapsed_ms * 1000.0f / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaFree(compressed_device));
  CudaOk(cudaFree(rope_device));
  CudaOk(cudaFree(state_device));
  CudaOk(cudaFree(q_output_device));
  CudaOk(cudaFree(write_locs_device));
  CudaOk(cudaFree(group_locs_device));
  CudaOk(cudaFree(cache_locs_device));
  CudaOk(cudaFree(positions_device));
  CudaOk(cudaFree(axis_map_device));
  CudaOk(cudaFree(cos_sin_device));
  CudaOk(cudaFree(k_weight_device));
  CudaOk(cudaFree(q_weight_device));
  CudaOk(cudaFree(qk_device));
  return 0;
}

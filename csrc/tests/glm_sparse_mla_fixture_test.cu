#include "sparkserve/glm_sparse_mla_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kBatch = 1;
constexpr uint32_t kTokens = 5;
constexpr uint32_t kPages = 1;
constexpr uint64_t kPageBytes =
    SPARKSERVE_GLM_SPARSE_MLA_PAGE_SIZE *
    SPARKSERVE_GLM_SPARSE_MLA_TOKEN_BYTES;

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
  assert(error == cudaSuccess);
}

void StatusOk(SparkServeStatus status) {
  if (status.code != SPARKSERVE_STATUS_OK) std::fprintf(stderr, "%s\n", status.message);
  assert(status.code == SPARKSERVE_STATUS_OK);
}

uint16_t FloatToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename T>
T* DeviceCopy(const std::vector<T>& input) {
  T* output = nullptr;
  CudaOk(cudaMalloc(&output, input.size() * sizeof(T)));
  CudaOk(cudaMemcpy(output, input.data(), input.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
  return output;
}

template <typename T>
T* DeviceAllocate(size_t count) {
  T* output = nullptr;
  CudaOk(cudaMalloc(&output, count * sizeof(T)));
  return output;
}

}  // namespace

int main() {
  int device = 0;
  CudaOk(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CudaOk(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 12 || properties.minor != 1 ||
      properties.multiProcessorCount !=
          static_cast<int>(SPARKSERVE_GLM_SPARSE_MLA_GB10_SMS)) {
    std::fprintf(stderr, "GLM sparse-MLA fixture requires 48-SM GB10/SM121\n");
    return 77;
  }

  std::vector<uint16_t> latent(kTokens * SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM);
  for (uint32_t token = 0; token < kTokens; ++token) {
    std::fill_n(latent.begin() +
                    static_cast<size_t>(token) *
                        SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM,
                SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM,
                FloatToBf16(static_cast<float>(token + 1)));
  }
  const std::vector<int32_t> locations = {0, 1, 2, 3, 4};
  uint16_t* latent_device = DeviceCopy(latent);
  int32_t* locations_device = DeviceCopy(locations);
  uint8_t* cache_device = DeviceAllocate<uint8_t>(kPageBytes);
  CudaOk(cudaMemset(cache_device, 0xa5, kPageBytes));

  SparkServeGlmSparseMlaPackKvArgs pack{};
  pack.struct_size = sizeof(pack);
  pack.abi_version = SPARKSERVE_GLM_SPARSE_MLA_ABI_VERSION;
  pack.tokens = kTokens;
  pack.page_size = SPARKSERVE_GLM_SPARSE_MLA_PAGE_SIZE;
  pack.latent_dim = SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM;
  pack.quant_group = 128;
  pack.num_pages = kPages;
  pack.input_bf16 = latent_device;
  pack.locations = locations_device;
  pack.cache = cache_device;
  pack.page_stride_bytes = kPageBytes;
  StatusOk(sparkserve_glm_sparse_mla_pack_kv_validate(&pack));
  StatusOk(sparkserve_glm_sparse_mla_pack_kv_launch(&pack));

  std::vector<uint16_t> query(
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS *
          SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM,
      FloatToBf16(0.0f));
  uint16_t* query_device = DeviceCopy(query);
  uint16_t* padded_query_device = DeviceAllocate<uint16_t>(
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS *
      SPARKSERVE_GLM_SPARSE_MLA_PADDED_Q_DIM);
  SparkServeGlmSparseMlaPadQueryArgs pad{};
  pad.struct_size = sizeof(pad);
  pad.abi_version = SPARKSERVE_GLM_SPARSE_MLA_ABI_VERSION;
  pad.batch_size = kBatch;
  pad.num_heads = SPARKSERVE_GLM_SPARSE_MLA_HEADS;
  pad.input_dim = SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM;
  pad.padded_dim = SPARKSERVE_GLM_SPARSE_MLA_PADDED_Q_DIM;
  pad.input_bf16 = query_device;
  pad.output_bf16 = padded_query_device;
  StatusOk(sparkserve_glm_sparse_mla_pad_query_validate(&pad));
  StatusOk(sparkserve_glm_sparse_mla_pad_query_launch(&pad));

  std::vector<int32_t> selected(SPARKSERVE_GLM_SPARSE_MLA_SELECTION_WIDTH, -1);
  for (uint32_t token = 0; token < kTokens; ++token) selected[token] = token;
  const std::vector<int64_t> query_positions = {4};
  const std::vector<int32_t> sequence_lengths = {5};
  int32_t* selected_device = DeviceCopy(selected);
  int64_t* query_positions_device = DeviceCopy(query_positions);
  int32_t* sequence_lengths_device = DeviceCopy(sequence_lengths);
  int32_t* history_indices_device =
      DeviceAllocate<int32_t>(kBatch * SPARKSERVE_GLM_SPARSE_MLA_HISTORY_TOPK);
  int32_t* tail_indices_device =
      DeviceAllocate<int32_t>(kBatch * SPARKSERVE_GLM_SPARSE_MLA_TAIL_TOPK);
  int32_t* history_lengths_device = DeviceAllocate<int32_t>(kBatch);
  int32_t* tail_lengths_device = DeviceAllocate<int32_t>(kBatch);

  const size_t history_mid_count =
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS *
      SPARKSERVE_GLM_SPARSE_MLA_HISTORY_SPLITS *
      SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM;
  const size_t history_mid_lse_count =
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS *
      SPARKSERVE_GLM_SPARSE_MLA_HISTORY_SPLITS;
  const size_t tail_mid_count =
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS *
      SPARKSERVE_GLM_SPARSE_MLA_TAIL_SPLITS *
      SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM;
  const size_t tail_mid_lse_count =
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS *
      SPARKSERVE_GLM_SPARSE_MLA_TAIL_SPLITS;
  const size_t output_count =
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS *
      SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM;
  const size_t output_lse_count =
      kBatch * SPARKSERVE_GLM_SPARSE_MLA_HEADS;
  uint16_t* history_mid_out_device = DeviceAllocate<uint16_t>(history_mid_count);
  float* history_mid_lse_device = DeviceAllocate<float>(history_mid_lse_count);
  uint16_t* output_device = DeviceAllocate<uint16_t>(output_count);
  float* output_lse_device = DeviceAllocate<float>(output_lse_count);
  uint16_t* tail_mid_out_device = DeviceAllocate<uint16_t>(tail_mid_count);
  float* tail_mid_lse_device = DeviceAllocate<float>(tail_mid_lse_count);
  uint16_t* tail_output_device = DeviceAllocate<uint16_t>(output_count);
  float* tail_output_lse_device = DeviceAllocate<float>(output_lse_count);

  SparkServeGlmSparseMlaDecodeArgs decode{};
  decode.struct_size = sizeof(decode);
  decode.abi_version = SPARKSERVE_GLM_SPARSE_MLA_ABI_VERSION;
  decode.batch_size = kBatch;
  decode.num_heads = SPARKSERVE_GLM_SPARSE_MLA_HEADS;
  decode.query_dim = SPARKSERVE_GLM_SPARSE_MLA_PADDED_Q_DIM;
  decode.value_dim = SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM;
  decode.page_size = SPARKSERVE_GLM_SPARSE_MLA_PAGE_SIZE;
  decode.history_topk = SPARKSERVE_GLM_SPARSE_MLA_HISTORY_TOPK;
  decode.tail_topk = SPARKSERVE_GLM_SPARSE_MLA_TAIL_TOPK;
  decode.history_splits = SPARKSERVE_GLM_SPARSE_MLA_HISTORY_SPLITS;
  decode.tail_splits = SPARKSERVE_GLM_SPARSE_MLA_TAIL_SPLITS;
  decode.num_pages = kPages;
  decode.num_sms = SPARKSERVE_GLM_SPARSE_MLA_GB10_SMS;
  decode.selected_stride = SPARKSERVE_GLM_SPARSE_MLA_SELECTION_WIDTH;
  decode.query_bf16 = padded_query_device;
  decode.cache = cache_device;
  decode.selected_indices = selected_device;
  decode.query_positions = query_positions_device;
  decode.sequence_lengths = sequence_lengths_device;
  decode.history_indices = history_indices_device;
  decode.tail_indices = tail_indices_device;
  decode.history_lengths = history_lengths_device;
  decode.tail_lengths = tail_lengths_device;
  decode.history_mid_out_bf16 = history_mid_out_device;
  decode.history_mid_lse = history_mid_lse_device;
  decode.output_bf16 = output_device;
  decode.output_lse = output_lse_device;
  decode.tail_mid_out_bf16 = tail_mid_out_device;
  decode.tail_mid_lse = tail_mid_lse_device;
  decode.tail_output_bf16 = tail_output_device;
  decode.tail_output_lse = tail_output_lse_device;
  decode.page_stride_bytes = kPageBytes;
  decode.softmax_scale = 1.0f / 16.0f;
  StatusOk(sparkserve_glm_sparse_mla_decode_validate(&decode));
  SparkServeGlmSparseMlaDecodeArgs invalid = decode;
  invalid.softmax_scale = 0.0f;
  assert(sparkserve_glm_sparse_mla_decode_validate(&invalid).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);
  StatusOk(sparkserve_glm_sparse_mla_decode_launch(&decode));
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> cache(kPageBytes);
  CudaOk(cudaMemcpy(cache.data(), cache_device, cache.size(), cudaMemcpyDeviceToHost));
  for (uint32_t token = 0; token < kTokens; ++token) {
    const uint8_t* row = cache.data() +
                         static_cast<size_t>(token) *
                             SPARKSERVE_GLM_SPARSE_MLA_TOKEN_BYTES;
    for (uint32_t byte = 528; byte < SPARKSERVE_GLM_SPARSE_MLA_TOKEN_BYTES; ++byte) {
      assert(row[byte] == 0);
    }
    for (uint32_t group = 0; group < 4; ++group) {
      float scale = 0.0f;
      std::memcpy(&scale, row + 512 + group * sizeof(float), sizeof(float));
      const float expected = static_cast<float>(token + 1) / 448.0f;
      assert(std::abs(scale - expected) <= 1.0e-6f);
    }
  }

  int32_t history_length = 0;
  int32_t tail_length = 0;
  CudaOk(cudaMemcpy(&history_length, history_lengths_device, sizeof(history_length),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(&tail_length, tail_lengths_device, sizeof(tail_length),
                    cudaMemcpyDeviceToHost));
  assert(history_length == 4);
  assert(tail_length == 1);

  std::vector<uint16_t> output(output_count);
  std::vector<float> output_lse(output_lse_count);
  CudaOk(cudaMemcpy(output.data(), output_device, output.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(output_lse.data(), output_lse_device,
                    output_lse.size() * sizeof(float), cudaMemcpyDeviceToHost));
  float max_error = 0.0f;
  for (uint16_t value : output) {
    max_error = std::max(max_error, std::abs(Bf16ToFloat(value) - 3.0f));
  }
  assert(max_error <= 0.08f);
  for (float lse : output_lse) {
    assert(std::abs(lse - std::log2(5.0f)) <= 3.0e-3f);
  }
  std::printf(
      "GLM FlashInfer sparse-MLA no-RoPE fixture passed, history=%d tail=%d max error=%g\n",
      history_length, tail_length, max_error);

  for (void* pointer : {
           static_cast<void*>(latent_device),
           static_cast<void*>(locations_device),
           static_cast<void*>(cache_device),
           static_cast<void*>(query_device),
           static_cast<void*>(padded_query_device),
           static_cast<void*>(selected_device),
           static_cast<void*>(query_positions_device),
           static_cast<void*>(sequence_lengths_device),
           static_cast<void*>(history_indices_device),
           static_cast<void*>(tail_indices_device),
           static_cast<void*>(history_lengths_device),
           static_cast<void*>(tail_lengths_device),
           static_cast<void*>(history_mid_out_device),
           static_cast<void*>(history_mid_lse_device),
           static_cast<void*>(output_device),
           static_cast<void*>(output_lse_device),
           static_cast<void*>(tail_mid_out_device),
           static_cast<void*>(tail_mid_lse_device),
           static_cast<void*>(tail_output_device),
           static_cast<void*>(tail_output_lse_device),
       }) {
    CudaOk(cudaFree(pointer));
  }
  return 0;
}

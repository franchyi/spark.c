#include "sparkserve/glm_mqa_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr uint32_t kBatch = 2;
constexpr uint32_t kHeads = 32;
constexpr uint32_t kDim = 128;
constexpr uint32_t kPage = 64;
constexpr uint32_t kPages = 8;
constexpr uint32_t kMaxContext = 193;
constexpr uint32_t kLogitsStride = 256;
constexpr uint32_t kBlockTableStride = 4;
constexpr uint64_t kKeyPageBytes = kPage * kDim;
constexpr uint64_t kScalePageBytes = kPage * sizeof(float);

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
  assert(error == cudaSuccess);
}

void StatusOk(SparkServeStatus status) {
  if (status.code != SPARKSERVE_STATUS_OK) std::fprintf(stderr, "%s\n", status.message);
  assert(status.code == SPARKSERVE_STATUS_OK);
}

float DecodeFp8(uint8_t raw) {
  const bool negative = (raw & 0x80u) != 0;
  const uint32_t exponent = (raw >> 3) & 0x0fu;
  const uint32_t mantissa = raw & 0x07u;
  float value;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
  } else {
    assert(exponent != 15 || mantissa != 7);
    value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                       static_cast<int>(exponent) - 7);
  }
  return negative ? -value : value;
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

std::vector<float> Reference(const std::vector<uint8_t>& query,
                             const std::vector<uint8_t>& keys,
                             const std::vector<float>& scales,
                             const std::vector<float>& weights,
                             const std::vector<uint32_t>& lengths,
                             const std::vector<uint32_t>& block_tables) {
  std::vector<float> output(kBatch * kLogitsStride, -999.0f);
  for (uint32_t batch = 0; batch < kBatch; ++batch) {
    for (uint32_t token = 0; token < lengths[batch]; ++token) {
      const uint32_t logical_page = token / kPage;
      const uint32_t page_offset = token % kPage;
      const uint32_t physical_page =
          block_tables[batch * kBlockTableStride + logical_page];
      float value = 0.0f;
      for (uint32_t head = 0; head < kHeads; ++head) {
        float dot = 0.0f;
        for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
          const size_t q_offset =
              (static_cast<size_t>(batch) * kHeads + head) * kDim + dimension;
          const size_t key_offset =
              (static_cast<size_t>(physical_page) * kPage + page_offset) * kDim +
              dimension;
          dot += DecodeFp8(query[q_offset]) * DecodeFp8(keys[key_offset]);
        }
        value += std::max(dot, 0.0f) *
                 weights[static_cast<size_t>(batch) * kHeads + head];
      }
      output[static_cast<size_t>(batch) * kLogitsStride + token] =
          value * scales[static_cast<size_t>(physical_page) * kPage + page_offset];
    }
  }
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
          static_cast<int>(SPARKSERVE_GLM_MQA_GB10_SMS)) {
    std::fprintf(stderr, "GLM paged-MQA fixture requires 48-SM GB10/SM121\n");
    return 77;
  }

  const uint8_t fp8_values[] = {0x00, 0x30, 0x38, 0xb0, 0x40, 0xb8};
  std::vector<uint8_t> query(kBatch * kHeads * kDim);
  std::vector<uint8_t> keys(kPages * kPage * kDim);
  std::vector<float> scales(kPages * kPage);
  std::vector<float> weights(kBatch * kHeads);
  const std::vector<uint32_t> lengths = {193, 130};
  const std::vector<uint32_t> block_tables = {2, 0, 3, 1, 7, 4, 6, 5};

  for (size_t index = 0; index < query.size(); ++index) {
    query[index] = fp8_values[(index * 5 + index / kDim) % 6];
  }
  for (size_t index = 0; index < keys.size(); ++index) {
    keys[index] = fp8_values[(index * 3 + index / kDim + 1) % 6];
  }
  for (uint32_t page = 0; page < kPages; ++page) {
    for (uint32_t token = 0; token < kPage; ++token) {
      scales[static_cast<size_t>(page) * kPage + token] =
          std::ldexp(1.0f, -3 + static_cast<int>((page + token) % 3));
    }
  }
  for (uint32_t batch = 0; batch < kBatch; ++batch) {
    for (uint32_t head = 0; head < kHeads; ++head) {
      const float magnitude = std::ldexp(1.0f, -7 + static_cast<int>(head % 4));
      weights[static_cast<size_t>(batch) * kHeads + head] =
          ((head + batch) % 5 == 0) ? -magnitude : magnitude;
    }
  }

  const std::vector<float> expected =
      Reference(query, keys, scales, weights, lengths, block_tables);
  uint8_t* query_device = DeviceCopy(query);
  uint8_t* keys_device = DeviceCopy(keys);
  float* scales_device = DeviceCopy(scales);
  float* weights_device = DeviceCopy(weights);
  uint32_t* lengths_device = DeviceCopy(lengths);
  uint32_t* block_tables_device = DeviceCopy(block_tables);
  float* logits_device = DeviceAllocate<float>(kBatch * kLogitsStride);
  uint32_t* schedule_device =
      DeviceAllocate<uint32_t>(SPARKSERVE_GLM_MQA_SCHEDULE_WORDS);
  CudaOk(cudaMemset(logits_device, 0xff,
                    kBatch * kLogitsStride * sizeof(float)));

  SparkServeGlmPagedMqaArgs args{};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_GLM_MQA_ABI_VERSION;
  args.batch_size = kBatch;
  args.num_heads = kHeads;
  args.head_dim = kDim;
  args.page_size = kPage;
  args.num_pages = kPages;
  args.num_sms = SPARKSERVE_GLM_MQA_GB10_SMS;
  args.max_context_len = kMaxContext;
  args.logits_stride = kLogitsStride;
  args.block_table_stride = kBlockTableStride;
  args.query_fp8 = query_device;
  args.key_cache_fp8 = keys_device;
  args.scale_cache = scales_device;
  args.logit_weights = weights_device;
  args.context_lens = lengths_device;
  args.logits = logits_device;
  args.block_tables = block_tables_device;
  args.schedule_metadata = schedule_device;
  args.key_page_stride_bytes = kKeyPageBytes;
  args.scale_page_stride_bytes = kScalePageBytes;

  StatusOk(sparkserve_glm_paged_mqa_validate(&args));
  SparkServeGlmPagedMqaArgs invalid = args;
  invalid.num_sms = 47;
  assert(sparkserve_glm_paged_mqa_validate(&invalid).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);
  StatusOk(sparkserve_glm_paged_mqa_launch(&args));
  CudaOk(cudaDeviceSynchronize());

  std::vector<float> actual(kBatch * kLogitsStride);
  CudaOk(cudaMemcpy(actual.data(), logits_device,
                    actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  float max_error = 0.0f;
  for (uint32_t batch = 0; batch < kBatch; ++batch) {
    for (uint32_t token = 0; token < lengths[batch]; ++token) {
      const size_t index = static_cast<size_t>(batch) * kLogitsStride + token;
      const float error = std::abs(actual[index] - expected[index]);
      max_error = std::max(max_error, error);
      assert(error <= 2.0e-4f * std::max(1.0f, std::abs(expected[index])));
    }
  }
  std::printf("GLM DeepGEMM paged-MQA fixture passed, max error=%g\n", max_error);

  CudaOk(cudaFree(query_device));
  CudaOk(cudaFree(keys_device));
  CudaOk(cudaFree(scales_device));
  CudaOk(cudaFree(weights_device));
  CudaOk(cudaFree(lengths_device));
  CudaOk(cudaFree(block_tables_device));
  CudaOk(cudaFree(logits_device));
  CudaOk(cudaFree(schedule_device));
  return 0;
}

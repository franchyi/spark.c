#include "sparkserve/glm_dsa_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kRows = 2;
constexpr uint32_t kPool = 4;
constexpr uint32_t kDim = 128;
constexpr uint32_t kPage = 64;
constexpr size_t kKeyPageBytes = kPage * kDim;
constexpr size_t kScalePageBytes = kPage * sizeof(float);

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

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

float RoundBf16(float value) { return Bf16ToFloat(FloatToBf16(value)); }

float DecodeFp8(uint8_t raw) {
  const bool negative = (raw & 0x80u) != 0;
  const uint32_t exponent = (raw >> 3) & 0x0fu;
  const uint32_t mantissa = raw & 0x07u;
  float value;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
  } else if (exponent == 15 && mantissa == 7) {
    return NAN;
  } else {
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

std::vector<float> Reference(const std::vector<uint16_t>& key,
                             const std::vector<uint16_t>& score,
                             const std::vector<float>& ape,
                             std::vector<float>* scales) {
  assert(key.size() == score.size());
  assert(key.size() % (kPool * kDim) == 0);
  const uint32_t rows = static_cast<uint32_t>(key.size() / (kPool * kDim));
  std::vector<float> output(rows * kDim);
  scales->resize(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    float* row_output = output.data() + row * kDim;
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      float maximum = -INFINITY;
      for (uint32_t slot = 0; slot < kPool; ++slot) {
        const size_t offset = (row * kPool + slot) * kDim + dimension;
        maximum = std::max(maximum,
                           Bf16ToFloat(score[offset]) + ape[slot * kDim + dimension]);
      }
      float accumulator = 0.0f;
      float denominator = 0.0f;
      for (uint32_t slot = 0; slot < kPool; ++slot) {
        const size_t offset = (row * kPool + slot) * kDim + dimension;
        const float probability =
            std::exp(Bf16ToFloat(score[offset]) + ape[slot * kDim + dimension] -
                     maximum);
        denominator += probability;
        accumulator += Bf16ToFloat(key[offset]) * probability;
      }
      row_output[dimension] = RoundBf16(accumulator / denominator);
    }
    for (uint32_t stride = 1; stride < kDim; stride *= 2) {
      for (uint32_t base = 0; base < kDim; base += stride * 2) {
        for (uint32_t offset = 0; offset < stride; ++offset) {
          const float a = row_output[base + offset];
          const float b = row_output[base + offset + stride];
          row_output[base + offset] = a + b;
          row_output[base + offset + stride] = a - b;
        }
      }
    }
    float absmax = 0.0f;
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      row_output[dimension] =
          RoundBf16(row_output[dimension] * 0.08838834764831845f);
      absmax = std::max(absmax, std::abs(row_output[dimension]));
    }
    (*scales)[row] = std::max(absmax, 1.0e-4f) / 448.0f;
  }
  return output;
}

void CompareCache(const std::vector<uint8_t>& key_cache,
                  const std::vector<float>& scale_cache,
                  uint64_t location, const float* expected,
                  float expected_scale) {
  const size_t page = location / kPage;
  const size_t page_offset = location % kPage;
  const float scale = scale_cache[page * kPage + page_offset];
  assert(std::abs(scale - expected_scale) <=
         2.0e-6f * std::max(1.0f, expected_scale));
  float max_error = 0.0f;
  for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
    const uint8_t raw = key_cache[page * kKeyPageBytes +
                                  page_offset * kDim + dimension];
    const float actual = DecodeFp8(raw) * scale;
    const float error = std::abs(actual - expected[dimension]);
    max_error = std::max(max_error, error);
    assert(error <= 20.0f * scale + 2.0e-5f);
  }
  std::printf("GLM KPool location %llu max dequant error=%g\n",
              static_cast<unsigned long long>(location), max_error);
}

std::vector<float> ReferenceIndexerQuery(const std::vector<float>& query,
                                         uint32_t rows,
                                         std::vector<float>* scales) {
  assert(query.size() == static_cast<size_t>(rows) * kDim);
  std::vector<float> transformed(query.size());
  scales->resize(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    float* output = transformed.data() + row * kDim;
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      output[dimension] = RoundBf16(query[row * kDim + dimension]);
    }
    for (uint32_t stride = 1; stride < kDim; stride *= 2) {
      for (uint32_t base = 0; base < kDim; base += stride * 2) {
        for (uint32_t offset = 0; offset < stride; ++offset) {
          const float a = output[base + offset];
          const float b = output[base + offset + stride];
          output[base + offset] = a + b;
          output[base + offset + stride] = a - b;
        }
      }
    }
    float absmax = 1.0e-10f;
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      output[dimension] =
          RoundBf16(output[dimension] * 0.08838834764831845f);
      absmax = std::max(absmax, std::abs(output[dimension]));
    }
    const float raw_scale = absmax / 448.0f;
    (*scales)[row] = std::exp2(std::ceil(std::log2(std::max(raw_scale, 1.0e-10f))));
  }
  return transformed;
}

std::vector<uint16_t> ReferenceIndexerKey(
    const std::vector<float>& key, const std::vector<float>& weight,
    const std::vector<float>& bias, float epsilon) {
  assert(key.size() % kDim == 0);
  const uint32_t tokens = static_cast<uint32_t>(key.size() / kDim);
  std::vector<uint16_t> output(key.size());
  std::vector<float> reduction(kDim);
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      reduction[dimension] = RoundBf16(key[token * kDim + dimension]);
    }
    for (uint32_t stride = kDim / 2; stride > 0; stride /= 2) {
      for (uint32_t dimension = 0; dimension < stride; ++dimension) {
        reduction[dimension] += reduction[dimension + stride];
      }
    }
    const float mean = reduction[0] / static_cast<float>(kDim);
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      const float centered =
          RoundBf16(key[token * kDim + dimension]) - mean;
      reduction[dimension] = centered * centered;
    }
    for (uint32_t stride = kDim / 2; stride > 0; stride /= 2) {
      for (uint32_t dimension = 0; dimension < stride; ++dimension) {
        reduction[dimension] += reduction[dimension + stride];
      }
    }
    const float inverse = 1.0f / std::sqrt(
        reduction[0] / static_cast<float>(kDim) + epsilon);
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      const float centered =
          RoundBf16(key[token * kDim + dimension]) - mean;
      output[token * kDim + dimension] = FloatToBf16(
          centered * inverse * weight[dimension] + bias[dimension]);
    }
  }
  return output;
}

void TestIndexerPrep() {
  constexpr uint32_t kTokens = 2;
  constexpr uint32_t kHeads = 32;
  constexpr uint32_t kQueryRows = kTokens * kHeads;
  constexpr float kEpsilon = 1.0e-6f;
  std::vector<float> query(kQueryRows * kDim);
  std::vector<float> key(kTokens * kDim);
  std::vector<float> norm_weight(kDim);
  std::vector<float> norm_bias(kDim);
  std::vector<float> head_gate(kQueryRows);
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.4f * std::sin(static_cast<float>(i) * 0.013f) +
               0.1f * std::cos(static_cast<float>(i) * 0.003f);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = 0.25f * std::cos(static_cast<float>(i) * 0.019f) - 0.02f;
  }
  for (uint32_t i = 0; i < kDim; ++i) {
    norm_weight[i] = 0.8f + 0.002f * static_cast<float>(i);
    norm_bias[i] = 0.01f * std::sin(static_cast<float>(i) * 0.07f);
  }
  for (uint32_t i = 0; i < kQueryRows; ++i) {
    head_gate[i] = 0.2f * std::cos(static_cast<float>(i) * 0.11f);
  }
  std::vector<float> expected_scales;
  const std::vector<float> expected_query =
      ReferenceIndexerQuery(query, kQueryRows, &expected_scales);
  const std::vector<uint16_t> expected_key =
      ReferenceIndexerKey(key, norm_weight, norm_bias, kEpsilon);

  float* d_query = DeviceCopy(query);
  float* d_key = DeviceCopy(key);
  float* d_norm_weight = DeviceCopy(norm_weight);
  float* d_norm_bias = DeviceCopy(norm_bias);
  float* d_head_gate = DeviceCopy(head_gate);
  uint8_t* d_query_fp8 = nullptr;
  float* d_query_scale = nullptr;
  uint16_t* d_key_bf16 = nullptr;
  float* d_logit_weights = nullptr;
  CudaOk(cudaMalloc(&d_query_fp8, query.size()));
  CudaOk(cudaMalloc(&d_query_scale, kQueryRows * sizeof(float)));
  CudaOk(cudaMalloc(&d_key_bf16, key.size() * sizeof(uint16_t)));
  CudaOk(cudaMalloc(&d_logit_weights, kQueryRows * sizeof(float)));

  SparkServeGlmIndexerPrepArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_GLM_DSA_ABI_VERSION;
  args.query_fp32 = d_query;
  args.key_fp32 = d_key;
  args.key_norm_weight = d_norm_weight;
  args.key_norm_bias = d_norm_bias;
  args.head_gate_fp32 = d_head_gate;
  args.query_fp8 = d_query_fp8;
  args.query_scale = d_query_scale;
  args.key_bf16 = d_key_bf16;
  args.logit_weights = d_logit_weights;
  args.tokens = kTokens;
  args.heads = kHeads;
  args.head_dim = kDim;
  args.layer_norm_epsilon = kEpsilon;
  args.round_scale = 1;
  StatusOk(sparkserve_glm_indexer_prep_launch(&args));
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> actual_query(query.size());
  std::vector<float> actual_scales(kQueryRows);
  std::vector<uint16_t> actual_key(key.size());
  std::vector<float> actual_weights(kQueryRows);
  CudaOk(cudaMemcpy(actual_query.data(), d_query_fp8, actual_query.size(),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_scales.data(), d_query_scale,
                    actual_scales.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_key.data(), d_key_bf16,
                    actual_key.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_weights.data(), d_logit_weights,
                    actual_weights.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
  assert(actual_key == expected_key);
  float max_query_error = 0.0f;
  for (uint32_t row = 0; row < kQueryRows; ++row) {
    assert(actual_scales[row] == expected_scales[row]);
    float expected_weight =
        head_gate[row] * (1.0f / std::sqrt(static_cast<float>(kHeads)));
    expected_weight *= expected_scales[row];
    expected_weight *= 0.08838834764831845f;
    assert(std::abs(actual_weights[row] - expected_weight) <=
           1.0e-6f * std::max(1.0f, std::abs(expected_weight)));
    for (uint32_t dimension = 0; dimension < kDim; ++dimension) {
      const size_t index = row * kDim + dimension;
      const float restored = DecodeFp8(actual_query[index]) * actual_scales[row];
      const float error = std::abs(restored - expected_query[index]);
      max_query_error = std::max(max_query_error, error);
      assert(error <= 20.0f * actual_scales[row] + 2.0e-5f);
    }
  }
  std::printf("GLM indexer prep max query dequant error=%g\n", max_query_error);

  args.heads = 64;
  assert(sparkserve_glm_indexer_prep_validate(&args).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);
  CudaOk(cudaFree(d_logit_weights));
  CudaOk(cudaFree(d_key_bf16));
  CudaOk(cudaFree(d_query_scale));
  CudaOk(cudaFree(d_query_fp8));
  CudaOk(cudaFree(d_head_gate));
  CudaOk(cudaFree(d_norm_bias));
  CudaOk(cudaFree(d_norm_weight));
  CudaOk(cudaFree(d_key));
  CudaOk(cudaFree(d_query));
}

}  // namespace

int main() {
  TestIndexerPrep();
  std::vector<uint16_t> key(kRows * kPool * kDim);
  std::vector<uint16_t> score(key.size());
  std::vector<float> ape(kPool * kDim);
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = FloatToBf16(0.2f * std::sin(static_cast<float>(i) * 0.017f));
    score[i] = FloatToBf16(0.15f * std::cos(static_cast<float>(i) * 0.011f));
  }
  for (size_t i = 0; i < ape.size(); ++i) {
    ape[i] = 0.03f * std::sin(static_cast<float>(i) * 0.007f);
  }
  std::vector<int64_t> locations = {0, 65};
  std::vector<float> expected_scales;
  const std::vector<float> expected = Reference(key, score, ape, &expected_scales);

  uint16_t* d_key = DeviceCopy(key);
  uint16_t* d_score = DeviceCopy(score);
  float* d_ape = DeviceCopy(ape);
  int64_t* d_locations = DeviceCopy(locations);
  uint8_t* d_key_cache = nullptr;
  float* d_scale_cache = nullptr;
  CudaOk(cudaMalloc(&d_key_cache, 2 * kKeyPageBytes));
  CudaOk(cudaMalloc(&d_scale_cache, 2 * kScalePageBytes));
  CudaOk(cudaMemset(d_key_cache, 0, 2 * kKeyPageBytes));
  CudaOk(cudaMemset(d_scale_cache, 0, 2 * kScalePageBytes));

  SparkServeGlmKPoolCompressArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_GLM_DSA_ABI_VERSION;
  args.rows = kRows;
  args.pool_size = kPool;
  args.head_dim = kDim;
  args.page_size = kPage;
  args.slot_key_bf16 = d_key;
  args.slot_score_bf16 = d_score;
  args.ape = d_ape;
  args.locations = d_locations;
  args.key_cache_fp8 = d_key_cache;
  args.scale_cache = d_scale_cache;
  args.key_page_stride_bytes = kKeyPageBytes;
  args.scale_page_stride_bytes = kScalePageBytes;
  StatusOk(sparkserve_glm_kpool_compress_launch(&args));
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> actual_key_cache(2 * kKeyPageBytes);
  std::vector<float> actual_scale_cache(2 * kPage);
  CudaOk(cudaMemcpy(actual_key_cache.data(), d_key_cache, actual_key_cache.size(),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_scale_cache.data(), d_scale_cache,
                    actual_scale_cache.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
  for (uint32_t row = 0; row < kRows; ++row) {
    CompareCache(actual_key_cache, actual_scale_cache, locations[row],
                 expected.data() + row * kDim, expected_scales[row]);
  }

  args.pool_size = 8;
  assert(sparkserve_glm_kpool_compress_validate(&args).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);

  // The decode adapter must preserve the full four-token state ring and only
  // publish a compressed cache entry when the fourth token completes a group.
  std::vector<int32_t> request_indices(2 * kPool, 0);
  std::vector<int64_t> positions(2 * kPool);
  std::vector<int32_t> sequence_lengths(2 * kPool);
  std::vector<int64_t> output_cache_locations(2 * kPool, 1);
  for (uint32_t token = 0; token < 2 * kPool; ++token) {
    positions[token] = token;
    sequence_lengths[token] = static_cast<int32_t>(token + 1);
  }
  std::vector<int32_t> block_tables = {0, -1, -1, -1};
  int32_t* d_request_indices = DeviceCopy(request_indices);
  int64_t* d_positions = DeviceCopy(positions);
  int32_t* d_sequence_lengths = DeviceCopy(sequence_lengths);
  int64_t* d_output_cache_locations = DeviceCopy(output_cache_locations);
  int32_t* d_block_tables = DeviceCopy(block_tables);
  uint16_t* d_tail_key = nullptr;
  uint16_t* d_tail_score = nullptr;
  CudaOk(cudaMalloc(&d_tail_key, kPool * kDim * sizeof(uint16_t)));
  CudaOk(cudaMalloc(&d_tail_score, kPool * kDim * sizeof(uint16_t)));
  CudaOk(cudaMemset(d_tail_key, 0, kPool * kDim * sizeof(uint16_t)));
  CudaOk(cudaMemset(d_tail_score, 0, kPool * kDim * sizeof(uint16_t)));
  CudaOk(cudaMemset(d_key_cache, 0, 2 * kKeyPageBytes));
  CudaOk(cudaMemset(d_scale_cache, 0, 2 * kScalePageBytes));

  SparkServeGlmKPoolDecodeArgs decode = {};
  decode.struct_size = sizeof(decode);
  decode.abi_version = SPARKSERVE_GLM_DSA_ABI_VERSION;
  decode.rows = 1;
  decode.request_capacity = 1;
  decode.tail_size = kPool;
  decode.head_dim = kDim;
  decode.pool_size = kPool;
  decode.page_size = kPage;
  decode.slots_per_page = kPage;
  decode.tail_key_bf16 = d_tail_key;
  decode.tail_score_bf16 = d_tail_score;
  decode.ape = d_ape;
  decode.block_tables = d_block_tables;
  decode.key_cache_fp8 = d_key_cache;
  decode.scale_cache = d_scale_cache;
  decode.block_table_stride = block_tables.size();
  decode.key_page_stride_bytes = kKeyPageBytes;
  decode.scale_page_stride_bytes = kScalePageBytes;
  for (uint32_t token = 0; token < 2 * kPool; ++token) {
    decode.key_bf16 = d_key + token * kDim;
    decode.score_bf16 = d_score + token * kDim;
    decode.request_indices = d_request_indices + token;
    decode.positions = d_positions + token;
    decode.sequence_lengths = d_sequence_lengths + token;
    decode.output_cache_locations = d_output_cache_locations + token;
    StatusOk(sparkserve_glm_kpool_decode_launch(&decode));
  }
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint16_t> actual_tail_key(kPool * kDim);
  std::vector<uint16_t> actual_tail_score(kPool * kDim);
  CudaOk(cudaMemcpy(actual_tail_key.data(), d_tail_key,
                    actual_tail_key.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_tail_score.data(), d_tail_score,
                    actual_tail_score.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost));
  assert(std::equal(actual_tail_key.begin(), actual_tail_key.end(),
                    key.begin() + kPool * kDim));
  assert(std::equal(actual_tail_score.begin(), actual_tail_score.end(),
                    score.begin() + kPool * kDim));

  CudaOk(cudaMemcpy(actual_key_cache.data(), d_key_cache, actual_key_cache.size(),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_scale_cache.data(), d_scale_cache,
                    actual_scale_cache.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
  CompareCache(actual_key_cache, actual_scale_cache, 0, expected.data(),
               expected_scales[0]);
  CompareCache(actual_key_cache, actual_scale_cache, 1,
               expected.data() + kDim, expected_scales[1]);
  decode.tail_size = 3;
  assert(sparkserve_glm_kpool_decode_validate(&decode).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);

  CudaOk(cudaFree(d_tail_score));
  CudaOk(cudaFree(d_tail_key));
  CudaOk(cudaFree(d_block_tables));
  CudaOk(cudaFree(d_output_cache_locations));
  CudaOk(cudaFree(d_sequence_lengths));
  CudaOk(cudaFree(d_positions));
  CudaOk(cudaFree(d_request_indices));
  CudaOk(cudaFree(d_scale_cache));
  CudaOk(cudaFree(d_key_cache));
  CudaOk(cudaFree(d_locations));
  CudaOk(cudaFree(d_ape));
  CudaOk(cudaFree(d_score));
  CudaOk(cudaFree(d_key));
  return 0;
}

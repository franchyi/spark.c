#include "q27_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr uint32_t kQHeads = Q27_ATTENTION_QUERY_HEADS;
constexpr uint32_t kKvHeads = Q27_ATTENTION_KV_HEADS;
constexpr uint32_t kHeadDim = Q27_ATTENTION_HEAD_DIM;
constexpr uint32_t kRotaryDim = Q27_ATTENTION_ROTARY_DIM;
constexpr uint32_t kRotaryHalf = kRotaryDim / 2;
constexpr uint32_t kPageSize = Q27_ATTENTION_PAGE_SIZE;
constexpr uint32_t kLength = 31;
constexpr uint32_t kPosition = kLength - 1;

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}

void StatusOk(q27_attention_status status) {
  if (status.code != Q27_ATTENTION_OK) {
    std::fprintf(stderr, "q27 attention: %s\n", status.message);
    std::abort();
  }
}

template <typename T>
T* Device(size_t elements) {
  T* pointer = nullptr;
  CudaOk(cudaMalloc(&pointer, elements * sizeof(T)));
  return pointer;
}

template <typename T>
T* Upload(const std::vector<T>& values) {
  T* pointer = Device<T>(values.size());
  CudaOk(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
  return pointer;
}

float Bf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

float Pattern(uint64_t index, uint32_t modulus, float scale) {
  return (static_cast<int64_t>((index * 37 + 11) % modulus) -
          static_cast<int64_t>(modulus / 2)) *
         scale;
}

void ReferencePrepare(
    const std::vector<__nv_bfloat16>& q_gate,
    const std::vector<__nv_bfloat16>& key,
    const std::vector<__nv_bfloat16>& value,
    const std::vector<__nv_bfloat16>& q_weight,
    const std::vector<__nv_bfloat16>& k_weight,
    const std::vector<float>& rope,
    std::vector<__nv_bfloat16>* query,
    std::vector<__nv_bfloat16>* gate,
    std::vector<__nv_fp8_e4m3>* key_cache,
    std::vector<__nv_fp8_e4m3>* value_cache) {
  for (uint32_t head = 0; head < kQHeads + kKvHeads; ++head) {
    const bool is_key = head >= kQHeads;
    const uint32_t local_head = is_key ? head - kQHeads : head;
    const auto& input = is_key ? key : q_gate;
    const auto& weight = is_key ? k_weight : q_weight;
    const uint64_t input_base =
        is_key ? static_cast<uint64_t>(local_head) * kHeadDim
               : static_cast<uint64_t>(local_head) * 2 * kHeadDim;
    float sum = 0.0f;
    for (uint32_t element = 0; element < kHeadDim; ++element) {
      const float x = __bfloat162float(input[input_base + element]);
      sum += x * x;
    }
    const float inverse_rms = std::sqrt(1.0f / (sum / kHeadDim + 1.0e-6f));
    std::vector<float> normalized(kHeadDim);
    for (uint32_t element = 0; element < kHeadDim; ++element) {
      normalized[element] = Bf16(
          __bfloat162float(input[input_base + element]) * inverse_rms *
          (__bfloat162float(weight[element]) + 1.0f));
    }
    for (uint32_t element = 0; element < kHeadDim; ++element) {
      float transformed = normalized[element];
      if (element < kRotaryDim) {
        const uint32_t pair = element < kRotaryHalf
                                  ? element + kRotaryHalf
                                  : element - kRotaryHalf;
        const uint32_t frequency = element % kRotaryHalf;
        const float cosine =
            rope[static_cast<uint64_t>(kPosition) * kRotaryDim + frequency];
        const float sine =
            rope[static_cast<uint64_t>(kPosition) * kRotaryDim + kRotaryHalf +
                 frequency];
        transformed = element < kRotaryHalf
                          ? normalized[element] * cosine - normalized[pair] * sine
                          : normalized[element] * cosine + normalized[pair] * sine;
      }
      const auto transformed_bf16 = __float2bfloat16_rn(transformed);
      if (!is_key) {
        const uint64_t out = static_cast<uint64_t>(local_head) * kHeadDim + element;
        (*query)[out] = transformed_bf16;
        (*gate)[out] = input[input_base + kHeadDim + element];
      } else {
        const uint64_t cache =
            ((static_cast<uint64_t>(kPosition) * kKvHeads + local_head) *
             kHeadDim) +
            element;
        (*key_cache)[cache] =
            __nv_fp8_e4m3(__bfloat162float(transformed_bf16));
        (*value_cache)[cache] = __nv_fp8_e4m3(
            __bfloat162float(value[static_cast<uint64_t>(local_head) * kHeadDim +
                                  element]));
      }
    }
  }
}

std::vector<__nv_bfloat16> ReferenceDecode(
    const std::vector<__nv_bfloat16>& query,
    const std::vector<__nv_bfloat16>& gate,
    const std::vector<__nv_fp8_e4m3>& key_cache,
    const std::vector<__nv_fp8_e4m3>& value_cache) {
  std::vector<__nv_bfloat16> output(kQHeads * kHeadDim);
  std::vector<float> scores(kLength);
  for (uint32_t q_head = 0; q_head < kQHeads; ++q_head) {
    const uint32_t kv_head = q_head / (kQHeads / kKvHeads);
    float maximum = -INFINITY;
    for (uint32_t token = 0; token < kLength; ++token) {
      float dot = 0.0f;
      for (uint32_t element = 0; element < kHeadDim; ++element) {
        const uint64_t cache =
            ((static_cast<uint64_t>(token) * kKvHeads + kv_head) * kHeadDim) +
            element;
        dot += __bfloat162float(query[static_cast<uint64_t>(q_head) * kHeadDim +
                                      element]) *
               static_cast<float>(key_cache[cache]);
      }
      scores[token] = dot * 0.0625f;
      maximum = std::max(maximum, scores[token]);
    }
    float denominator = 0.0f;
    for (float& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (uint32_t element = 0; element < kHeadDim; ++element) {
      float value = 0.0f;
      for (uint32_t token = 0; token < kLength; ++token) {
        const uint64_t cache =
            ((static_cast<uint64_t>(token) * kKvHeads + kv_head) * kHeadDim) +
            element;
        value += scores[token] * static_cast<float>(value_cache[cache]);
      }
      value /= denominator;
      const uint64_t out = static_cast<uint64_t>(q_head) * kHeadDim + element;
      const float g = __bfloat162float(gate[out]);
      output[out] = __float2bfloat16_rn(value / (1.0f + std::exp(-g)));
    }
  }
  return output;
}

float TimeDecode(q27_attention_decode_args* args, uint32_t iterations) {
  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  for (int warmup = 0; warmup < 5; ++warmup) StatusOk(q27_attention_decode(args));
  CudaOk(cudaEventRecord(begin));
  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    StatusOk(q27_attention_decode(args));
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CudaOk(cudaEventElapsedTime(&milliseconds, begin, end));
  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  return milliseconds * 1000.0f / iterations;
}

}  // namespace

int main() {
  constexpr uint32_t kBenchCapacity = 32768;
  constexpr uint32_t kBenchPages = kBenchCapacity / kPageSize;
  constexpr uint64_t kCacheElements =
      static_cast<uint64_t>(kBenchCapacity) * kKvHeads * kHeadDim;
  constexpr uint64_t kQElements = static_cast<uint64_t>(kQHeads) * kHeadDim;

  std::vector<__nv_bfloat16> q_gate(kQHeads * 2 * kHeadDim);
  std::vector<__nv_bfloat16> key(kKvHeads * kHeadDim);
  std::vector<__nv_bfloat16> value(kKvHeads * kHeadDim);
  std::vector<__nv_bfloat16> q_weight(kHeadDim);
  std::vector<__nv_bfloat16> k_weight(kHeadDim);
  std::vector<float> rope((kPosition + 1) * kRotaryDim);
  for (uint64_t i = 0; i < q_gate.size(); ++i)
    q_gate[i] = __float2bfloat16_rn(Pattern(i, 101, 0.0125f));
  for (uint64_t i = 0; i < key.size(); ++i) {
    key[i] = __float2bfloat16_rn(Pattern(i, 83, 0.014f));
    value[i] = __float2bfloat16_rn(Pattern(i + 7, 79, 0.013f));
  }
  for (uint32_t i = 0; i < kHeadDim; ++i) {
    q_weight[i] = __float2bfloat16_rn(Pattern(i, 41, 0.001f));
    k_weight[i] = __float2bfloat16_rn(Pattern(i + 3, 37, 0.001f));
  }
  for (uint32_t position = 0; position <= kPosition; ++position) {
    for (uint32_t frequency = 0; frequency < kRotaryHalf; ++frequency) {
      const float inverse_frequency =
          std::pow(10000000.0f, -2.0f * frequency / kRotaryDim);
      const float angle = position * inverse_frequency;
      rope[static_cast<uint64_t>(position) * kRotaryDim + frequency] =
          std::cos(angle);
      rope[static_cast<uint64_t>(position) * kRotaryDim + kRotaryHalf + frequency] =
          std::sin(angle);
    }
  }

  std::vector<__nv_fp8_e4m3> key_cache(kCacheElements);
  std::vector<__nv_fp8_e4m3> value_cache(kCacheElements);
  for (uint64_t i = 0; i < static_cast<uint64_t>(kLength) * kKvHeads * kHeadDim;
       ++i) {
    key_cache[i] = __nv_fp8_e4m3(Pattern(i, 61, 0.009f));
    value_cache[i] = __nv_fp8_e4m3(Pattern(i + 5, 67, 0.008f));
  }
  std::vector<__nv_bfloat16> expected_query(kQElements);
  std::vector<__nv_bfloat16> expected_gate(kQElements);
  ReferencePrepare(q_gate, key, value, q_weight, k_weight, rope,
                   &expected_query, &expected_gate, &key_cache, &value_cache);
  const auto expected_output =
      ReferenceDecode(expected_query, expected_gate, key_cache, value_cache);
  auto device_key_cache = key_cache;
  auto device_value_cache = value_cache;
  const uint64_t current_token_base =
      static_cast<uint64_t>(kPosition) * kKvHeads * kHeadDim;
  for (uint64_t i = current_token_base;
       i < current_token_base + static_cast<uint64_t>(kKvHeads) * kHeadDim;
       ++i) {
    device_key_cache[i] = __nv_fp8_e4m3(0.0f);
    device_value_cache[i] = __nv_fp8_e4m3(0.0f);
  }

  auto* d_q_gate = Upload(q_gate);
  auto* d_key = Upload(key);
  auto* d_value = Upload(value);
  auto* d_q_weight = Upload(q_weight);
  auto* d_k_weight = Upload(k_weight);
  auto* d_rope = Upload(rope);
  auto* d_key_cache = Upload(device_key_cache);
  auto* d_value_cache = Upload(device_value_cache);
  auto* d_query = Device<__nv_bfloat16>(kQElements);
  auto* d_gate = Device<__nv_bfloat16>(kQElements);
  auto* d_output = Device<__nv_bfloat16>(kQElements);
  std::vector<int32_t> pages(kBenchPages);
  for (uint32_t page = 0; page < kBenchPages; ++page) pages[page] = page;
  auto* d_pages = Upload(pages);
  std::vector<uint32_t> length = {kLength};
  auto* d_length = Upload(length);
  void* d_workspace = nullptr;
  CudaOk(cudaMalloc(&d_workspace, Q27_ATTENTION_WORKSPACE_BYTES));

  q27_attention_prepare_store_args prepare = {};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = Q27_ATTENTION_ABI_VERSION;
  prepare.q_gate_bf16 = d_q_gate;
  prepare.key_bf16 = d_key;
  prepare.value_bf16 = d_value;
  prepare.q_norm_weight_bf16 = d_q_weight;
  prepare.k_norm_weight_bf16 = d_k_weight;
  prepare.rope_cos_sin_f32 = d_rope;
  prepare.rope_row_stride_elements = kRotaryDim;
  prepare.position = kPosition;
  prepare.query_bf16 = d_query;
  prepare.gate_bf16 = d_gate;
  prepare.key_cache_fp8_e4m3 = d_key_cache;
  prepare.value_cache_fp8_e4m3 = d_value_cache;
  prepare.physical_page_index = kPosition;
  prepare.token_offset_in_page = 0;
  prepare.key_scale = 1.0f;
  prepare.value_scale = 1.0f;
  StatusOk(q27_attention_prepare_store(&prepare));
  CudaOk(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual_query(kQElements);
  std::vector<__nv_bfloat16> actual_gate(kQElements);
  CudaOk(cudaMemcpy(actual_query.data(), d_query,
                    actual_query.size() * sizeof(actual_query[0]),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_gate.data(), d_gate,
                    actual_gate.size() * sizeof(actual_gate[0]),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(device_key_cache.data(), d_key_cache,
                    device_key_cache.size() * sizeof(device_key_cache[0]),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(device_value_cache.data(), d_value_cache,
                    device_value_cache.size() * sizeof(device_value_cache[0]),
                    cudaMemcpyDeviceToHost));
  assert(std::equal(reinterpret_cast<const uint16_t*>(actual_query.data()),
                    reinterpret_cast<const uint16_t*>(actual_query.data()) + kQElements,
                    reinterpret_cast<const uint16_t*>(expected_query.data())));
  assert(std::equal(reinterpret_cast<const uint16_t*>(actual_gate.data()),
                    reinterpret_cast<const uint16_t*>(actual_gate.data()) + kQElements,
                    reinterpret_cast<const uint16_t*>(expected_gate.data())));
  assert(std::equal(reinterpret_cast<const uint8_t*>(device_key_cache.data()),
                    reinterpret_cast<const uint8_t*>(device_key_cache.data()) +
                        static_cast<uint64_t>(kLength) * kKvHeads * kHeadDim,
                    reinterpret_cast<const uint8_t*>(key_cache.data())));
  assert(std::equal(reinterpret_cast<const uint8_t*>(device_value_cache.data()),
                    reinterpret_cast<const uint8_t*>(device_value_cache.data()) +
                        static_cast<uint64_t>(kLength) * kKvHeads * kHeadDim,
                    reinterpret_cast<const uint8_t*>(value_cache.data())));

  q27_attention_decode_args decode = {};
  decode.struct_size = sizeof(decode);
  decode.abi_version = Q27_ATTENTION_ABI_VERSION;
  decode.query_bf16 = d_query;
  decode.gate_bf16 = d_gate;
  decode.key_cache_fp8_e4m3 = d_key_cache;
  decode.value_cache_fp8_e4m3 = d_value_cache;
  decode.block_table_i32 = d_pages;
  decode.sequence_length_u32 = d_length;
  decode.max_sequence_length = kLength;
  decode.kv_scale = 1.0f;
  decode.output_bf16 = d_output;
  decode.workspace = d_workspace;
  decode.workspace_bytes = Q27_ATTENTION_WORKSPACE_BYTES;
  decode.multiprocessor_count = 48;
  decode.enable_pdl = 1;
  StatusOk(q27_attention_decode(&decode));
  CudaOk(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual_output(kQElements);
  CudaOk(cudaMemcpy(actual_output.data(), d_output,
                    actual_output.size() * sizeof(actual_output[0]),
                    cudaMemcpyDeviceToHost));
  float maximum_error = 0.0f;
  for (uint64_t i = 0; i < kQElements; ++i) {
    maximum_error = std::max(
        maximum_error,
        std::fabs(__bfloat162float(actual_output[i]) -
                  __bfloat162float(expected_output[i])));
  }
  std::printf("q27 attention length-31 max BF16 error: %.8f\n", maximum_error);
  assert(maximum_error <= 0.015625f);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < 1000; ++iteration)
    StatusOk(q27_attention_prepare_store(&prepare));
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CudaOk(cudaEventElapsedTime(&milliseconds, begin, end));
  std::printf("q27 prepare/store mean: %.3f us\n", milliseconds);
  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));

  std::printf("q27 attention decode length-31 mean: %.3f us\n",
              TimeDecode(&decode, 200));
  length[0] = 2048;
  CudaOk(cudaMemcpy(d_length, length.data(), sizeof(uint32_t),
                    cudaMemcpyHostToDevice));
  decode.max_sequence_length = 2048;
  std::printf("q27 attention decode length-2048 mean: %.3f us\n",
              TimeDecode(&decode, 100));
  length[0] = kBenchCapacity;
  CudaOk(cudaMemcpy(d_length, length.data(), sizeof(uint32_t),
                    cudaMemcpyHostToDevice));
  decode.max_sequence_length = kBenchCapacity;
  std::printf("q27 attention decode length-32768 mean: %.3f us\n",
              TimeDecode(&decode, 20));

  cudaFree(d_workspace);
  cudaFree(d_length);
  cudaFree(d_pages);
  cudaFree(d_output);
  cudaFree(d_gate);
  cudaFree(d_query);
  cudaFree(d_value_cache);
  cudaFree(d_key_cache);
  cudaFree(d_rope);
  cudaFree(d_k_weight);
  cudaFree(d_q_weight);
  cudaFree(d_value);
  cudaFree(d_key);
  cudaFree(d_q_gate);
  return 0;
}

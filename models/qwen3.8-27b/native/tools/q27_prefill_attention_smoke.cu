#include "q27_prefill_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kCapacity = 257;
constexpr uint32_t kCommitted = 3;
constexpr uint32_t kQHeads = Q27_ATTENTION_QUERY_HEADS;
constexpr uint32_t kKvHeads = Q27_ATTENTION_KV_HEADS;
constexpr uint32_t kHeadDim = Q27_ATTENTION_HEAD_DIM;
constexpr uint32_t kQColumns = kQHeads * kHeadDim;
constexpr uint32_t kKvColumns = kKvHeads * kHeadDim;
constexpr uint32_t kTile = Q27_PREFILL_ATTENTION_TILE_TOKENS;
constexpr uint64_t kCacheBytes =
    static_cast<uint64_t>(kCapacity) * kKvColumns;

uint16_t Bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>(
      (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

float FromBf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

float FromFp8(uint8_t value) {
  const float sign = (value & 0x80U) != 0 ? -1.0F : 1.0F;
  const uint32_t exponent = (value >> 3U) & 0xFU;
  const uint32_t mantissa = value & 0x7U;
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa), -9);
  }
  if (exponent == 15 && mantissa == 7) return NAN;
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void CheckStatus(q27_prefill_attention_status status,
                 const char* operation) {
  if (status.code != Q27_PREFILL_ATTENTION_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void ExpectStatus(q27_prefill_attention_status status, int32_t code,
                  const char* operation) {
  if (status.code != code) {
    throw std::runtime_error(std::string(operation) + " status mismatch");
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                const char* operation) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(operation) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void* DeviceBytes(uint64_t bytes) {
  void* output = nullptr;
  CheckCuda(cudaMalloc(&output, bytes), "cudaMalloc");
  CheckCuda(cudaMemset(output, 0, bytes), "cudaMemset");
  return output;
}

template <typename T>
std::vector<T> HostCopy(const T* device, uint64_t elements) {
  std::vector<T> output(elements);
  CheckCuda(cudaMemcpy(output.data(), device, elements * sizeof(T),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H");
  return output;
}

__global__ void InitializeBlockTable(int32_t* block_table,
                                     uint32_t capacity) {
  for (uint32_t logical = blockIdx.x * blockDim.x + threadIdx.x;
       logical < capacity; logical += blockDim.x * gridDim.x) {
    block_table[logical] = static_cast<int32_t>(capacity - 1 - logical);
  }
}

__global__ void InitializeRope(float* rope, uint32_t position_capacity) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < position_capacity * Q27_ATTENTION_ROTARY_DIM;
       index += blockDim.x * gridDim.x) {
    const uint32_t position = index / Q27_ATTENTION_ROTARY_DIM;
    const uint32_t within = index % Q27_ATTENTION_ROTARY_DIM;
    const uint32_t frequency = within % (Q27_ATTENTION_ROTARY_DIM / 2);
    const float angle = 0.01F * static_cast<float>(position) *
                        static_cast<float>(frequency + 1);
    rope[index] = within < Q27_ATTENTION_ROTARY_DIM / 2 ? cosf(angle)
                                                        : sinf(angle);
  }
}

__global__ void InitializeCommitted(__nv_fp8_e4m3* key_cache,
                                    __nv_fp8_e4m3* value_cache,
                                    const int32_t* block_table,
                                    uint32_t committed_tokens) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < committed_tokens * kKvColumns;
       index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kKvColumns;
    const uint32_t within = index - token * kKvColumns;
    const uint32_t head = within / kHeadDim;
    const uint32_t dimension = within - head * kHeadDim;
    const uint32_t physical = block_table[token];
    const uint64_t cache =
        (static_cast<uint64_t>(physical) * kKvHeads + head) * kHeadDim +
        dimension;
    float key = 0.0F;
    const float bounded_token = static_cast<float>(token % 17U + 1U);
    if (dimension == 0) key = 0.5F * bounded_token + 0.01F * head;
    if (dimension == 32) key = -0.25F * bounded_token;
    const float value =
        dimension == 0
            ? bounded_token + 0.25F * head
            : 0.0F;
    key_cache[cache] = __nv_fp8_e4m3(key);
    value_cache[cache] = __nv_fp8_e4m3(value);
  }
}

__global__ void InitializeInputs(__nv_bfloat16* q_gate,
                                 __nv_bfloat16* key,
                                 __nv_bfloat16* value,
                                 __nv_bfloat16* q_norm,
                                 __nv_bfloat16* k_norm,
                                 uint32_t committed_tokens) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTile * kQHeads * 2 * kHeadDim;
       index += blockDim.x * gridDim.x) {
    const uint32_t token = index / (kQHeads * 2 * kHeadDim);
    const uint32_t within = index % (kQHeads * 2 * kHeadDim);
    const uint32_t head = within / (2 * kHeadDim);
    const uint32_t head_within = within % (2 * kHeadDim);
    const uint32_t absolute = committed_tokens + token;
    const float bounded_token = static_cast<float>(absolute % 17U + 1U);
    float output = 0.0F;
    if (head_within == 0)
      output = 0.25F * bounded_token + 0.01F * head;
    if (head_within == 32)
      output = 0.125F * bounded_token;
    q_gate[index] = __float2bfloat16_rn(output);
  }
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTile * kKvColumns; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kKvColumns;
    const uint32_t within = index - token * kKvColumns;
    const uint32_t head = within / kHeadDim;
    const uint32_t dimension = within - head * kHeadDim;
    const uint32_t absolute = committed_tokens + token;
    const float bounded_token = static_cast<float>(absolute % 17U + 1U);
    float key_value = 0.0F;
    if (dimension == 0)
      key_value = 0.3F * bounded_token + 0.01F * head;
    if (dimension == 32)
      key_value = -0.1F * bounded_token;
    key[index] = __float2bfloat16_rn(key_value);
    value[index] = __float2bfloat16_rn(
        dimension == 0
            ? bounded_token + 0.25F * head
            : 0.0F);
  }
  if (blockIdx.x == 0 && threadIdx.x < kHeadDim) {
    q_norm[threadIdx.x] = __float2bfloat16_rn(0.0F);
    k_norm[threadIdx.x] = __float2bfloat16_rn(0.0F);
  }
}

float Reference(const std::vector<uint16_t>& query,
                const std::vector<uint16_t>& gate,
                const std::vector<uint8_t>& key_cache,
                const std::vector<uint8_t>& value_cache, uint32_t token,
                uint32_t q_head, uint32_t dimension) {
  const uint32_t kv_head = q_head / (kQHeads / kKvHeads);
  const uint32_t visible = kCommitted + token + 1;
  const uint64_t q_base =
      (static_cast<uint64_t>(token) * kQHeads + q_head) * kHeadDim;
  std::vector<float> logits(visible);
  float maximum = -INFINITY;
  for (uint32_t logical = 0; logical < visible; ++logical) {
    const uint32_t physical = kCapacity - 1 - logical;
    const uint64_t cache_base =
        (static_cast<uint64_t>(physical) * kKvHeads + kv_head) * kHeadDim;
    float dot = 0.0F;
    for (uint32_t column = 0; column < kHeadDim; ++column) {
      dot += FromBf16(query[q_base + column]) *
             FromFp8(key_cache[cache_base + column]);
    }
    logits[logical] = dot * 0.0625F;
    maximum = std::max(maximum, logits[logical]);
  }
  float numerator = 0.0F;
  float denominator = 0.0F;
  for (uint32_t logical = 0; logical < visible; ++logical) {
    const float probability = std::exp(logits[logical] - maximum);
    const uint32_t physical = kCapacity - 1 - logical;
    const uint64_t cache =
        (static_cast<uint64_t>(physical) * kKvHeads + kv_head) * kHeadDim +
        dimension;
    numerator += probability * FromFp8(value_cache[cache]);
    denominator += probability;
  }
  const float gate_value = FromBf16(gate[q_base + dimension]);
  const float sigmoid = 1.0F / (1.0F + std::exp(-gate_value));
  return numerator / denominator * sigmoid;
}

void RunCase(uint32_t valid_tokens, bool corrupt_table) {
  void* q_gate = DeviceBytes(Q27_PREFILL_ATTENTION_Q_GATE_BYTES);
  void* key = DeviceBytes(Q27_PREFILL_ATTENTION_KV_INPUT_BYTES);
  void* value = DeviceBytes(Q27_PREFILL_ATTENTION_KV_INPUT_BYTES);
  void* q_norm = DeviceBytes(kHeadDim * 2ULL);
  void* k_norm = DeviceBytes(kHeadDim * 2ULL);
  void* rope = DeviceBytes(
      static_cast<uint64_t>(kCapacity) * Q27_ATTENTION_ROTARY_DIM * 4ULL);
  void* block_table = DeviceBytes(kCapacity * sizeof(int32_t));
  void* key_cache = DeviceBytes(kCacheBytes);
  void* value_cache = DeviceBytes(kCacheBytes);
  void* query = DeviceBytes(Q27_PREFILL_ATTENTION_QUERY_BYTES);
  void* gate = DeviceBytes(Q27_PREFILL_ATTENTION_QUERY_BYTES);
  void* output = DeviceBytes(Q27_PREFILL_ATTENTION_QUERY_BYTES);
  const uint64_t workspace_bytes =
      Q27_PREFILL_ATTENTION_WORKSPACE_BYTES(kCapacity);
  void* workspace = DeviceBytes(workspace_bytes);
  cudaStream_t stream = nullptr;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  InitializeBlockTable<<<2, 256, 0, stream>>>(
      static_cast<int32_t*>(block_table), kCapacity);
  InitializeRope<<<64, 256, 0, stream>>>(static_cast<float*>(rope),
                                         kCapacity);
  InitializeCommitted<<<16, 256, 0, stream>>>(
      static_cast<__nv_fp8_e4m3*>(key_cache),
      static_cast<__nv_fp8_e4m3*>(value_cache),
      static_cast<const int32_t*>(block_table), kCommitted);
  InitializeInputs<<<128, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(q_gate),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(q_norm),
      static_cast<__nv_bfloat16*>(k_norm), kCommitted);
  CheckCuda(cudaGetLastError(), "prefill fixture initialization");
  CheckCuda(cudaStreamSynchronize(stream), "prefill fixture init sync");

  if (corrupt_table) {
    const int32_t invalid = static_cast<int32_t>(kCapacity);
    CheckCuda(cudaMemcpy(static_cast<int32_t*>(block_table) + kCommitted,
                         &invalid,
                         sizeof(invalid), cudaMemcpyHostToDevice),
              "corrupt target page table");
  }

  q27_prefill_attention_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_PREFILL_ATTENTION_ABI_VERSION;
  args.q_gate_bf16 = q_gate;
  args.key_bf16 = key;
  args.value_bf16 = value;
  args.q_norm_weight_bf16 = q_norm;
  args.k_norm_weight_bf16 = k_norm;
  args.rope_cos_sin_f32 = static_cast<float*>(rope);
  args.rope_row_stride_elements = Q27_ATTENTION_ROTARY_DIM;
  args.rope_position_capacity = kCapacity;
  args.valid_tokens = valid_tokens;
  args.committed_tokens = kCommitted;
  args.cache_capacity = kCapacity;
  args.block_table_i32 = static_cast<int32_t*>(block_table);
  args.block_table_entries = kCapacity;
  args.key_cache_fp8_e4m3 = key_cache;
  args.value_cache_fp8_e4m3 = value_cache;
  args.key_scale = 1.0F;
  args.value_scale = 1.0F;
  args.query_bf16 = query;
  args.gate_bf16 = gate;
  args.output_bf16 = output;
  args.workspace = workspace;
  args.workspace_bytes = workspace_bytes;
  args.cuda_stream = stream;

  q27_prefill_attention_args bad = args;
  --bad.workspace_bytes;
  ExpectStatus(q27_prefill_attention(&bad),
               Q27_PREFILL_ATTENTION_INVALID_ARGUMENT,
               "short target prefill workspace");
  CheckStatus(q27_prefill_attention(&args), "target prefill attention");
  CheckCuda(cudaStreamSynchronize(stream), "target prefill attention sync");
  const uint32_t invalid = HostCopy(
      q27_prefill_attention_invalid_count(workspace, workspace_bytes), 1)[0];
  if (corrupt_table) {
    if (invalid == 0)
      throw std::runtime_error("invalid target page table was not detected");
    const std::vector<uint8_t> key_page_zero = HostCopy(
        static_cast<uint8_t*>(key_cache), kKvColumns);
    const std::vector<uint8_t> value_page_zero = HostCopy(
        static_cast<uint8_t*>(value_cache), kKvColumns);
    if (std::any_of(key_page_zero.begin(), key_page_zero.end(),
                    [](uint8_t value) { return value != 0; }) ||
        std::any_of(value_page_zero.begin(), value_page_zero.end(),
                    [](uint8_t value) { return value != 0; })) {
      throw std::runtime_error(
          "invalid current target page corrupted physical page zero");
    }
  } else {
    if (invalid != 0)
      throw std::runtime_error("valid target page table failed validation");
    const std::vector<uint16_t> query_host = HostCopy(
        static_cast<uint16_t*>(query), kTile * kQColumns);
    const std::vector<uint16_t> gate_host = HostCopy(
        static_cast<uint16_t*>(gate), kTile * kQColumns);
    const std::vector<uint16_t> output_host = HostCopy(
        static_cast<uint16_t*>(output), kTile * kQColumns);
    const std::vector<uint8_t> key_host = HostCopy(
        static_cast<uint8_t*>(key_cache), kCacheBytes);
    const std::vector<uint8_t> value_host = HostCopy(
        static_cast<uint8_t*>(value_cache), kCacheBytes);
    for (uint32_t token = 0; token < valid_tokens; ++token) {
      for (uint32_t head : {0U, 5U, 6U, 23U}) {
        for (uint32_t dimension : {0U, 255U}) {
          const uint64_t index =
              (static_cast<uint64_t>(token) * kQHeads + head) * kHeadDim +
              dimension;
          ExpectNear(FromBf16(output_host[index]),
                     Reference(query_host, gate_host, key_host, value_host,
                               token, head, dimension),
                     0.20F, "target paged FP8 prefill output");
        }
      }
    }
    const uint64_t padded =
        static_cast<uint64_t>(valid_tokens) * kQColumns;
    if (valid_tokens < kTile && output_host[padded] != 0) {
      throw std::runtime_error("target prefill wrote a padded output row");
    }
  }

  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  for (void* pointer : {q_gate, key, value, q_norm, k_norm, rope, block_table,
                        key_cache, value_cache, query, gate, output, workspace}) {
    cudaFree(pointer);
  }
}

void RunTimingCase(uint32_t committed_tokens) {
  const uint32_t capacity = committed_tokens + kTile;
  const uint64_t cache_bytes =
      static_cast<uint64_t>(capacity) * kKvColumns;
  void* q_gate = DeviceBytes(Q27_PREFILL_ATTENTION_Q_GATE_BYTES);
  void* key = DeviceBytes(Q27_PREFILL_ATTENTION_KV_INPUT_BYTES);
  void* value = DeviceBytes(Q27_PREFILL_ATTENTION_KV_INPUT_BYTES);
  void* q_norm = DeviceBytes(kHeadDim * 2ULL);
  void* k_norm = DeviceBytes(kHeadDim * 2ULL);
  void* rope = DeviceBytes(
      static_cast<uint64_t>(capacity) * Q27_ATTENTION_ROTARY_DIM * 4ULL);
  void* block_table = DeviceBytes(capacity * sizeof(int32_t));
  void* key_cache = DeviceBytes(cache_bytes);
  void* value_cache = DeviceBytes(cache_bytes);
  void* query = DeviceBytes(Q27_PREFILL_ATTENTION_QUERY_BYTES);
  void* gate = DeviceBytes(Q27_PREFILL_ATTENTION_QUERY_BYTES);
  void* output = DeviceBytes(Q27_PREFILL_ATTENTION_QUERY_BYTES);
  const uint64_t workspace_bytes =
      Q27_PREFILL_ATTENTION_WORKSPACE_BYTES(capacity);
  void* workspace = DeviceBytes(workspace_bytes);
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate timing");
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate start");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  InitializeBlockTable<<<64, 256, 0, stream>>>(
      static_cast<int32_t*>(block_table), capacity);
  InitializeRope<<<128, 256, 0, stream>>>(static_cast<float*>(rope),
                                          capacity);
  InitializeCommitted<<<128, 256, 0, stream>>>(
      static_cast<__nv_fp8_e4m3*>(key_cache),
      static_cast<__nv_fp8_e4m3*>(value_cache),
      static_cast<const int32_t*>(block_table), committed_tokens);
  InitializeInputs<<<128, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(q_gate),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(q_norm),
      static_cast<__nv_bfloat16*>(k_norm), committed_tokens);
  CheckCuda(cudaGetLastError(), "prefill timing initialization");

  q27_prefill_attention_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_PREFILL_ATTENTION_ABI_VERSION;
  args.q_gate_bf16 = q_gate;
  args.key_bf16 = key;
  args.value_bf16 = value;
  args.q_norm_weight_bf16 = q_norm;
  args.k_norm_weight_bf16 = k_norm;
  args.rope_cos_sin_f32 = static_cast<float*>(rope);
  args.rope_row_stride_elements = Q27_ATTENTION_ROTARY_DIM;
  args.rope_position_capacity = capacity;
  args.valid_tokens = kTile;
  args.committed_tokens = committed_tokens;
  args.cache_capacity = capacity;
  args.block_table_i32 = static_cast<int32_t*>(block_table);
  args.block_table_entries = capacity;
  args.key_cache_fp8_e4m3 = key_cache;
  args.value_cache_fp8_e4m3 = value_cache;
  args.key_scale = 1.0F;
  args.value_scale = 1.0F;
  args.query_bf16 = query;
  args.gate_bf16 = gate;
  args.output_bf16 = output;
  args.workspace = workspace;
  args.workspace_bytes = workspace_bytes;
  args.cuda_stream = stream;

  CheckStatus(q27_prefill_attention(&args), "target prefill timing warmup");
  CheckCuda(cudaStreamSynchronize(stream), "target prefill timing warmup sync");
  constexpr uint32_t kIterations = 5;
  CheckCuda(cudaEventRecord(start, stream), "target prefill timing start");
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    CheckStatus(q27_prefill_attention(&args), "target prefill timing");
  }
  CheckCuda(cudaEventRecord(stop, stream), "target prefill timing stop");
  CheckCuda(cudaEventSynchronize(stop), "target prefill timing sync");
  float elapsed_ms = 0.0F;
  CheckCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
            "target prefill elapsed time");
  const uint32_t invalid = HostCopy(
      q27_prefill_attention_invalid_count(workspace, workspace_bytes), 1)[0];
  if (invalid != 0) {
    throw std::runtime_error("target prefill timing page table invalid");
  }
  std::cout << "q27_prefill_attention_timing valid_tokens=" << kTile
            << " committed_tokens=" << committed_tokens
            << " iterations=" << kIterations
            << " mean_ms=" << elapsed_ms / kIterations << '\n';

  CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
  CheckCuda(cudaEventDestroy(start), "cudaEventDestroy start");
  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy timing");
  for (void* pointer : {q_gate, key, value, q_norm, k_norm, rope, block_table,
                        key_cache, value_cache, query, gate, output, workspace}) {
    cudaFree(pointer);
  }
}

}  // namespace

int main() {
  try {
    RunCase(/*valid_tokens=*/5, /*corrupt_table=*/false);
    RunCase(/*valid_tokens=*/2, /*corrupt_table=*/false);
    RunCase(/*valid_tokens=*/5, /*corrupt_table=*/true);
    RunTimingCase(/*committed_tokens=*/64);
    RunTimingCase(/*committed_tokens=*/4096);
    RunTimingCase(/*committed_tokens=*/12288);
    std::cout << "q27_prefill_attention_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_prefill_attention_smoke: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}

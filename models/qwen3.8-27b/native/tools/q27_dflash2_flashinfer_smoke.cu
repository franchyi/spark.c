#include "q27_dflash2_flashinfer.h"

#include <cuda_bf16.h>
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

constexpr uint32_t kHistory = 3;
constexpr uint64_t kPrefix = kHistory;
constexpr uint32_t kTokens = Q27_DFLASH2_BLOCK_SIZE;
constexpr uint32_t kQHeads = Q27_DFLASH2_QUERY_HEADS;
constexpr uint32_t kKvHeads = Q27_DFLASH2_KV_HEADS;
constexpr uint32_t kHeadDim = Q27_DFLASH2_HEAD_DIM;
constexpr uint32_t kQColumns = kQHeads * kHeadDim;
constexpr uint32_t kKvColumns = kKvHeads * kHeadDim;
constexpr uint32_t kLayer = 2;
constexpr float kScale = 0.08838834764831845F;

uint16_t Bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>(
      (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

float FromBf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

float RoundBf16(float value) { return FromBf16(Bf16(value)); }

float QueryValue(uint32_t token, uint32_t head) {
  return RoundBf16(0.2F * static_cast<float>(token + 1) +
                   0.01F * static_cast<float>(head));
}

float KeyValue(uint32_t absolute_token, uint32_t kv_head) {
  return RoundBf16(0.125F * static_cast<float>(absolute_token + 1) +
                   0.01F * static_cast<float>(kv_head));
}

float ValueValue(uint32_t absolute_token, uint32_t kv_head) {
  return RoundBf16(static_cast<float>(absolute_token + 1) +
                   0.25F * static_cast<float>(kv_head));
}

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void CheckStatus(q27_dflash2_status status, const char* operation) {
  if (status.code != Q27_DFLASH2_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void ExpectStatus(q27_dflash2_status status, int32_t code,
                  const char* operation) {
  if (status.code != code) {
    throw std::runtime_error(std::string(operation) + " status mismatch");
  }
}

void* DeviceBytes(uint64_t bytes) {
  void* device = nullptr;
  CheckCuda(cudaMalloc(&device, bytes), "cudaMalloc");
  CheckCuda(cudaMemset(device, 0, bytes), "cudaMemset");
  return device;
}

template <typename T>
std::vector<T> HostCopy(const T* device, uint64_t elements) {
  std::vector<T> host(elements);
  CheckCuda(cudaMemcpy(host.data(), device, elements * sizeof(T),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H");
  return host;
}

__global__ void InitializeRing(__nv_bfloat16* key_cache,
                               __nv_bfloat16* value_cache,
                               uint64_t* position_tags) {
  const uint32_t elements = kHistory * kKvColumns;
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kKvColumns;
    const uint32_t within = index - token * kKvColumns;
    const uint32_t head = within / kHeadDim;
    const uint32_t dimension = within - head * kHeadDim;
    const uint64_t cache =
        (static_cast<uint64_t>(kLayer) * Q27_DFLASH2_SLIDING_WINDOW + token) *
            kKvColumns +
        within;
    key_cache[cache] = __float2bfloat16_rn(
        dimension == 0
            ? 0.125F * static_cast<float>(token + 1) +
                  0.01F * static_cast<float>(head)
            : 0.0F);
    value_cache[cache] = __float2bfloat16_rn(
        dimension == 0
            ? static_cast<float>(token + 1) + 0.25F * static_cast<float>(head)
            : 0.0F);
    if (within == 0) position_tags[token] = token;
  }
}

__global__ void InitializeBlock(__nv_bfloat16* q, __nv_bfloat16* k,
                                __nv_bfloat16* v, uint64_t* positions) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTokens * kQColumns; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kQColumns;
    const uint32_t within = index - token * kQColumns;
    const uint32_t head = within / kHeadDim;
    const uint32_t dimension = within - head * kHeadDim;
    q[index] = __float2bfloat16_rn(
        dimension == 0
            ? 0.2F * static_cast<float>(token + 1) +
                  0.01F * static_cast<float>(head)
            : 0.0F);
  }
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTokens * kKvColumns; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kKvColumns;
    const uint32_t within = index - token * kKvColumns;
    const uint32_t head = within / kHeadDim;
    const uint32_t dimension = within - head * kHeadDim;
    const uint32_t absolute_token = kHistory + token;
    k[index] = __float2bfloat16_rn(
        dimension == 0
            ? 0.125F * static_cast<float>(absolute_token + 1) +
                  0.01F * static_cast<float>(head)
            : 0.0F);
    v[index] = __float2bfloat16_rn(
        dimension == 0
            ? static_cast<float>(absolute_token + 1) +
                  0.25F * static_cast<float>(head)
            : 0.0F);
  }
  if (blockIdx.x == 0 && threadIdx.x < kTokens) {
    positions[threadIdx.x] = kPrefix + threadIdx.x;
  }
}

float Reference(uint32_t token, uint32_t q_head) {
  const uint32_t kv_head = q_head / (kQHeads / kKvHeads);
  const float q = QueryValue(token, q_head);
  const uint32_t visible = kHistory + token + 1;
  std::vector<float> logits(visible);
  float maximum = -INFINITY;
  for (uint32_t key = 0; key < visible; ++key) {
    logits[key] = q * KeyValue(key, kv_head) * kScale;
    maximum = std::max(maximum, logits[key]);
  }
  float denominator = 0.0F;
  float numerator = 0.0F;
  for (uint32_t key = 0; key < visible; ++key) {
    const float probability = std::exp(logits[key] - maximum);
    denominator += probability;
    numerator += probability * ValueValue(key, kv_head);
  }
  return numerator / denominator;
}

void ExpectNear(float actual, float expected, float tolerance,
                const char* label) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(label) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void SmokeFlashInfer() {
  void* key_cache = DeviceBytes(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  void* value_cache = DeviceBytes(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  void* tags = DeviceBytes(Q27_DFLASH2_POSITION_TAG_BYTES);
  void* q = DeviceBytes(kTokens * kQColumns * 2ULL);
  void* k = DeviceBytes(kTokens * kKvColumns * 2ULL);
  void* v = DeviceBytes(kTokens * kKvColumns * 2ULL);
  void* positions = DeviceBytes(kTokens * sizeof(uint64_t));
  void* context = DeviceBytes(kTokens * kQColumns * 2ULL);
  void* workspace = DeviceBytes(Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES);
  CheckCuda(cudaMemset(tags, 0xFF, Q27_DFLASH2_POSITION_TAG_BYTES),
            "invalidate fixture tags");

  cudaStream_t stream = nullptr;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  InitializeRing<<<64, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(key_cache),
      static_cast<__nv_bfloat16*>(value_cache),
      static_cast<uint64_t*>(tags));
  InitializeBlock<<<128, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v), static_cast<uint64_t*>(positions));
  CheckCuda(cudaGetLastError(), "FlashInfer fixture initialization");

  q27_dflash2_state_view state{};
  state.struct_size = sizeof(state);
  state.abi_version = Q27_DFLASH2_ABI_VERSION;
  state.key_cache_bf16 = key_cache;
  state.value_cache_bf16 = value_cache;
  state.position_tags_u64 = static_cast<uint64_t*>(tags);
  state.committed_length = kPrefix;
  q27_dflash2_sliding_attention_call call{};
  call.struct_size = sizeof(call);
  call.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  call.layer_index = kLayer;
  call.token_count = kTokens;
  call.positions_u64 = static_cast<uint64_t*>(positions);
  call.q_bf16 = q;
  call.k_bf16 = k;
  call.v_bf16 = v;
  call.context_bf16 = context;
  call.state = &state;
  call.workspace = workspace;
  call.workspace_bytes = Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES;
  call.scale = kScale;
  call.window_left = Q27_DFLASH2_SLIDING_WINDOW - 1;
  call.cuda_stream = stream;

  ExpectStatus(q27_dflash2_flashinfer_sliding_attention(&call, &state),
               Q27_DFLASH2_INVALID_ARGUMENT, "non-null hook user data");
  q27_dflash2_sliding_attention_call bad = call;
  --bad.workspace_bytes;
  ExpectStatus(q27_dflash2_flashinfer_sliding_attention(&bad, nullptr),
               Q27_DFLASH2_INVALID_ARGUMENT, "short FlashInfer workspace");

  const uint16_t* live_layer_key =
      static_cast<uint16_t*>(key_cache) +
      (static_cast<uint64_t>(kLayer) * Q27_DFLASH2_SLIDING_WINDOW) *
          kKvColumns;
  const uint16_t* live_layer_value =
      static_cast<uint16_t*>(value_cache) +
      (static_cast<uint64_t>(kLayer) * Q27_DFLASH2_SLIDING_WINDOW) *
          kKvColumns;
  const std::vector<uint16_t> key_before =
      HostCopy(live_layer_key, kHistory * kKvColumns);
  const std::vector<uint16_t> value_before =
      HostCopy(live_layer_value, kHistory * kKvColumns);
  CheckStatus(q27_dflash2_flashinfer_sliding_attention(&call, nullptr),
              "FlashInfer sliding attention");
  CheckCuda(cudaStreamSynchronize(stream), "FlashInfer attention synchronize");
  const uint32_t invalid = HostCopy(
      q27_dflash2_flashinfer_invalid_count(workspace, call.workspace_bytes), 1)[0];
  if (invalid != 0) {
    throw std::runtime_error("valid tagged ring reported an invariant failure");
  }

  const std::vector<uint16_t> output = HostCopy(
      static_cast<uint16_t*>(context), kTokens * kQColumns);
  for (uint32_t token = 0; token < kTokens; ++token) {
    for (uint32_t head : {0U, 3U, 4U, 31U}) {
      const uint64_t base =
          (static_cast<uint64_t>(token) * kQHeads + head) * kHeadDim;
      ExpectNear(FromBf16(output[base]), Reference(token, head), 0.08F,
                 "FlashInfer context dim0");
      ExpectNear(FromBf16(output[base + kHeadDim - 1]), 0.0F, 0.001F,
                 "FlashInfer zero dimension");
    }
  }

  const std::vector<uint16_t> key_after =
      HostCopy(live_layer_key, kHistory * kKvColumns);
  const std::vector<uint16_t> value_after =
      HostCopy(live_layer_value, kHistory * kKvColumns);
  if (key_before != key_after || value_before != value_after) {
    throw std::runtime_error("FlashInfer hook mutated committed live KV");
  }

  const uint64_t stale = 999;
  CheckCuda(cudaMemcpy(static_cast<uint64_t*>(tags) + 1, &stale, sizeof(stale),
                       cudaMemcpyHostToDevice),
            "corrupt one ring tag");
  CheckStatus(q27_dflash2_flashinfer_sliding_attention(&call, nullptr),
              "FlashInfer stale-tag attention");
  CheckCuda(cudaStreamSynchronize(stream), "stale-tag synchronize");
  const uint32_t stale_count = HostCopy(
      q27_dflash2_flashinfer_invalid_count(workspace, call.workspace_bytes), 1)[0];
  if (stale_count == 0) {
    throw std::runtime_error("stale tagged ring was not detected");
  }

  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  for (void* pointer :
       {key_cache, value_cache, tags, q, k, v, positions, context, workspace}) {
    cudaFree(pointer);
  }
}

}  // namespace

int main() {
  try {
    SmokeFlashInfer();
    std::cout << "q27_dflash2_flashinfer_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_dflash2_flashinfer_smoke: FAIL: " << error.what() << '\n';
    return 1;
  }
}

#include "q27_dflash2_kv.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTokens = 4;
constexpr uint32_t kHidden = Q27_DFLASH2_HIDDEN_SIZE;
constexpr uint32_t kKvColumns =
    Q27_DFLASH2_KV_HEADS * Q27_DFLASH2_HEAD_DIM;
constexpr uint64_t kFirstPosition = 2046;

uint16_t Bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>(
      (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

float FromBf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

float RoundBf16(float value) { return FromBf16(Bf16(value)); }

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
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

__global__ void InitializeContext(__nv_bfloat16* context) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTokens * kHidden; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kHidden;
    const uint32_t column = index - token * kHidden;
    context[index] = __float2bfloat16_rn(
        column == 0 ? static_cast<float>(token + 1) : 0.0F);
  }
}

__global__ void InitializeNorm(__nv_bfloat16* gamma) {
  if (threadIdx.x < Q27_DFLASH2_HEAD_DIM) {
    gamma[threadIdx.x] = __float2bfloat16_rn(1.0F);
  }
}

__global__ void InitializeKWeight(__nv_bfloat16* weight, float scale) {
  const uint32_t head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head < Q27_DFLASH2_KV_HEADS) {
    const uint64_t first =
        static_cast<uint64_t>(head * Q27_DFLASH2_HEAD_DIM) * kHidden;
    const uint64_t second =
        static_cast<uint64_t>(head * Q27_DFLASH2_HEAD_DIM + 64) * kHidden;
    weight[first] = __float2bfloat16_rn(scale);
    weight[second] = __float2bfloat16_rn(2.0F * scale);
  }
}

__global__ void InitializeVWeight(__nv_bfloat16* weight, float scale) {
  const uint32_t head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head < Q27_DFLASH2_KV_HEADS) {
    const uint64_t row =
        static_cast<uint64_t>(head * Q27_DFLASH2_HEAD_DIM) * kHidden;
    weight[row] = __float2bfloat16_rn(scale);
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                const char* label) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(label) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void SmokeContextKv() {
  const uint64_t kv_weight_bytes =
      static_cast<uint64_t>(kKvColumns) * kHidden * 2ULL;
  std::vector<void*> allocations;
  q27_dflash2_weights weights{};
  weights.struct_size = sizeof(weights);
  weights.abi_version = Q27_DFLASH2_ABI_VERSION;
  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    void* k_weight = DeviceBytes(kv_weight_bytes);
    void* v_weight = DeviceBytes(kv_weight_bytes);
    void* k_norm = DeviceBytes(Q27_DFLASH2_HEAD_DIM * 2ULL);
    allocations.push_back(k_weight);
    allocations.push_back(v_weight);
    allocations.push_back(k_norm);
    weights.layers[layer].k_proj = {k_weight, kv_weight_bytes};
    weights.layers[layer].v_proj = {v_weight, kv_weight_bytes};
    weights.layers[layer].k_norm = {k_norm, Q27_DFLASH2_HEAD_DIM * 2ULL};
  }
  void* context = DeviceBytes(kTokens * kHidden * 2ULL);
  void* frequencies =
      DeviceBytes(Q27_DFLASH2_ATTENTION_ROPE_FREQUENCY_BYTES);
  void* rope_cache =
      DeviceBytes(kTokens * Q27_DFLASH2_KV_ROPE_CACHE_BYTES_PER_TOKEN);
  void* k_scratch =
      DeviceBytes(kTokens * Q27_DFLASH2_KV_SCRATCH_BYTES_PER_TOKEN);
  void* v_scratch =
      DeviceBytes(kTokens * Q27_DFLASH2_KV_SCRATCH_BYTES_PER_TOKEN);
  void* key_cache = DeviceBytes(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  void* value_cache = DeviceBytes(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  void* tags = DeviceBytes(Q27_DFLASH2_POSITION_TAG_BYTES);
  for (void* pointer : {context, frequencies, rope_cache, k_scratch, v_scratch,
                        key_cache, value_cache, tags}) {
    allocations.push_back(pointer);
  }

  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  CheckCublas(cublasCreate(&handle), "cublasCreate");
  InitializeContext<<<80, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(context));
  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    InitializeNorm<<<1, 128, 0, stream>>>(static_cast<__nv_bfloat16*>(
        const_cast<void*>(weights.layers[layer].k_norm.data)));
    InitializeKWeight<<<1, 8, 0, stream>>>(static_cast<__nv_bfloat16*>(
        const_cast<void*>(weights.layers[layer].k_proj.data)),
        static_cast<float>(layer + 1));
    InitializeVWeight<<<1, 8, 0, stream>>>(static_cast<__nv_bfloat16*>(
        const_cast<void*>(weights.layers[layer].v_proj.data)),
        static_cast<float>(10 + layer));
  }
  CheckCuda(cudaGetLastError(), "KV fixture initialization launch");

  q27_dflash2_rope_init_args rope{};
  rope.struct_size = sizeof(rope);
  rope.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  rope.inverse_frequencies_f32 = static_cast<float*>(frequencies);
  rope.cuda_stream = stream;
  CheckStatus(q27_dflash2_initialize_rope(&rope), "initialize rope");

  q27_dflash2_state_view state{};
  state.struct_size = sizeof(state);
  state.abi_version = Q27_DFLASH2_ABI_VERSION;
  state.key_cache_bf16 = key_cache;
  state.value_cache_bf16 = value_cache;
  state.position_tags_u64 = static_cast<uint64_t*>(tags);
  state.committed_length = 123;
  q27_dflash2_kv_reset_args reset{};
  reset.struct_size = sizeof(reset);
  reset.abi_version = Q27_DFLASH2_KV_ABI_VERSION;
  reset.state = &state;
  reset.cuda_stream = stream;
  CheckStatus(q27_dflash2_reset_kv(&reset), "reset KV tags");
  if (state.committed_length != 0) {
    throw std::runtime_error("KV reset did not clear committed length");
  }

  q27_dflash2_kv_materialize_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_DFLASH2_KV_ABI_VERSION;
  args.weights = &weights;
  args.context_hidden_bf16 = context;
  args.first_position = kFirstPosition;
  args.token_count = kTokens;
  args.rms_epsilon = 1.0e-6F;
  args.rope_inverse_frequencies_f32 = static_cast<float*>(frequencies);
  args.rope_cache_f32 = static_cast<float*>(rope_cache);
  args.k_scratch_bf16 = k_scratch;
  args.v_scratch_bf16 = v_scratch;
  args.state = &state;
  args.cublas_handle = handle;
  args.cuda_stream = stream;

  q27_dflash2_kv_materialize_args bad = args;
  bad.token_count = Q27_DFLASH2_KV_MAX_CHUNK_TOKENS + 1;
  ExpectStatus(q27_dflash2_materialize_context_kv(&bad),
               Q27_DFLASH2_INVALID_ARGUMENT, "oversized KV chunk");
  bad = args;
  bad.first_position = Q27_DFLASH2_MAX_POSITION - kTokens + 1;
  ExpectStatus(q27_dflash2_materialize_context_kv(&bad),
               Q27_DFLASH2_INVALID_ARGUMENT, "KV position overflow");
  CheckStatus(q27_dflash2_materialize_context_kv(&args),
              "materialize context KV");
  CheckCuda(cudaStreamSynchronize(stream), "context KV synchronize");

  const std::vector<uint64_t> host_tags = HostCopy(
      static_cast<uint64_t*>(tags), Q27_DFLASH2_SLIDING_WINDOW);
  for (uint32_t token = 0; token < kTokens; ++token) {
    const uint64_t position = kFirstPosition + token;
    const uint32_t slot = position & (Q27_DFLASH2_SLIDING_WINDOW - 1);
    if (host_tags[slot] != position) {
      throw std::runtime_error("KV ring position tag mismatch");
    }
  }
  if (host_tags[2] != UINT64_MAX) {
    throw std::runtime_error("KV reset left an untouched tag valid");
  }

  const std::vector<uint16_t> host_k = HostCopy(
      static_cast<uint16_t*>(key_cache), Q27_DFLASH2_ONE_KV_CACHE_BYTES / 2);
  const std::vector<uint16_t> host_v = HostCopy(
      static_cast<uint16_t*>(value_cache), Q27_DFLASH2_ONE_KV_CACHE_BYTES / 2);
  for (uint32_t layer : {0U, static_cast<uint32_t>(Q27_DFLASH2_LAYERS - 1)}) {
    const float layer_scale = static_cast<float>(layer + 1);
    for (uint32_t token = 0; token < kTokens; ++token) {
      const float x = static_cast<float>(token + 1);
      const float raw0 = RoundBf16(layer_scale * x);
      const float raw64 = RoundBf16(2.0F * layer_scale * x);
      const float inverse_rms =
          1.0F / std::sqrt((raw0 * raw0 + raw64 * raw64) / 128.0F + 1e-6F);
      const float norm0 = RoundBf16(raw0 * inverse_rms);
      const float norm64 = RoundBf16(raw64 * inverse_rms);
      const float angle = static_cast<float>(kFirstPosition + token);
      const float expected0 =
          RoundBf16(norm0 * std::cos(angle) - norm64 * std::sin(angle));
      const float expected64 =
          RoundBf16(norm64 * std::cos(angle) + norm0 * std::sin(angle));
      const uint32_t slot =
          static_cast<uint32_t>(kFirstPosition + token) &
          (Q27_DFLASH2_SLIDING_WINDOW - 1);
      for (uint32_t head :
           {0U, static_cast<uint32_t>(Q27_DFLASH2_KV_HEADS - 1)}) {
        const uint64_t base =
            (static_cast<uint64_t>(layer) * Q27_DFLASH2_SLIDING_WINDOW + slot) *
                kKvColumns +
            head * Q27_DFLASH2_HEAD_DIM;
        ExpectNear(FromBf16(host_k[base]), expected0, 0.05F,
                   "context K dim0");
        ExpectNear(FromBf16(host_k[base + 64]), expected64, 0.05F,
                   "context K dim64");
        ExpectNear(FromBf16(host_v[base]),
                   static_cast<float>(10 + layer) * x, 0.05F,
                   "context V projection");
      }
    }
  }

  CheckCublas(cublasDestroy(handle), "cublasDestroy");
  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  for (void* pointer : allocations) cudaFree(pointer);
}

}  // namespace

int main() {
  try {
    SmokeContextKv();
    std::cout << "q27_dflash2_kv_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_dflash2_kv_smoke: FAIL: " << error.what() << '\n';
    return 1;
  }
}

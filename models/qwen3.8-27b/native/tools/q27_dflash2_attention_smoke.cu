#include "q27_dflash2_attention.h"
#include "q27_dflash2_flashinfer.h"

#include <cublas_v2.h>
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

constexpr uint32_t kTokens = Q27_DFLASH2_BLOCK_SIZE;
constexpr uint32_t kHidden = Q27_DFLASH2_HIDDEN_SIZE;
constexpr uint32_t kQColumns =
    Q27_DFLASH2_QUERY_HEADS * Q27_DFLASH2_HEAD_DIM;
constexpr uint32_t kKvColumns =
    Q27_DFLASH2_KV_HEADS * Q27_DFLASH2_HEAD_DIM;

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

__global__ void InitializeInput(__nv_bfloat16* input, uint64_t* positions) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTokens * kHidden; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kHidden;
    const uint32_t column = index - token * kHidden;
    input[index] = __float2bfloat16_rn(
        column == 0 ? static_cast<float>(token + 1) : 0.0F);
  }
  if (blockIdx.x == 0 && threadIdx.x < kTokens) {
    positions[threadIdx.x] = threadIdx.x;
  }
}

__global__ void InitializeNorm(__nv_bfloat16* gamma) {
  if (threadIdx.x < Q27_DFLASH2_HEAD_DIM) {
    gamma[threadIdx.x] = __float2bfloat16_rn(1.0F);
  }
}

__global__ void InitializeQkWeight(__nv_bfloat16* weight, uint32_t heads,
                                   uint32_t input_columns) {
  const uint32_t head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head < heads) {
    const uint64_t first =
        static_cast<uint64_t>(head * Q27_DFLASH2_HEAD_DIM) * input_columns;
    const uint64_t second =
        static_cast<uint64_t>(head * Q27_DFLASH2_HEAD_DIM + 64) *
        input_columns;
    weight[first] = __float2bfloat16_rn(1.0F);
    weight[second] = __float2bfloat16_rn(2.0F);
  }
}

__global__ void InitializeVWeight(__nv_bfloat16* weight) {
  const uint32_t head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head < Q27_DFLASH2_KV_HEADS) {
    const uint64_t row =
        static_cast<uint64_t>(head * Q27_DFLASH2_HEAD_DIM) * kHidden;
    weight[row] = __float2bfloat16_rn(0.5F);
  }
}

__global__ void InitializeOWeight(__nv_bfloat16* weight) {
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < kHidden;
       row += blockDim.x * gridDim.x) {
    weight[static_cast<uint64_t>(row) * kQColumns] =
        __float2bfloat16_rn(0.25F);
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

float ReferenceContext(const std::vector<uint16_t>& q,
                       const std::vector<uint16_t>& k,
                       const std::vector<uint16_t>& v, uint32_t token,
                       uint32_t q_head, uint32_t dimension) {
  const uint32_t kv_head =
      q_head / (Q27_DFLASH2_QUERY_HEADS / Q27_DFLASH2_KV_HEADS);
  std::vector<float> logits(token + 1);
  float maximum = -INFINITY;
  const uint64_t q_base =
      (static_cast<uint64_t>(token) * Q27_DFLASH2_QUERY_HEADS + q_head) *
      Q27_DFLASH2_HEAD_DIM;
  for (uint32_t key_token = 0; key_token <= token; ++key_token) {
    const uint64_t k_base =
        (static_cast<uint64_t>(key_token) * Q27_DFLASH2_KV_HEADS + kv_head) *
        Q27_DFLASH2_HEAD_DIM;
    float dot = 0.0F;
    for (uint32_t column = 0; column < Q27_DFLASH2_HEAD_DIM; ++column) {
      dot += FromBf16(q[q_base + column]) * FromBf16(k[k_base + column]);
    }
    logits[key_token] = dot * 0.08838834764831845F;
    maximum = std::max(maximum, logits[key_token]);
  }
  float denominator = 0.0F;
  float numerator = 0.0F;
  for (uint32_t key_token = 0; key_token <= token; ++key_token) {
    const float probability = std::exp(logits[key_token] - maximum);
    const uint64_t v_index =
        (static_cast<uint64_t>(key_token) * Q27_DFLASH2_KV_HEADS + kv_head) *
            Q27_DFLASH2_HEAD_DIM +
        dimension;
    denominator += probability;
    numerator += probability * FromBf16(v[v_index]);
  }
  return numerator / denominator;
}

void SmokeAttentionShell() {
  const uint64_t q_weight_bytes =
      static_cast<uint64_t>(kQColumns) * kHidden * 2ULL;
  const uint64_t kv_weight_bytes =
      static_cast<uint64_t>(kKvColumns) * kHidden * 2ULL;
  const uint64_t o_weight_bytes =
      static_cast<uint64_t>(kHidden) * kQColumns * 2ULL;
  void* d_q_weight = DeviceBytes(q_weight_bytes);
  void* d_k_weight = DeviceBytes(kv_weight_bytes);
  void* d_v_weight = DeviceBytes(kv_weight_bytes);
  void* d_o_weight = DeviceBytes(o_weight_bytes);
  void* d_q_norm = DeviceBytes(Q27_DFLASH2_HEAD_DIM * 2ULL);
  void* d_k_norm = DeviceBytes(Q27_DFLASH2_HEAD_DIM * 2ULL);
  void* d_input = DeviceBytes(Q27_DFLASH2_ATTENTION_OUTPUT_BYTES);
  void* d_positions = DeviceBytes(kTokens * sizeof(uint64_t));
  void* d_frequencies =
      DeviceBytes(Q27_DFLASH2_ATTENTION_ROPE_FREQUENCY_BYTES);
  void* d_rope_cache = DeviceBytes(Q27_DFLASH2_ATTENTION_ROPE_CACHE_BYTES);
  void* d_q = DeviceBytes(Q27_DFLASH2_ATTENTION_Q_BYTES);
  void* d_k = DeviceBytes(Q27_DFLASH2_ATTENTION_KV_BYTES);
  void* d_v = DeviceBytes(Q27_DFLASH2_ATTENTION_KV_BYTES);
  void* d_context = DeviceBytes(Q27_DFLASH2_ATTENTION_CONTEXT_BYTES);
  void* d_output = DeviceBytes(Q27_DFLASH2_ATTENTION_OUTPUT_BYTES);
  void* d_key_cache = DeviceBytes(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  void* d_value_cache = DeviceBytes(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  void* d_tags = DeviceBytes(Q27_DFLASH2_POSITION_TAG_BYTES);
  void* d_workspace =
      DeviceBytes(Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES);
  CheckCuda(cudaMemset(d_tags, 0xFF, Q27_DFLASH2_POSITION_TAG_BYTES),
            "invalidate draft KV tags");

  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  CheckCublas(cublasCreate(&handle), "cublasCreate");
  InitializeInput<<<160, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(d_input),
      static_cast<uint64_t*>(d_positions));
  InitializeNorm<<<1, 128, 0, stream>>>(
      static_cast<__nv_bfloat16*>(d_q_norm));
  InitializeNorm<<<1, 128, 0, stream>>>(
      static_cast<__nv_bfloat16*>(d_k_norm));
  InitializeQkWeight<<<1, 32, 0, stream>>>(
      static_cast<__nv_bfloat16*>(d_q_weight), Q27_DFLASH2_QUERY_HEADS,
      kHidden);
  InitializeQkWeight<<<1, 8, 0, stream>>>(
      static_cast<__nv_bfloat16*>(d_k_weight), Q27_DFLASH2_KV_HEADS, kHidden);
  InitializeVWeight<<<1, 8, 0, stream>>>(
      static_cast<__nv_bfloat16*>(d_v_weight));
  InitializeOWeight<<<20, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(d_o_weight));
  CheckCuda(cudaGetLastError(), "fixture initialization launch");

  q27_dflash2_rope_init_args rope{};
  rope.struct_size = sizeof(rope);
  rope.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  rope.inverse_frequencies_f32 = static_cast<float*>(d_frequencies);
  rope.cuda_stream = stream;
  CheckStatus(q27_dflash2_initialize_rope(&rope), "initialize rope");

  q27_dflash2_layer_weights weights{};
  weights.q_proj = {d_q_weight, q_weight_bytes};
  weights.k_proj = {d_k_weight, kv_weight_bytes};
  weights.v_proj = {d_v_weight, kv_weight_bytes};
  weights.o_proj = {d_o_weight, o_weight_bytes};
  weights.q_norm = {d_q_norm, Q27_DFLASH2_HEAD_DIM * 2ULL};
  weights.k_norm = {d_k_norm, Q27_DFLASH2_HEAD_DIM * 2ULL};
  q27_dflash2_state_view state{};
  state.struct_size = sizeof(state);
  state.abi_version = Q27_DFLASH2_ABI_VERSION;
  state.key_cache_bf16 = d_key_cache;
  state.value_cache_bf16 = d_value_cache;
  state.position_tags_u64 = static_cast<uint64_t*>(d_tags);
  state.committed_length = 0;

  q27_dflash2_attention_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  args.layer_index = 2;
  args.weights = &weights;
  args.input_bf16 = d_input;
  args.positions_u64 = static_cast<uint64_t*>(d_positions);
  args.rope_inverse_frequencies_f32 = static_cast<float*>(d_frequencies);
  args.rope_cache_f32 = static_cast<float*>(d_rope_cache);
  args.q_bf16 = d_q;
  args.k_bf16 = d_k;
  args.v_bf16 = d_v;
  args.context_bf16 = d_context;
  args.output_bf16 = d_output;
  args.state = &state;
  args.workspace = d_workspace;
  args.workspace_bytes = Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES;
  args.rms_epsilon = 1.0e-6F;
  args.cublas_handle = handle;
  args.cuda_stream = stream;

  q27_dflash2_attention_args bad = args;
  --bad.workspace_bytes;
  ExpectStatus(q27_dflash2_attention_forward(&bad),
               Q27_DFLASH2_INVALID_ARGUMENT, "required workspace");
  bad = args;
  bad.output_bf16 = d_input;
  ExpectStatus(q27_dflash2_attention_forward(&bad),
               Q27_DFLASH2_INVALID_ARGUMENT, "alias rejection");
  CheckStatus(q27_dflash2_attention_forward(&args), "fixed attention");
  CheckCuda(cudaStreamSynchronize(stream), "fixed attention synchronize");
  const uint32_t invalid = HostCopy(
      q27_dflash2_flashinfer_invalid_count(d_workspace,
                                           args.workspace_bytes),
      1)[0];
  if (invalid != 0) {
    throw std::runtime_error("fixed attention reported an invariant failure");
  }

  const std::vector<uint16_t> q = HostCopy(static_cast<uint16_t*>(d_q),
                                           kTokens * kQColumns);
  const std::vector<uint16_t> k = HostCopy(static_cast<uint16_t*>(d_k),
                                           kTokens * kKvColumns);
  const std::vector<uint16_t> v = HostCopy(static_cast<uint16_t*>(d_v),
                                           kTokens * kKvColumns);
  const std::vector<uint16_t> context = HostCopy(
      static_cast<uint16_t*>(d_context), kTokens * kQColumns);
  const std::vector<uint16_t> output = HostCopy(
      static_cast<uint16_t*>(d_output), kTokens * kHidden);
  const std::vector<float> rope_cache = HostCopy(
      static_cast<float*>(d_rope_cache), kTokens * 64ULL * 2ULL);

  for (uint32_t token = 0; token < kTokens; ++token) {
    const float x = static_cast<float>(token + 1);
    const float inverse_rms = 1.0F / std::sqrt(5.0F * x * x / 128.0F + 1e-6F);
    const float norm0 = RoundBf16(x * inverse_rms);
    const float norm64 = RoundBf16(2.0F * x * inverse_rms);
    const float expected0 =
        RoundBf16(norm0 * std::cos(static_cast<float>(token)) -
                   norm64 * std::sin(static_cast<float>(token)));
    const float expected64 =
        RoundBf16(norm64 * std::cos(static_cast<float>(token)) +
                   norm0 * std::sin(static_cast<float>(token)));
    for (uint32_t head :
         {0U, static_cast<uint32_t>(Q27_DFLASH2_QUERY_HEADS - 1)}) {
      const uint64_t base =
          (static_cast<uint64_t>(token) * Q27_DFLASH2_QUERY_HEADS + head) * 128;
      ExpectNear(FromBf16(q[base]), expected0, 0.04F, "Q RoPE dim0");
      ExpectNear(FromBf16(q[base + 64]), expected64, 0.04F,
                 "Q RoPE dim64");
    }
    for (uint32_t head :
         {0U, static_cast<uint32_t>(Q27_DFLASH2_KV_HEADS - 1)}) {
      const uint64_t base =
          (static_cast<uint64_t>(token) * Q27_DFLASH2_KV_HEADS + head) * 128;
      ExpectNear(FromBf16(k[base]), expected0, 0.04F, "K RoPE dim0");
      ExpectNear(FromBf16(k[base + 64]), expected64, 0.04F,
                 "K RoPE dim64");
      ExpectNear(FromBf16(v[base]), 0.5F * x, 0.01F, "V projection");
    }
    const uint64_t rope = static_cast<uint64_t>(token) * 64 * 2;
    ExpectNear(rope_cache[rope], std::cos(static_cast<float>(token)), 1.0e-6F,
               "RoPE cosine");
    ExpectNear(rope_cache[rope + 1], std::sin(static_cast<float>(token)),
               1.0e-6F, "RoPE sine");
    for (uint32_t head : {0U, 3U, 4U, 31U}) {
      const uint64_t base =
          (static_cast<uint64_t>(token) * Q27_DFLASH2_QUERY_HEADS + head) *
          Q27_DFLASH2_HEAD_DIM;
      ExpectNear(FromBf16(context[base]),
                 ReferenceContext(q, k, v, token, head, 0), 0.08F,
                 "fixed attention context dim0");
      ExpectNear(FromBf16(context[base + Q27_DFLASH2_HEAD_DIM - 1]),
                 ReferenceContext(q, k, v, token, head,
                                  Q27_DFLASH2_HEAD_DIM - 1),
                 0.02F, "fixed attention context dim127");
    }
    const float projected =
        0.25F * ReferenceContext(q, k, v, token, 0, 0);
    for (uint32_t column : {0U, static_cast<uint32_t>(kHidden - 1)}) {
      const uint64_t index = static_cast<uint64_t>(token) * kHidden + column;
      ExpectNear(FromBf16(output[index]), projected, 0.04F,
                 "O projection");
    }
  }

  CheckCublas(cublasDestroy(handle), "cublasDestroy");
  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  for (void* pointer : {d_q_weight, d_k_weight, d_v_weight, d_o_weight,
                        d_q_norm, d_k_norm, d_input, d_positions, d_frequencies,
                        d_rope_cache, d_q, d_k, d_v, d_context, d_output,
                        d_key_cache, d_value_cache, d_tags, d_workspace}) {
    cudaFree(pointer);
  }
}

}  // namespace

int main() {
  try {
    SmokeAttentionShell();
    std::cout << "q27_dflash2_attention_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_dflash2_attention_smoke: FAIL: " << error.what() << '\n';
    return 1;
  }
}

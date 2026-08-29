// Fixed batched embedding and Gemma RMSNorm for Qwen3.8-27B prefill.

#include "q27_prefill_core.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;
thread_local std::string g_error;

q27_prefill_core_status Ok() { return {Q27_PREFILL_CORE_OK, "ok"}; }

q27_prefill_core_status Invalid(const char* message) {
  return {Q27_PREFILL_CORE_INVALID_ARGUMENT, message};
}

q27_prefill_core_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_PREFILL_CORE_CUDA_ERROR, g_error.c_str()};
}

__device__ float WarpSum(float value) {
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

__global__ void GatherEmbedding(const uint32_t* token_ids,
                                const __nv_bfloat16* embedding,
                                __nv_bfloat16* output, uint32_t valid_tokens,
                                uint32_t* invalid_count) {
  const uint32_t row = blockIdx.x;
  const uint32_t token = row < valid_tokens ? token_ids[row] : 0;
  const bool valid = row < valid_tokens && token < Q27_PREFILL_CORE_VOCAB;
  if (row < valid_tokens && token >= Q27_PREFILL_CORE_VOCAB &&
      threadIdx.x == 0) {
    atomicAdd(invalid_count, 1U);
  }
  const uint64_t output_base =
      static_cast<uint64_t>(row) * Q27_PREFILL_CORE_HIDDEN;
  const uint64_t input_base =
      static_cast<uint64_t>(token) * Q27_PREFILL_CORE_HIDDEN;
  for (uint32_t column = threadIdx.x; column < Q27_PREFILL_CORE_HIDDEN;
       column += blockDim.x) {
    output[output_base + column] =
        valid ? embedding[input_base + column] : __float2bfloat16_rn(0.0F);
  }
}

__global__ void GemmaNormRows(const __nv_bfloat16* input,
                              const __nv_bfloat16* residual,
                              const __nv_bfloat16* weight,
                              __nv_bfloat16* output,
                              __nv_bfloat16* residual_output,
                              uint32_t valid_tokens, bool has_residual,
                              float epsilon) {
  const uint32_t row = blockIdx.x;
  const uint64_t base = static_cast<uint64_t>(row) * Q27_PREFILL_CORE_HIDDEN;
  if (row >= valid_tokens) {
    for (uint32_t column = threadIdx.x; column < Q27_PREFILL_CORE_HIDDEN;
         column += blockDim.x) {
      output[base + column] = __float2bfloat16_rn(0.0F);
      residual_output[base + column] = __float2bfloat16_rn(0.0F);
    }
    return;
  }

  float square_sum = 0.0F;
  for (uint32_t column = threadIdx.x; column < Q27_PREFILL_CORE_HIDDEN;
       column += blockDim.x) {
    float value = __bfloat162float(input[base + column]);
    if (has_residual) value += __bfloat162float(residual[base + column]);
    const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
    residual_output[base + column] = rounded;
    const float norm_value = __bfloat162float(rounded);
    square_sum += norm_value * norm_value;
  }
  square_sum = WarpSum(square_sum);
  __shared__ float warp_sums[kWarps];
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  if (lane == 0) warp_sums[warp] = square_sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? warp_sums[lane] : 0.0F;
    total = WarpSum(total);
    if (lane == 0) warp_sums[0] = total;
  }
  __syncthreads();
  const float inverse_rms =
      rsqrtf(warp_sums[0] / Q27_PREFILL_CORE_HIDDEN + epsilon);
  for (uint32_t column = threadIdx.x; column < Q27_PREFILL_CORE_HIDDEN;
       column += blockDim.x) {
    const float value = __bfloat162float(residual_output[base + column]);
    const float gamma = 1.0F + __bfloat162float(weight[column]);
    output[base + column] = __float2bfloat16_rn(value * inverse_rms * gamma);
  }
}

bool DistinctNormBuffers(const q27_prefill_norm_args* args) {
  return args->input_bf16 != args->output_bf16 &&
         args->input_bf16 != args->residual_output_bf16 &&
         args->output_bf16 != args->residual_output_bf16 &&
         (!args->has_residual ||
          (args->residual_bf16 != args->output_bf16 &&
           args->residual_bf16 != args->residual_output_bf16));
}

q27_prefill_core_status Embedding(const q27_prefill_embedding_args* args,
                                  uint32_t tile_tokens) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_CORE_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > tile_tokens ||
      args->token_ids_u32 == nullptr || args->embedding_bf16 == nullptr ||
      args->output_bf16 == nullptr ||
      args->invalid_token_count_u32 == nullptr) {
    return Invalid("invalid Q27 batched embedding arguments");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cudaError_t error = cudaMemsetAsync(args->invalid_token_count_u32, 0,
                                      sizeof(*args->invalid_token_count_u32),
                                      stream);
  if (error != cudaSuccess) {
    return CudaError("clear Q27 invalid-token counter", error);
  }
  GatherEmbedding<<<tile_tokens, kThreads, 0, stream>>>(
      args->token_ids_u32,
      static_cast<const __nv_bfloat16*>(args->embedding_bf16),
      static_cast<__nv_bfloat16*>(args->output_bf16), args->valid_tokens,
      args->invalid_token_count_u32);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("Q27 batched embedding", error);
}

q27_prefill_core_status Norm(const q27_prefill_norm_args* args,
                             uint32_t tile_tokens) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_CORE_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > tile_tokens ||
      args->has_residual > 1 || args->input_bf16 == nullptr ||
      args->checkpoint_weight_bf16 == nullptr || args->output_bf16 == nullptr ||
      args->residual_output_bf16 == nullptr ||
      (args->has_residual && args->residual_bf16 == nullptr) ||
      !DistinctNormBuffers(args) || !std::isfinite(args->epsilon) ||
      args->epsilon <= 0.0F) {
    return Invalid("invalid Q27 batched Gemma RMSNorm arguments");
  }
  GemmaNormRows<<<tile_tokens, kThreads, 0,
                  static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input_bf16),
      static_cast<const __nv_bfloat16*>(args->residual_bf16),
      static_cast<const __nv_bfloat16*>(args->checkpoint_weight_bf16),
      static_cast<__nv_bfloat16*>(args->output_bf16),
      static_cast<__nv_bfloat16*>(args->residual_output_bf16),
      args->valid_tokens, args->has_residual != 0, args->epsilon);
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("Q27 batched Gemma RMSNorm", error);
}

}  // namespace

extern "C" q27_prefill_core_status q27_prefill_embedding(
    const q27_prefill_embedding_args* args) {
  return Embedding(args, Q27_PREFILL_CORE_TOKENS);
}

extern "C" q27_prefill_core_status q27_prefill_norm(
    const q27_prefill_norm_args* args) {
  return Norm(args, Q27_PREFILL_CORE_TOKENS);
}

extern "C" q27_prefill_core_status q27_prefill_embedding_m512(
    const q27_prefill_embedding_args* args) {
  return Embedding(args, Q27_PREFILL_CORE_M512_TOKENS);
}

extern "C" q27_prefill_core_status q27_prefill_norm_m512(
    const q27_prefill_norm_args* args) {
  return Norm(args, Q27_PREFILL_CORE_M512_TOKENS);
}

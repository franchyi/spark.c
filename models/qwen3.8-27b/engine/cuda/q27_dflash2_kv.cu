// SPDX-License-Identifier: Apache-2.0
// Chunked batch-one translation of pinned DFlash2 context-KV materialization.

#include "q27_dflash2_kv.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr uint32_t kHidden = Q27_DFLASH2_HIDDEN_SIZE;
constexpr uint32_t kKvHeads = Q27_DFLASH2_KV_HEADS;
constexpr uint32_t kHeadDim = Q27_DFLASH2_HEAD_DIM;
constexpr uint32_t kKvColumns = kKvHeads * kHeadDim;
constexpr uint32_t kRotaryPairs = kHeadDim / 2;
constexpr uint32_t kNormThreads = kHeadDim;
constexpr uint32_t kNormWarps = kNormThreads / 32;
constexpr uint32_t kCopyThreads = 256;
constexpr float kDefaultEpsilon = 1.0e-6F;

constexpr uint64_t kKvWeightBytes =
    static_cast<uint64_t>(kKvColumns) * kHidden * 2ULL;
constexpr uint64_t kNormBytes = kHeadDim * 2ULL;

thread_local std::string g_error;

q27_dflash2_status Ok() { return {Q27_DFLASH2_OK, "ok"}; }

q27_dflash2_status Invalid(const char* message) {
  return {Q27_DFLASH2_INVALID_ARGUMENT, message};
}

q27_dflash2_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_DFLASH2_CUDA_ERROR, g_error.c_str()};
}

q27_dflash2_status CublasError(const char* operation, cublasStatus_t status) {
  g_error.assign(operation);
  g_error.append(": cuBLAS status ");
  g_error.append(std::to_string(static_cast<int>(status)));
  return {Q27_DFLASH2_CUDA_ERROR, g_error.c_str()};
}

template <typename Args>
q27_dflash2_status ValidateHeader(const Args* args) {
  if (args == nullptr) return Invalid("DFlash2 KV arguments are null");
  if (args->struct_size < sizeof(Args)) {
    return Invalid("DFlash2 KV struct_size is too small");
  }
  if (args->abi_version != Q27_DFLASH2_KV_ABI_VERSION) {
    return Invalid("DFlash2 KV ABI version mismatch");
  }
  return Ok();
}

struct BufferRange {
  const void* data;
  uint64_t bytes;
};

bool InvalidRange(BufferRange range, uintptr_t alignment = 2) {
  if (range.data == nullptr ||
      (reinterpret_cast<uintptr_t>(range.data) & (alignment - 1)) != 0) {
    return true;
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(range.data);
  return range.bytes > std::numeric_limits<uintptr_t>::max() - begin;
}

bool Overlap(BufferRange left, BufferRange right) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left.data);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right.data);
  return left_begin < right_begin + static_cast<uintptr_t>(right.bytes) &&
         right_begin < left_begin + static_cast<uintptr_t>(left.bytes);
}

template <size_t N>
bool Disjoint(const std::array<BufferRange, N>& ranges) {
  for (size_t left = 0; left < N; ++left) {
    if (InvalidRange(ranges[left])) return false;
    for (size_t right = left + 1; right < N; ++right) {
      if (InvalidRange(ranges[right]) || Overlap(ranges[left], ranges[right])) {
        return false;
      }
    }
  }
  return true;
}

bool ValidState(const q27_dflash2_state_view* state) {
  return state != nullptr && state->struct_size >= sizeof(*state) &&
         state->abi_version == Q27_DFLASH2_ABI_VERSION &&
         state->key_cache_bf16 != nullptr && state->value_cache_bf16 != nullptr &&
         state->position_tags_u64 != nullptr &&
         (reinterpret_cast<uintptr_t>(state->position_tags_u64) & 7U) == 0;
}

bool ValidLayerWeights(const q27_dflash2_layer_weights& weights) {
  return weights.k_proj.data != nullptr &&
         weights.k_proj.bytes == kKvWeightBytes &&
         weights.v_proj.data != nullptr &&
         weights.v_proj.bytes == kKvWeightBytes &&
         weights.k_norm.data != nullptr && weights.k_norm.bytes == kNormBytes;
}

__device__ float WarpSum(float value) {
#pragma unroll
  for (uint32_t offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

__global__ void BuildRopeCache(uint64_t first_position, uint32_t token_count,
                               const float* inverse_frequencies,
                               float* rope_cache) {
  const uint32_t elements = token_count * kRotaryPairs;
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kRotaryPairs;
    const uint32_t pair = index - token * kRotaryPairs;
    const float angle = static_cast<float>(first_position + token) *
                        inverse_frequencies[pair];
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    rope_cache[static_cast<uint64_t>(index) * 2] = cosine;
    rope_cache[static_cast<uint64_t>(index) * 2 + 1] = sine;
  }
}

/*
 * Keep the projected K/V rows in scratch only until this kernel.  The old
 * path normalized K in place and then launched a second, bandwidth-only
 * kernel to read K and V again and publish them to the ring.  A token/head
 * block already owns every element needed for both operations, so publish the
 * normalized/rotated K and unmodified V directly.  This preserves the exact
 * FP32 reduction and BF16 rounding points while removing one launch and one
 * full K scratch write/read round trip per draft layer.
 */
__global__ void KNormNeoXRopeWriteRing(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* gamma, const float* rope_cache, float epsilon,
    uint64_t first_position, uint32_t layer, __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache, uint64_t* position_tags) {
  const uint32_t token = blockIdx.x / kKvHeads;
  const uint32_t head = blockIdx.x - token * kKvHeads;
  const uint32_t dimension = threadIdx.x;
  const uint64_t scratch_base =
      (static_cast<uint64_t>(token) * kKvHeads + head) * kHeadDim;
  const float value = __bfloat162float(k[scratch_base + dimension]);
  float square_sum = WarpSum(value * value);
  __shared__ float warp_sums[kNormWarps];
  const uint32_t lane = dimension & 31U;
  const uint32_t warp = dimension >> 5U;
  if (lane == 0) warp_sums[warp] = square_sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kNormWarps ? warp_sums[lane] : 0.0F;
    total = WarpSum(total);
    if (lane == 0) warp_sums[0] = total;
  }
  __syncthreads();
  const float inverse_rms =
      rsqrtf(warp_sums[0] / static_cast<float>(kHeadDim) + epsilon);

  __shared__ __nv_bfloat16 normalized[kHeadDim];
  normalized[dimension] = __float2bfloat16_rn(
      value * inverse_rms * __bfloat162float(gamma[dimension]));
  __syncthreads();

  const uint64_t position = first_position + token;
  const uint32_t slot =
      static_cast<uint32_t>(position) & (Q27_DFLASH2_SLIDING_WINDOW - 1);
  const uint64_t cache_base =
      (static_cast<uint64_t>(layer) * Q27_DFLASH2_SLIDING_WINDOW + slot) *
          kKvColumns +
      static_cast<uint64_t>(head) * kHeadDim;
  value_cache[cache_base + dimension] = v[scratch_base + dimension];
  if (dimension < kRotaryPairs) {
    const float first = __bfloat162float(normalized[dimension]);
    const float second =
        __bfloat162float(normalized[dimension + kRotaryPairs]);
    const uint64_t rope =
        (static_cast<uint64_t>(token) * kRotaryPairs + dimension) * 2;
    const float cosine = rope_cache[rope];
    const float sine = rope_cache[rope + 1];
    key_cache[cache_base + dimension] =
        __float2bfloat16_rn(first * cosine - second * sine);
    key_cache[cache_base + dimension + kRotaryPairs] =
        __float2bfloat16_rn(second * cosine + first * sine);
  }
  if (layer == 0 && head == 0 && dimension == 0) {
    position_tags[slot] = position;
  }
}

q27_dflash2_status ProjectRows(cublasHandle_t handle, const void* input,
                               const void* weight, void* output, uint32_t rows,
                               uint32_t input_columns,
                               uint32_t output_columns,
                               const char* operation) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns, rows, input_columns,
      &alpha, weight, CUDA_R_16BF, input_columns, input, CUDA_R_16BF,
      input_columns, &beta, output, CUDA_R_16BF, output_columns,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? Ok() : CublasError(operation, status);
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_reset_kv(
    const q27_dflash2_kv_reset_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (!ValidState(args->state)) {
    return Invalid("invalid DFlash2 KV reset state");
  }
  const cudaError_t error = cudaMemsetAsync(
      args->state->position_tags_u64, 0xFF, Q27_DFLASH2_POSITION_TAG_BYTES,
      static_cast<cudaStream_t>(args->cuda_stream));
  if (error != cudaSuccess) return CudaError("DFlash2 KV tag reset", error);
  args->state->committed_length = 0;
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_materialize_context_kv(
    const q27_dflash2_kv_materialize_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->weights == nullptr || !ValidState(args->state) ||
      args->token_count == 0 ||
      args->token_count > Q27_DFLASH2_KV_MAX_CHUNK_TOKENS ||
      args->first_position > Q27_DFLASH2_MAX_POSITION - args->token_count ||
      args->cublas_handle == nullptr || !std::isfinite(args->rms_epsilon) ||
      args->rms_epsilon < 0.0F) {
    return Invalid("invalid DFlash2 context-KV metadata");
  }
  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    if (!ValidLayerWeights(args->weights->layers[layer])) {
      return Invalid("invalid DFlash2 context-KV layer weights");
    }
  }

  const uint64_t context_bytes =
      static_cast<uint64_t>(args->token_count) * kHidden * 2ULL;
  const uint64_t scratch_bytes =
      static_cast<uint64_t>(args->token_count) *
      Q27_DFLASH2_KV_SCRATCH_BYTES_PER_TOKEN;
  const uint64_t rope_bytes =
      static_cast<uint64_t>(args->token_count) *
      Q27_DFLASH2_KV_ROPE_CACHE_BYTES_PER_TOKEN;
  const std::array<BufferRange, 4> runtime_buffers{{
      {args->context_hidden_bf16, context_bytes},
      {args->k_scratch_bf16, scratch_bytes},
      {args->v_scratch_bf16, scratch_bytes},
      {args->rope_cache_f32, rope_bytes},
  }};
  if (!Disjoint(runtime_buffers) ||
      InvalidRange({args->rope_inverse_frequencies_f32,
                    Q27_DFLASH2_ATTENTION_ROPE_FREQUENCY_BYTES},
                   4) ||
      (reinterpret_cast<uintptr_t>(args->rope_cache_f32) & 3U) != 0) {
    return Invalid("DFlash2 context-KV scratch is null, misaligned, or overlaps");
  }
  const BufferRange frequencies{
      args->rope_inverse_frequencies_f32,
      Q27_DFLASH2_ATTENTION_ROPE_FREQUENCY_BYTES};
  for (BufferRange runtime : runtime_buffers) {
    if (Overlap(runtime, frequencies)) {
      return Invalid("DFlash2 context-KV scratch overlaps RoPE frequencies");
    }
  }
  const std::array<BufferRange, 3> state_buffers{{
      {args->state->key_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {args->state->value_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {args->state->position_tags_u64, Q27_DFLASH2_POSITION_TAG_BYTES},
  }};
  if (!Disjoint(state_buffers)) {
    return Invalid("DFlash2 context-KV state buffers overlap");
  }
  std::array<BufferRange, Q27_DFLASH2_LAYERS * 3> weight_buffers{};
  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    const q27_dflash2_layer_weights& weights = args->weights->layers[layer];
    weight_buffers[layer * 3] = {weights.k_proj.data, kKvWeightBytes};
    weight_buffers[layer * 3 + 1] = {weights.v_proj.data, kKvWeightBytes};
    weight_buffers[layer * 3 + 2] = {weights.k_norm.data, kNormBytes};
  }
  if (!Disjoint(weight_buffers)) {
    return Invalid("DFlash2 context-KV checkpoint weights overlap");
  }
  for (BufferRange runtime : runtime_buffers) {
    for (BufferRange state : state_buffers) {
      if (Overlap(runtime, state)) {
        return Invalid("DFlash2 context-KV scratch overlaps persistent state");
      }
    }
    for (BufferRange weight : weight_buffers) {
      if (Overlap(runtime, weight)) {
        return Invalid("DFlash2 context-KV scratch overlaps checkpoint weights");
      }
    }
  }
  for (BufferRange state : state_buffers) {
    if (Overlap(state, frequencies)) {
      return Invalid("DFlash2 context-KV frequencies overlap persistent state");
    }
    for (BufferRange weight : weight_buffers) {
      if (Overlap(state, weight)) {
        return Invalid("DFlash2 context-KV state overlaps checkpoint weights");
      }
    }
  }
  for (BufferRange weight : weight_buffers) {
    if (Overlap(weight, frequencies)) {
      return Invalid("DFlash2 context-KV frequencies overlap checkpoint weights");
    }
  }

  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("DFlash2 context-KV set stream", cublas_status);
  }
  const uint32_t rope_elements = args->token_count * kRotaryPairs;
  const uint32_t rope_blocks = (rope_elements + kCopyThreads - 1) / kCopyThreads;
  BuildRopeCache<<<rope_blocks, kCopyThreads, 0, stream>>>(
      args->first_position, args->token_count,
      args->rope_inverse_frequencies_f32, args->rope_cache_f32);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("DFlash2 context-KV RoPE cache launch", error);
  }

  const float epsilon =
      args->rms_epsilon > 0.0F ? args->rms_epsilon : kDefaultEpsilon;
  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    const q27_dflash2_layer_weights& weights = args->weights->layers[layer];
    status = ProjectRows(handle, args->context_hidden_bf16,
                         weights.k_proj.data, args->k_scratch_bf16,
                         args->token_count, kHidden, kKvColumns,
                         "DFlash2 context K projection");
    if (status.code != Q27_DFLASH2_OK) return status;
    status = ProjectRows(handle, args->context_hidden_bf16,
                         weights.v_proj.data, args->v_scratch_bf16,
                         args->token_count, kHidden, kKvColumns,
                         "DFlash2 context V projection");
    if (status.code != Q27_DFLASH2_OK) return status;
    KNormNeoXRopeWriteRing<<<args->token_count * kKvHeads, kNormThreads, 0,
                            stream>>>(
        static_cast<const __nv_bfloat16*>(args->k_scratch_bf16),
        static_cast<const __nv_bfloat16*>(args->v_scratch_bf16),
        static_cast<const __nv_bfloat16*>(weights.k_norm.data),
        args->rope_cache_f32, epsilon, args->first_position, layer,
        static_cast<__nv_bfloat16*>(args->state->key_cache_bf16),
        static_cast<__nv_bfloat16*>(args->state->value_cache_bf16),
        args->state->position_tags_u64);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaError("DFlash2 fused context K norm/RoPE/KV write launch",
                       error);
    }
  }
  return Ok();
}

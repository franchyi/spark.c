// SPDX-License-Identifier: Apache-2.0
// Revision-pinned FlashInfer BF16 DFlash2 sliding-attention specialization.

#include "q27_dflash2_flashinfer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

namespace {

constexpr uint32_t kTokens = Q27_DFLASH2_BLOCK_SIZE;
constexpr uint32_t kQHeads = Q27_DFLASH2_QUERY_HEADS;
constexpr uint32_t kKvHeads = Q27_DFLASH2_KV_HEADS;
constexpr uint32_t kHeadDim = Q27_DFLASH2_HEAD_DIM;
constexpr uint32_t kQColumns = kQHeads * kHeadDim;
constexpr uint32_t kKvColumns = kKvHeads * kHeadDim;
constexpr uint32_t kWindowLeft = Q27_DFLASH2_SLIDING_WINDOW - 1;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxGatherBlocks = 512;
constexpr float kScale = 0.08838834764831845F;

constexpr uint64_t kQBytes = kTokens * kQColumns * 2ULL;
constexpr uint64_t kKvBytes = kTokens * kKvColumns * 2ULL;
constexpr uint64_t kContextBytes = kQBytes;

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

q27_dflash2_status ExceptionError(const char* operation,
                                  const std::exception& error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(error.what());
  return {Q27_DFLASH2_CUDA_ERROR, g_error.c_str()};
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

__global__ void GatherTaggedRingAndBlock(
    const __nv_bfloat16* live_k, const __nv_bfloat16* live_v,
    const uint64_t* position_tags, const __nv_bfloat16* block_k,
    const __nv_bfloat16* block_v, const uint64_t* positions,
    uint64_t committed_length, uint32_t history_tokens, uint32_t kv_tokens,
    __nv_bfloat16* staging_k, __nv_bfloat16* staging_v,
    uint32_t* invalid_count) {
  const uint32_t elements = kv_tokens * kKvColumns;
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kKvColumns;
    const uint32_t within_token = index - token * kKvColumns;
    if (token < history_tokens) {
      const uint64_t expected_position =
          committed_length - history_tokens + token;
      const uint32_t slot = static_cast<uint32_t>(expected_position) &
                            (Q27_DFLASH2_SLIDING_WINDOW - 1);
      const bool valid = position_tags[slot] == expected_position;
      if (!valid && within_token == 0) atomicAdd(invalid_count, 1U);
      const uint64_t source = static_cast<uint64_t>(slot) * kKvColumns +
                              within_token;
      staging_k[index] =
          valid ? live_k[source] : __float2bfloat16_rn(0.0F);
      staging_v[index] =
          valid ? live_v[source] : __float2bfloat16_rn(0.0F);
    } else {
      const uint32_t block_token = token - history_tokens;
      const uint64_t source =
          static_cast<uint64_t>(block_token) * kKvColumns + within_token;
      staging_k[index] = block_k[source];
      staging_v[index] = block_v[source];
    }
  }
  if (blockIdx.x == 0 && threadIdx.x < kTokens) {
    const uint32_t token = threadIdx.x;
    if (positions[token] != committed_length + token) {
      atomicAdd(invalid_count, 1U);
    }
  }
}

bool ValidateCall(const q27_dflash2_sliding_attention_call* call) {
  if (call == nullptr || call->struct_size < sizeof(*call) ||
      call->abi_version != Q27_DFLASH2_ATTENTION_ABI_VERSION ||
      call->layer_index >= Q27_DFLASH2_LAYERS || call->token_count != kTokens ||
      call->positions_u64 == nullptr || call->q_bf16 == nullptr ||
      call->k_bf16 == nullptr || call->v_bf16 == nullptr ||
      call->context_bf16 == nullptr || call->state == nullptr ||
      call->state->struct_size < sizeof(*call->state) ||
      call->state->abi_version != Q27_DFLASH2_ABI_VERSION ||
      call->state->key_cache_bf16 == nullptr ||
      call->state->value_cache_bf16 == nullptr ||
      call->state->position_tags_u64 == nullptr || call->workspace == nullptr ||
      call->workspace_bytes < Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES ||
      call->state->committed_length >
          Q27_DFLASH2_MAX_POSITION - Q27_DFLASH2_BLOCK_SIZE ||
      call->window_left != kWindowLeft || !std::isfinite(call->scale) ||
      std::abs(call->scale - kScale) > 1.0e-8F) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(call->positions_u64) & 7U) != 0 ||
      (reinterpret_cast<uintptr_t>(call->state->position_tags_u64) & 7U) != 0 ||
      (reinterpret_cast<uintptr_t>(call->workspace) & 15U) != 0) {
    return false;
  }
  const BufferRange buffers[] = {
      {call->positions_u64, kTokens * sizeof(uint64_t)},
      {call->q_bf16, kQBytes},
      {call->k_bf16, kKvBytes},
      {call->v_bf16, kKvBytes},
      {call->context_bf16, kContextBytes},
      {call->state->key_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {call->state->value_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {call->state->position_tags_u64, Q27_DFLASH2_POSITION_TAG_BYTES},
      {call->workspace, Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES},
  };
  for (uint32_t left = 0; left < sizeof(buffers) / sizeof(buffers[0]); ++left) {
    if (InvalidRange(buffers[left])) return false;
    for (uint32_t right = left + 1;
         right < sizeof(buffers) / sizeof(buffers[0]); ++right) {
      if (InvalidRange(buffers[right]) || Overlap(buffers[left], buffers[right])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

extern "C" uint32_t* q27_dflash2_flashinfer_invalid_count(
    void* workspace, uint64_t workspace_bytes) {
  if (workspace == nullptr ||
      workspace_bytes < Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES ||
      (reinterpret_cast<uintptr_t>(workspace) & 15U) != 0) {
    return nullptr;
  }
  return reinterpret_cast<uint32_t*>(
      static_cast<uint8_t*>(workspace) +
      Q27_DFLASH2_FLASHINFER_INVALID_COUNT_OFFSET);
}

extern "C" q27_dflash2_status q27_dflash2_flashinfer_sliding_attention(
    const q27_dflash2_sliding_attention_call* call, void* user_data) {
  if (user_data != nullptr) {
    return Invalid("DFlash2 FlashInfer hook user_data must be null");
  }
  if (!ValidateCall(call)) {
    return Invalid("invalid DFlash2 FlashInfer sliding-attention call");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(call->cuda_stream);
  uint8_t* workspace = static_cast<uint8_t*>(call->workspace);
  auto* staging_k = reinterpret_cast<__nv_bfloat16*>(workspace);
  auto* staging_v = reinterpret_cast<__nv_bfloat16*>(
      workspace + Q27_DFLASH2_FLASHINFER_ONE_STAGING_BYTES);
  uint32_t* invalid_count = q27_dflash2_flashinfer_invalid_count(
      call->workspace, call->workspace_bytes);
  cudaError_t error =
      cudaMemsetAsync(invalid_count, 0, sizeof(*invalid_count), stream);
  if (error != cudaSuccess) {
    return CudaError("DFlash2 FlashInfer invariant clear", error);
  }

  const uint64_t committed_length = call->state->committed_length;
  const uint32_t history_tokens = static_cast<uint32_t>(
      committed_length < Q27_DFLASH2_FLASHINFER_HISTORY_TOKENS
          ? committed_length
          : Q27_DFLASH2_FLASHINFER_HISTORY_TOKENS);
  const uint32_t kv_tokens = history_tokens + kTokens;
  const uint32_t gather_elements = kv_tokens * kKvColumns;
  const uint32_t gather_blocks =
      (gather_elements + kThreads - 1) / kThreads < kMaxGatherBlocks
          ? (gather_elements + kThreads - 1) / kThreads
          : kMaxGatherBlocks;
  const uint64_t layer_elements =
      static_cast<uint64_t>(call->layer_index) *
      Q27_DFLASH2_SLIDING_WINDOW * kKvColumns;
  const auto* live_k = static_cast<const __nv_bfloat16*>(
                           call->state->key_cache_bf16) +
                       layer_elements;
  const auto* live_v = static_cast<const __nv_bfloat16*>(
                           call->state->value_cache_bf16) +
                       layer_elements;
  GatherTaggedRingAndBlock<<<gather_blocks, kThreads, 0, stream>>>(
      live_k, live_v, call->state->position_tags_u64,
      static_cast<const __nv_bfloat16*>(call->k_bf16),
      static_cast<const __nv_bfloat16*>(call->v_bf16), call->positions_u64,
      committed_length, history_tokens, kv_tokens, staging_k, staging_v,
      invalid_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("DFlash2 FlashInfer ring gather", error);
  }

  using DType = nv_bfloat16;
  using Params = flashinfer::SinglePrefillParams<DType, DType, DType>;
  using Variant = flashinfer::DefaultAttention<
      /*use_custom_mask=*/false, /*use_sliding_window=*/true,
      /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
  Params params(
      reinterpret_cast<DType*>(const_cast<void*>(call->q_bf16)),
      reinterpret_cast<DType*>(staging_k), reinterpret_cast<DType*>(staging_v),
      /*maybe_custom_mask=*/nullptr,
      reinterpret_cast<DType*>(call->context_bf16), /*lse=*/nullptr,
      /*maybe_alibi_slopes=*/nullptr, kQHeads, kKvHeads, kTokens, kv_tokens,
      kQColumns, kHeadDim, kKvColumns, kHeadDim, kHeadDim,
      static_cast<int32_t>(call->window_left), /*logits_soft_cap=*/0.0F,
      call->scale, /*rope_scale=*/1.0F, /*rope_theta=*/10000000.0F);
  try {
    error = flashinfer::SinglePrefillWithKVCacheDispatched<
        kHeadDim, kHeadDim, flashinfer::PosEncodingMode::kNone,
        /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kCausal,
        Variant>(params, /*tmp=*/nullptr, stream);
  } catch (const std::exception& exception) {
    return ExceptionError("DFlash2 FlashInfer dispatch", exception);
  }
  return error == cudaSuccess
             ? Ok()
             : CudaError("DFlash2 FlashInfer sliding attention", error);
}

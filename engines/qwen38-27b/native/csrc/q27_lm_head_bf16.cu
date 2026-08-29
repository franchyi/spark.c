/*
 * Fixed Qwen3.8-27B BF16 LM-head streaming GEMV.
 *
 * The one-output-row-per-warp organization is adapted from ds4.c's
 * glm53_matvec_bf16_f32_kernel (ds4 commit
 * a60a2a0d25137a849a101e04e86ea830a346073a, ds4_cuda.cu:26660-26683),
 * used under its MIT license. This specialization changes the activation from
 * FP32 to BF16, uses paired BF16 loads, and fixes the only supported shape.
 *
 * Copyright (c) 2026 The ds4.c authors
 * Copyright (c) 2023-2026 The ggml authors
 * SPDX-License-Identifier: MIT
 */

#include "q27_lm_head_bf16.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kVocabulary = 248320;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kRowsPerBlock = 8;
constexpr uint32_t kThreads = kRowsPerBlock * 32;
thread_local std::string g_error;

q27_kernel_status Ok() { return {Q27_KERNEL_OK, "ok"}; }

q27_kernel_status Invalid(const char* message) {
  return {Q27_KERNEL_INVALID_ARGUMENT, message};
}

q27_kernel_status CudaError(cudaError_t error) {
  g_error.assign("q27 streaming BF16 LM head: ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_KERNEL_CUDA_ERROR, g_error.c_str()};
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

__global__ __launch_bounds__(kThreads, 1) void StreamingBf16Gemv(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ hidden,
    float* __restrict__ logits) {
  const uint32_t warp = threadIdx.x >> 5U;
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t row = blockIdx.x * kRowsPerBlock + warp;
  if (row >= kVocabulary) return;

  constexpr uint32_t kPairs = kHidden / 2;
  const auto* weight_pairs = reinterpret_cast<const __nv_bfloat162*>(
      weight + static_cast<uint64_t>(row) * kHidden);
  const auto* hidden_pairs =
      reinterpret_cast<const __nv_bfloat162*>(hidden);
  float sum = 0.0F;
  for (uint32_t pair = lane; pair < kPairs; pair += 32U) {
    const float2 w = __bfloat1622float2(weight_pairs[pair]);
    const float2 x = __bfloat1622float2(hidden_pairs[pair]);
    sum = fmaf(w.x, x.x, sum);
    sum = fmaf(w.y, x.y, sum);
  }
  sum = WarpSum(sum);
  if (lane == 0U) logits[row] = sum;
}

}  // namespace

extern "C" q27_kernel_status q27_lm_head_bf16_stream(
    const q27_lm_head_args* arguments) {
  if (arguments == nullptr || arguments->struct_size != sizeof(*arguments) ||
      arguments->abi_version != Q27_KERNEL_ABI_VERSION ||
      arguments->vocabulary != kVocabulary ||
      arguments->hidden_size != kHidden ||
      arguments->hidden_bf16 == nullptr || arguments->weight_bf16 == nullptr ||
      arguments->logits_f32 == nullptr) {
    return Invalid("invalid q27 streaming BF16 LM-head arguments");
  }
  StreamingBf16Gemv<<<(kVocabulary + kRowsPerBlock - 1) / kRowsPerBlock,
                       kThreads, 0,
                       static_cast<cudaStream_t>(arguments->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(arguments->weight_bf16),
      static_cast<const __nv_bfloat16*>(arguments->hidden_bf16),
      arguments->logits_f32);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError(error);
}

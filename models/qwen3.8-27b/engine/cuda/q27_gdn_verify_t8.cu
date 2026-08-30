// SPDX-License-Identifier: Apache-2.0
// Non-mutating causal-convolution journal and selected-state commit for the
// fixed Qwen3.8-27B DFlash2 T=8 verifier.

#include "q27_gdn_verify_t8.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr uint64_t kProjectedBytes =
    Q27_GDN_VERIFY_TOKENS * Q27_GDN_VERIFY_QKV_WIDTH * 2ULL;
constexpr uint64_t kWeightBytes =
    Q27_GDN_VERIFY_QKV_WIDTH * Q27_GDN_VERIFY_CONV_KERNEL * 2ULL;
constexpr uint64_t kLiveConvBytes =
    Q27_GDN_VERIFY_GDN_LAYERS *
    Q27_GDN_VERIFY_CONV_STATE_BYTES_PER_LAYER;
constexpr uint64_t kLiveRecurrentBytes =
    Q27_GDN_VERIFY_GDN_LAYERS *
    Q27_GDN_VERIFY_RECURRENT_STATE_BYTES_PER_LAYER;
constexpr uint64_t kConvJournalBytes =
    Q27_GDN_VERIFY_GDN_LAYERS *
    Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER;
constexpr uint64_t kRecurrentJournalBytes =
    Q27_GDN_VERIFY_GDN_LAYERS *
    Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER;
constexpr uint32_t kThreads = 256;

static_assert(Q27_GDN_VERIFY_CONV_STATE_BYTES_PER_LAYER == 61440);
static_assert(Q27_GDN_VERIFY_RECURRENT_STATE_BYTES_PER_LAYER == 1572864);
static_assert(Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER == 491520);
static_assert(Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER == 12582912);
static_assert(kConvJournalBytes == 23592960);
static_assert(kRecurrentJournalBytes == 603979776);

thread_local std::string g_error;

q27_gdn_verify_t8_status Ok() { return {Q27_GDN_VERIFY_T8_OK, "ok"}; }

q27_gdn_verify_t8_status Invalid(const char* message) {
  return {Q27_GDN_VERIFY_T8_INVALID_ARGUMENT, message};
}

q27_gdn_verify_t8_status CudaError(const char* operation,
                                   cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_VERIFY_T8_CUDA_ERROR, g_error.c_str()};
}

bool Aligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

uint32_t Blocks(uint64_t elements) {
  constexpr uint32_t kMaxBlocks = 65535;
  const uint64_t required = (elements + kThreads - 1) / kThreads;
  return static_cast<uint32_t>(required < kMaxBlocks ? required : kMaxBlocks);
}

/*
 * One thread owns one convolution channel. This is the exact T=8 restriction
 * of the pinned SGLang causal-convolution arithmetic: each BF16 product is
 * rounded before FP32 accumulation, then SiLU is rounded to BF16. The live
 * history is never written.
 */
__global__ void ConvolveAndJournal(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* live_state, __nv_bfloat16* output,
    __nv_bfloat16* checkpoints) {
  const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= Q27_GDN_VERIFY_QKV_WIDTH) return;

  __nv_bfloat16 h0 =
      live_state[channel * Q27_GDN_VERIFY_CONV_HISTORY + 0];
  __nv_bfloat16 h1 =
      live_state[channel * Q27_GDN_VERIFY_CONV_HISTORY + 1];
  __nv_bfloat16 h2 =
      live_state[channel * Q27_GDN_VERIFY_CONV_HISTORY + 2];
  const __nv_bfloat16* channel_weight =
      weight + channel * Q27_GDN_VERIFY_CONV_KERNEL;

#pragma unroll
  for (uint32_t token = 0; token < Q27_GDN_VERIFY_TOKENS; ++token) {
    const __nv_bfloat16 x =
        input[static_cast<uint64_t>(token) * Q27_GDN_VERIFY_QKV_WIDTH +
              channel];
    const __nv_bfloat16 values[Q27_GDN_VERIFY_CONV_KERNEL] = {h0, h1, h2,
                                                              x};
    float sum = 0.0F;
#pragma unroll
    for (uint32_t tap = 0; tap < Q27_GDN_VERIFY_CONV_KERNEL; ++tap) {
      const __nv_bfloat16 product = __float2bfloat16_rn(
          __bfloat162float(channel_weight[tap]) *
          __bfloat162float(values[tap]));
      sum += __bfloat162float(product);
    }
    output[static_cast<uint64_t>(token) * Q27_GDN_VERIFY_QKV_WIDTH +
           channel] = __float2bfloat16_rn(sum / (1.0F + __expf(-sum)));
    h0 = h1;
    h1 = h2;
    h2 = x;
    const uint64_t checkpoint =
        (static_cast<uint64_t>(token) * Q27_GDN_VERIFY_QKV_WIDTH + channel) *
        Q27_GDN_VERIFY_CONV_HISTORY;
    checkpoints[checkpoint + 0] = h0;
    checkpoints[checkpoint + 1] = h1;
    checkpoints[checkpoint + 2] = h2;
  }
}

__global__ void SelectJournalVectors(const uint4* checkpoints, uint4* live,
                                     uint64_t vectors_per_layer,
                                     const uint32_t* selected_row,
                                     uint32_t* device_error) {
  const uint32_t row = selected_row[0];
  if (row >= Q27_GDN_VERIFY_TOKENS) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicCAS(device_error, 0U, 1U);
    return;
  }
  const uint64_t total =
      static_cast<uint64_t>(Q27_GDN_VERIFY_GDN_LAYERS) * vectors_per_layer;
  for (uint64_t destination =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       destination < total;
       destination += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint64_t layer = destination / vectors_per_layer;
    const uint64_t offset = destination - layer * vectors_per_layer;
    const uint64_t source =
        (layer * Q27_GDN_VERIFY_TOKENS + row) * vectors_per_layer + offset;
    live[destination] = checkpoints[source];
  }
}

}  // namespace

extern "C" q27_gdn_verify_t8_status q27_gdn_verify_t8_convolve(
    const q27_gdn_verify_t8_conv_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_VERIFY_T8_ABI_VERSION ||
      !Aligned(args->projected_qkv_bf16, 2) ||
      !Aligned(args->conv_weight_bf16, 2) ||
      !Aligned(args->live_convolution_state_bf16, 2) ||
      !Aligned(args->convolved_qkv_bf16, 2) ||
      !Aligned(args->checkpoint_convolution_bf16, 2) ||
      args->projected_qkv_bytes < kProjectedBytes ||
      args->conv_weight_bytes < kWeightBytes ||
      args->live_convolution_state_bytes <
          Q27_GDN_VERIFY_CONV_STATE_BYTES_PER_LAYER ||
      args->convolved_qkv_bytes < kProjectedBytes ||
      args->checkpoint_convolution_bytes <
          Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER)
    return Invalid("invalid Q27 T=8 verify convolution arguments");

  ConvolveAndJournal<<<
      (Q27_GDN_VERIFY_QKV_WIDTH + kThreads - 1) / kThreads, kThreads, 0,
      static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->projected_qkv_bf16),
      static_cast<const __nv_bfloat16*>(args->conv_weight_bf16),
      static_cast<const __nv_bfloat16*>(args->live_convolution_state_bf16),
      static_cast<__nv_bfloat16*>(args->convolved_qkv_bf16),
      static_cast<__nv_bfloat16*>(args->checkpoint_convolution_bf16));
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("Q27 T=8 verify convolution", error);
}

extern "C" q27_gdn_verify_t8_status q27_gdn_verify_t8_commit(
    const q27_gdn_verify_t8_commit_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_VERIFY_T8_ABI_VERSION ||
      !Aligned(args->checkpoint_convolution_bf16, 16) ||
      !Aligned(args->checkpoint_recurrent_bf16, 16) ||
      !Aligned(args->live_convolution_state_bf16, 16) ||
      !Aligned(args->live_recurrent_state_bf16, 16) ||
      !Aligned(args->selected_row_u32, alignof(uint32_t)) ||
      !Aligned(args->device_error_u32, alignof(uint32_t)) ||
      args->checkpoint_convolution_bytes < kConvJournalBytes ||
      args->checkpoint_recurrent_bytes < kRecurrentJournalBytes ||
      args->live_convolution_state_bytes < kLiveConvBytes ||
      args->live_recurrent_state_bytes < kLiveRecurrentBytes)
    return Invalid("invalid Q27 T=8 verify commit arguments");

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cudaError_t error =
      cudaMemsetAsync(args->device_error_u32, 0, sizeof(uint32_t), stream);
  if (error != cudaSuccess)
    return CudaError("Q27 T=8 verify commit error reset", error);

  constexpr uint64_t kConvVectorsPerLayer =
      Q27_GDN_VERIFY_CONV_STATE_BYTES_PER_LAYER / sizeof(uint4);
  constexpr uint64_t kRecurrentVectorsPerLayer =
      Q27_GDN_VERIFY_RECURRENT_STATE_BYTES_PER_LAYER / sizeof(uint4);
  SelectJournalVectors<<<
      Blocks(Q27_GDN_VERIFY_GDN_LAYERS * kConvVectorsPerLayer), kThreads, 0,
      stream>>>(
      static_cast<const uint4*>(args->checkpoint_convolution_bf16),
      static_cast<uint4*>(args->live_convolution_state_bf16),
      kConvVectorsPerLayer, args->selected_row_u32, args->device_error_u32);
  SelectJournalVectors<<<
      Blocks(Q27_GDN_VERIFY_GDN_LAYERS * kRecurrentVectorsPerLayer), kThreads,
      0, stream>>>(
      static_cast<const uint4*>(args->checkpoint_recurrent_bf16),
      static_cast<uint4*>(args->live_recurrent_state_bf16),
      kRecurrentVectorsPerLayer, args->selected_row_u32,
      args->device_error_u32);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("Q27 T=8 verify commit", error);
}

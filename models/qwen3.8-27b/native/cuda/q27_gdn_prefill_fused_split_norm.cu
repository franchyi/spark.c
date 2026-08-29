/*
 * Exact fixed-M128 fusion of QKVZ split, BF16 causal convolution, and Q/K
 * L2Norm for Qwen3.8-27B GDN prefill. See the companion provenance note.
 */

#include "q27_gdn_prefill_fused_split_norm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr uint64_t kBf16Bytes = 2;
constexpr int kQWidth = Q27_GDN_FUSED_QK_HEADS * Q27_GDN_FUSED_HEAD_DIM;
constexpr int kKOffset = kQWidth;
constexpr int kVOffset = 2 * kQWidth;
constexpr uint64_t kFusedBytes =
    static_cast<uint64_t>(Q27_GDN_FUSED_TOKENS) *
    Q27_GDN_FUSED_QKVZ_WIDTH * kBf16Bytes;
constexpr uint64_t kWeightBytes =
    static_cast<uint64_t>(Q27_GDN_FUSED_QKV_WIDTH) *
    Q27_GDN_FUSED_CONV_KERNEL * kBf16Bytes;
constexpr uint64_t kStateBytes =
    static_cast<uint64_t>(Q27_GDN_FUSED_QKV_WIDTH) *
    Q27_GDN_FUSED_CONV_HISTORY * kBf16Bytes;
constexpr uint64_t kQkBytes =
    static_cast<uint64_t>(Q27_GDN_FUSED_TOKENS) *
    Q27_GDN_FUSED_QK_HEADS * Q27_GDN_FUSED_HEAD_DIM * kBf16Bytes;
constexpr uint64_t kValueBytes =
    static_cast<uint64_t>(Q27_GDN_FUSED_TOKENS) *
    Q27_GDN_FUSED_VALUE_HEADS * Q27_GDN_FUSED_HEAD_DIM * kBf16Bytes;

static_assert(kFusedBytes == 4194304);
static_assert(kWeightBytes == 81920);
static_assert(kStateBytes == 61440);
static_assert(kQkBytes == 524288);
static_assert(kValueBytes == 1572864);

thread_local std::string g_error;

q27_gdn_fused_split_norm_status Ok() {
  return {Q27_GDN_FUSED_SPLIT_NORM_OK, "ok"};
}

q27_gdn_fused_split_norm_status Invalid(const char* message) {
  return {Q27_GDN_FUSED_SPLIT_NORM_INVALID_ARGUMENT, message};
}

q27_gdn_fused_split_norm_status CudaError(const char* prefix,
                                           cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_FUSED_SPLIT_NORM_CUDA_ERROR, g_error.c_str()};
}

bool AlignedBf16(const void* pointer) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (kBf16Bytes - 1)) == 0;
}

/*
 * One block owns one Q or K head. Each lane owns exactly one convolution
 * channel, preserving the donor's per-channel recurrence. The block then
 * applies the legacy 128-lane, stride-halving L2 reduction to the rounded
 * BF16 convolution result before advancing to the next token. The pinned
 * c427 Triton L2 reduction may differ from this fallback by 1--2 BF16 ULP.
 */
__global__ void ConvNormalizeQK128(const __nv_bfloat16* fused,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* state,
                                   __nv_bfloat16* q_normalized,
                                   __nv_bfloat16* k_normalized,
                                   uint32_t valid_tokens,
                                   uint32_t source_row) {
  const int q_or_k = static_cast<int>(blockIdx.x) / Q27_GDN_FUSED_QK_HEADS;
  const int head = static_cast<int>(blockIdx.x) % Q27_GDN_FUSED_QK_HEADS;
  const int dimension = static_cast<int>(threadIdx.x);
  const int channel = q_or_k * kQWidth +
                      head * Q27_GDN_FUSED_HEAD_DIM + dimension;

  __nv_bfloat16 h0 =
      state[channel * Q27_GDN_FUSED_CONV_HISTORY + 0];
  __nv_bfloat16 h1 =
      state[channel * Q27_GDN_FUSED_CONV_HISTORY + 1];
  __nv_bfloat16 h2 =
      state[channel * Q27_GDN_FUSED_CONV_HISTORY + 2];
  const __nv_bfloat16* channel_weight =
      weight + channel * Q27_GDN_FUSED_CONV_KERNEL;
  __nv_bfloat16* output =
      q_or_k == 0 ? q_normalized : k_normalized;
  __shared__ float reduction[Q27_GDN_FUSED_HEAD_DIM];

#pragma unroll 1
  for (int token = 0; token < Q27_GDN_FUSED_TOKENS; ++token) {
    const uint64_t output_index =
        (static_cast<uint64_t>(token) * Q27_GDN_FUSED_QK_HEADS + head) *
            Q27_GDN_FUSED_HEAD_DIM +
        dimension;
    if (static_cast<uint32_t>(token) >= valid_tokens) {
      output[output_index] = __float2bfloat16_rn(0.0F);
      continue;
    }

    const __nv_bfloat16 x =
        fused[(static_cast<uint64_t>(source_row) + token) *
                  Q27_GDN_FUSED_QKVZ_WIDTH +
              channel];
    const __nv_bfloat16 values[Q27_GDN_FUSED_CONV_KERNEL] = {h0, h1, h2,
                                                             x};
    float sum = 0.0F;
#pragma unroll
    for (int tap = 0; tap < Q27_GDN_FUSED_CONV_KERNEL; ++tap) {
      const __nv_bfloat16 product = __float2bfloat16_rn(
          __bfloat162float(channel_weight[tap]) *
          __bfloat162float(values[tap]));
      sum += __bfloat162float(product);
    }
    const __nv_bfloat16 convolved =
        __float2bfloat16_rn(sum / (1.0F + __expf(-sum)));
    const float value = __bfloat162float(convolved);
    reduction[dimension] = value * value;
    __syncthreads();
#pragma unroll
    for (int stride = Q27_GDN_FUSED_HEAD_DIM / 2; stride != 0;
         stride /= 2) {
      if (dimension < stride)
        reduction[dimension] += reduction[dimension + stride];
      __syncthreads();
    }
    output[output_index] =
        __float2bfloat16_rn(value * rsqrtf(reduction[0] + 1.0e-6F));
    __syncthreads();
    h0 = h1;
    h1 = h2;
    h2 = x;
  }

  state[channel * Q27_GDN_FUSED_CONV_HISTORY + 0] = h0;
  state[channel * Q27_GDN_FUSED_CONV_HISTORY + 1] = h1;
  state[channel * Q27_GDN_FUSED_CONV_HISTORY + 2] = h2;
}

/*
 * The value channels have no cross-channel normalization. One CUDA thread
 * therefore retains the donor's entire per-channel causal history while also
 * splitting the matching raw Z feature. Z is copied for every physical row,
 * exactly like the current SplitQkvz boundary; only convolved V is masked.
 */
__global__ void ConvValueSplitZ128(const __nv_bfloat16* fused,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* state,
                                   __nv_bfloat16* value,
                                   __nv_bfloat16* projected_z,
                                   uint32_t valid_tokens,
                                   uint32_t source_row) {
  const int value_channel =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (value_channel >= Q27_GDN_FUSED_Z_WIDTH) return;
  const int channel = kVOffset + value_channel;
  __nv_bfloat16 h0 =
      state[channel * Q27_GDN_FUSED_CONV_HISTORY + 0];
  __nv_bfloat16 h1 =
      state[channel * Q27_GDN_FUSED_CONV_HISTORY + 1];
  __nv_bfloat16 h2 =
      state[channel * Q27_GDN_FUSED_CONV_HISTORY + 2];
  const __nv_bfloat16* channel_weight =
      weight + channel * Q27_GDN_FUSED_CONV_KERNEL;

#pragma unroll 1
  for (int token = 0; token < Q27_GDN_FUSED_TOKENS; ++token) {
    const uint64_t fused_row =
        (static_cast<uint64_t>(source_row) + token) *
        Q27_GDN_FUSED_QKVZ_WIDTH;
    const uint64_t value_index =
        static_cast<uint64_t>(token) * Q27_GDN_FUSED_Z_WIDTH + value_channel;
    projected_z[value_index] =
        fused[fused_row + Q27_GDN_FUSED_QKV_WIDTH + value_channel];
    if (static_cast<uint32_t>(token) >= valid_tokens) {
      value[value_index] = __float2bfloat16_rn(0.0F);
      continue;
    }
    const __nv_bfloat16 x = fused[fused_row + channel];
    const __nv_bfloat16 values[Q27_GDN_FUSED_CONV_KERNEL] = {h0, h1, h2,
                                                             x};
    float sum = 0.0F;
#pragma unroll
    for (int tap = 0; tap < Q27_GDN_FUSED_CONV_KERNEL; ++tap) {
      const __nv_bfloat16 product = __float2bfloat16_rn(
          __bfloat162float(channel_weight[tap]) *
          __bfloat162float(values[tap]));
      sum += __bfloat162float(product);
    }
    value[value_index] =
        __float2bfloat16_rn(sum / (1.0F + __expf(-sum)));
    h0 = h1;
    h1 = h2;
    h2 = x;
  }

  state[channel * Q27_GDN_FUSED_CONV_HISTORY + 0] = h0;
  state[channel * Q27_GDN_FUSED_CONV_HISTORY + 1] = h1;
  state[channel * Q27_GDN_FUSED_CONV_HISTORY + 2] = h2;
}

}  // namespace

extern "C" q27_gdn_fused_split_norm_status q27_gdn_fused_split_norm(
    const q27_gdn_fused_split_norm_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_FUSED_SPLIT_NORM_ABI_VERSION ||
      args->valid_tokens == 0 ||
      args->valid_tokens > Q27_GDN_FUSED_TOKENS ||
      !AlignedBf16(args->fused_qkvz_bf16) ||
      !AlignedBf16(args->conv_weight_bf16) ||
      !AlignedBf16(args->convolution_state_bf16) ||
      !AlignedBf16(args->q_normalized_bf16) ||
      !AlignedBf16(args->k_normalized_bf16) ||
      !AlignedBf16(args->value_bf16) ||
      !AlignedBf16(args->projected_z_bf16) ||
      args->fused_qkvz_bytes <
          (static_cast<uint64_t>(args->source_row) + Q27_GDN_FUSED_TOKENS) *
              Q27_GDN_FUSED_QKVZ_WIDTH * kBf16Bytes ||
      args->conv_weight_bytes < kWeightBytes ||
      args->convolution_state_bytes < kStateBytes ||
      args->q_normalized_bytes < kQkBytes ||
      args->k_normalized_bytes < kQkBytes ||
      args->value_bytes < kValueBytes ||
      args->projected_z_bytes < kValueBytes)
    return Invalid("invalid q27 fused GDN split/norm arguments");

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  ConvNormalizeQK128<<<2 * Q27_GDN_FUSED_QK_HEADS,
                       Q27_GDN_FUSED_HEAD_DIM, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->fused_qkvz_bf16),
      static_cast<const __nv_bfloat16*>(args->conv_weight_bf16),
      static_cast<__nv_bfloat16*>(args->convolution_state_bf16),
      static_cast<__nv_bfloat16*>(args->q_normalized_bf16),
      static_cast<__nv_bfloat16*>(args->k_normalized_bf16),
      args->valid_tokens, args->source_row);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess)
    return CudaError("q27 fused GDN Q/K conv+norm: ", error);

  constexpr int kThreads = 256;
  ConvValueSplitZ128<<<
      (Q27_GDN_FUSED_Z_WIDTH + kThreads - 1) / kThreads, kThreads, 0,
      stream>>>(
      static_cast<const __nv_bfloat16*>(args->fused_qkvz_bf16),
      static_cast<const __nv_bfloat16*>(args->conv_weight_bf16),
      static_cast<__nv_bfloat16*>(args->convolution_state_bf16),
      static_cast<__nv_bfloat16*>(args->value_bf16),
      static_cast<__nv_bfloat16*>(args->projected_z_bf16),
      args->valid_tokens, args->source_row);
  error = cudaPeekAtLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("q27 fused GDN V conv/Z split: ", error);
}

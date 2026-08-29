/*
 * Fixed-shape Qwen3.8-27B BF16 GDN prefill primitives.
 *
 * This is a raw CUDA/cuBLAS translation of the state-carrying portion of
 * SGLang's Triton GDN prefill path in the Q27 oracle image at
 * c4271c3fe1262fc2adbd162c33b25de5255251c5. In particular, it preserves
 * the donor's 64-token chunk boundary, BF16 persistent/chunk state, BF16
 * rounding before W @ state, FP32 live state accumulation, and head repeat
 * from 16 key heads to 48 value heads.  See Q27_GDN_PREFILL.md for exact
 * provenance and the intentionally separate intra-chunk/output stages.
 */

#include "q27_gdn_prefill.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kF32Bytes = 4;
constexpr uint64_t kScratchAlignment = 256;
constexpr int kThreads = 256;
constexpr int kWarpThreads = 32;

constexpr uint64_t Align(uint64_t value) {
  return (value + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
}

constexpr uint64_t kConvStateBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_CONV_WIDTH) *
    Q27_GDN_PREFILL_CONV_HISTORY * kBf16Bytes;
constexpr uint64_t kRecurrentStateElements =
    static_cast<uint64_t>(Q27_GDN_PREFILL_VALUE_HEADS) *
    Q27_GDN_PREFILL_HEAD_DIM * Q27_GDN_PREFILL_HEAD_DIM;
constexpr uint64_t kRecurrentStateBytes =
    kRecurrentStateElements * kBf16Bytes;
constexpr uint64_t kMixedQkvBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_TOKENS) *
    Q27_GDN_PREFILL_CONV_WIDTH * kBf16Bytes;
constexpr uint64_t kQkBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_TOKENS) *
    Q27_GDN_PREFILL_QK_HEADS * Q27_GDN_PREFILL_HEAD_DIM * kBf16Bytes;
constexpr uint64_t kValueElements =
    static_cast<uint64_t>(Q27_GDN_PREFILL_TOKENS) *
    Q27_GDN_PREFILL_VALUE_HEADS * Q27_GDN_PREFILL_HEAD_DIM;
constexpr uint64_t kValueBytes = kValueElements * kBf16Bytes;
constexpr uint64_t kGateElements =
    static_cast<uint64_t>(Q27_GDN_PREFILL_TOKENS) *
    Q27_GDN_PREFILL_VALUE_HEADS;
constexpr uint64_t kGateInputBytes = kGateElements * kBf16Bytes;
constexpr uint64_t kGateOutputBytes = kGateElements * kF32Bytes;
constexpr uint64_t kChunkStatesBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_CHUNKS) * kRecurrentStateBytes;

constexpr uint64_t kStateF32Offset = 0;
constexpr uint64_t kStateF32Bytes = kRecurrentStateElements * kF32Bytes;
constexpr uint64_t kRoundedStateOffset =
    Align(kStateF32Offset + kStateF32Bytes);
constexpr uint64_t kRoundedStateBytes = kRecurrentStateBytes;
constexpr uint64_t kPackedKOffset =
    Align(kRoundedStateOffset + kRoundedStateBytes);
constexpr uint64_t kPackedChunkElements =
    static_cast<uint64_t>(Q27_GDN_PREFILL_VALUE_HEADS) *
    Q27_GDN_PREFILL_CHUNK * Q27_GDN_PREFILL_HEAD_DIM;
constexpr uint64_t kPackedChunkBytes = kPackedChunkElements * kBf16Bytes;
constexpr uint64_t kPackedWOffset = Align(kPackedKOffset + kPackedChunkBytes);
constexpr uint64_t kPredictionOffset =
    Align(kPackedWOffset + kPackedChunkBytes);
constexpr uint64_t kPredictionBytes = kPackedChunkElements * kF32Bytes;
constexpr uint64_t kVNewChunkOffset =
    Align(kPredictionOffset + kPredictionBytes);
constexpr uint64_t kVNewChunkBytes = kPackedChunkBytes;
constexpr uint64_t kChunkScratchBytes =
    Align(kVNewChunkOffset + kVNewChunkBytes);

static_assert(kConvStateBytes == 61440);
static_assert(kRecurrentStateBytes == 1572864);
static_assert(kMixedQkvBytes == 2621440);
static_assert(kQkBytes == 524288);
static_assert(kValueBytes == 1572864);
static_assert(kGateInputBytes == 12288);
static_assert(kGateOutputBytes == 24576);
static_assert(kChunkStatesBytes == 3145728);
static_assert(kChunkScratchBytes == 8650752);

thread_local std::string g_error;

q27_gdn_prefill_status Ok() { return {Q27_GDN_PREFILL_OK, "ok"}; }

q27_gdn_prefill_status Invalid(const char* message) {
  return {Q27_GDN_PREFILL_INVALID_ARGUMENT, message};
}

q27_gdn_prefill_status Unimplemented(const char* message) {
  return {Q27_GDN_PREFILL_UNIMPLEMENTED, message};
}

q27_gdn_prefill_status CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_PREFILL_CUDA_ERROR, g_error.c_str()};
}

q27_gdn_prefill_status CublasError(const char* prefix,
                                   cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append("cuBLAS status ");
  g_error.append(std::to_string(static_cast<int>(status)));
  return {Q27_GDN_PREFILL_CUBLAS_ERROR, g_error.c_str()};
}

bool Aligned(const void* pointer, uint64_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

__global__ void CausalConv128(const __nv_bfloat16* input,
                              const __nv_bfloat16* weight,
                              __nv_bfloat16* state,
                              __nv_bfloat16* output, uint32_t valid_tokens) {
  const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= Q27_GDN_PREFILL_CONV_WIDTH) return;

  __nv_bfloat16 h0 = state[channel * Q27_GDN_PREFILL_CONV_HISTORY + 0];
  __nv_bfloat16 h1 = state[channel * Q27_GDN_PREFILL_CONV_HISTORY + 1];
  __nv_bfloat16 h2 = state[channel * Q27_GDN_PREFILL_CONV_HISTORY + 2];
  const __nv_bfloat16* channel_weight =
      weight + channel * Q27_GDN_PREFILL_CONV_KERNEL;

#pragma unroll 1
  for (int token = 0; token < Q27_GDN_PREFILL_TOKENS; ++token) {
    if (static_cast<uint32_t>(token) >= valid_tokens) {
      output[static_cast<int64_t>(token) * Q27_GDN_PREFILL_CONV_WIDTH +
             channel] = __float2bfloat16_rn(0.0F);
      continue;
    }
    const __nv_bfloat16 x =
        input[static_cast<int64_t>(token) * Q27_GDN_PREFILL_CONV_WIDTH +
              channel];
    const __nv_bfloat16 values[Q27_GDN_PREFILL_CONV_KERNEL] = {h0, h1, h2,
                                                               x};
    float sum = 0.0F;
#pragma unroll
    for (int tap = 0; tap < Q27_GDN_PREFILL_CONV_KERNEL; ++tap) {
      // The donor's BF16 causal convolution rounds each product before its
      // FP32 accumulation.
      const __nv_bfloat16 product = __float2bfloat16_rn(
          __bfloat162float(channel_weight[tap]) *
          __bfloat162float(values[tap]));
      sum += __bfloat162float(product);
    }
    const float activated = sum / (1.0F + __expf(-sum));
    output[static_cast<int64_t>(token) * Q27_GDN_PREFILL_CONV_WIDTH +
           channel] = __float2bfloat16_rn(activated);
    h0 = h1;
    h1 = h2;
    h2 = x;
  }

  state[channel * Q27_GDN_PREFILL_CONV_HISTORY + 0] = h0;
  state[channel * Q27_GDN_PREFILL_CONV_HISTORY + 1] = h1;
  state[channel * Q27_GDN_PREFILL_CONV_HISTORY + 2] = h2;
}

__global__ void PrepareGates128(const __nv_bfloat16* a,
                                const __nv_bfloat16* b, const float* a_log,
                                const float* dt_bias, float* cumulative_g,
                                float* beta, uint32_t valid_tokens) {
  const int head = static_cast<int>(blockIdx.x);
  if (head >= Q27_GDN_PREFILL_VALUE_HEADS || threadIdx.x != 0) return;
  const float decay = -__expf(a_log[head]);

#pragma unroll
  for (int chunk = 0; chunk < Q27_GDN_PREFILL_CHUNKS; ++chunk) {
    float cumulative = 0.0F;
#pragma unroll 1
    for (int local = 0; local < Q27_GDN_PREFILL_CHUNK; ++local) {
      const int token = chunk * Q27_GDN_PREFILL_CHUNK + local;
      const int64_t index =
          static_cast<int64_t>(token) * Q27_GDN_PREFILL_VALUE_HEADS + head;
      if (static_cast<uint32_t>(token) >= valid_tokens) {
        cumulative_g[index] = 0.0F;
        beta[index] = 0.0F;
        continue;
      }
      const float gate_input = __bfloat162float(a[index]) + dt_bias[head];
      const float softplus = gate_input <= 20.0F
                                 ? logf(1.0F + __expf(gate_input))
                                 : gate_input;
      cumulative += decay * softplus;
      cumulative_g[index] = cumulative;
      const float beta_input = __bfloat162float(b[index]);
      const float sigmoid = 1.0F / (1.0F + __expf(-beta_input));
      // fused_gdn_gating.py allocates beta_output as FP32 but explicitly
      // rounds the sigmoid through b.dtype (BF16) before that FP32 store.
      beta[index] =
          __bfloat162float(__float2bfloat16_rn(sigmoid));
    }
  }
}

__global__ void LoadStateF32(const __nv_bfloat16* input, float* output) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < kRecurrentStateElements)
    output[index] = __bfloat162float(input[index]);
}

__global__ void SaveRoundedState(const float* input,
                                 __nv_bfloat16* rounded,
                                 __nv_bfloat16* chunk_state) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kRecurrentStateElements) return;
  const __nv_bfloat16 value = __float2bfloat16_rn(input[index]);
  rounded[index] = value;
  chunk_state[index] = value;
}

__global__ void PackChunkKW(const __nv_bfloat16* k,
                            const __nv_bfloat16* w, int token_offset,
                            int chunk_tokens,
                            __nv_bfloat16* packed_k,
                            __nv_bfloat16* packed_w) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kPackedChunkElements) return;
  const int dimension = index % Q27_GDN_PREFILL_HEAD_DIM;
  const uint64_t row = index / Q27_GDN_PREFILL_HEAD_DIM;
  const int local_token = row % Q27_GDN_PREFILL_CHUNK;
  const int value_head = row / Q27_GDN_PREFILL_CHUNK;
  const int token = token_offset + local_token;
  const int key_head = value_head /
                       (Q27_GDN_PREFILL_VALUE_HEADS /
                        Q27_GDN_PREFILL_QK_HEADS);
  const uint64_t packed_index =
      (static_cast<uint64_t>(value_head) * Q27_GDN_PREFILL_CHUNK +
       local_token) *
          Q27_GDN_PREFILL_HEAD_DIM +
      dimension;
  if (local_token >= chunk_tokens) {
    packed_k[packed_index] = __float2bfloat16_rn(0.0F);
    packed_w[packed_index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const uint64_t k_index =
      (static_cast<uint64_t>(token) * Q27_GDN_PREFILL_QK_HEADS + key_head) *
          Q27_GDN_PREFILL_HEAD_DIM +
      dimension;
  const uint64_t w_index =
      (static_cast<uint64_t>(token) * Q27_GDN_PREFILL_VALUE_HEADS +
       value_head) *
          Q27_GDN_PREFILL_HEAD_DIM +
      dimension;
  packed_k[packed_index] = k[k_index];
  packed_w[packed_index] = w[w_index];
}

__global__ void FormVNew(const __nv_bfloat16* u, const float* prediction,
                         const float* cumulative_g, int token_offset,
                         int chunk_tokens,
                         __nv_bfloat16* packed_v_new,
                         __nv_bfloat16* token_major_v_new) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kPackedChunkElements) return;
  const int dimension = index % Q27_GDN_PREFILL_HEAD_DIM;
  const uint64_t row = index / Q27_GDN_PREFILL_HEAD_DIM;
  const int local_token = row % Q27_GDN_PREFILL_CHUNK;
  const int head = row / Q27_GDN_PREFILL_CHUNK;
  const int token = token_offset + local_token;
  if (local_token >= chunk_tokens) {
    packed_v_new[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const int last_token = token_offset + chunk_tokens - 1;
  const uint64_t token_index =
      (static_cast<uint64_t>(token) * Q27_GDN_PREFILL_VALUE_HEADS + head) *
          Q27_GDN_PREFILL_HEAD_DIM +
      dimension;
  const float log_scale =
      cumulative_g[static_cast<int64_t>(last_token) *
                       Q27_GDN_PREFILL_VALUE_HEADS +
                   head] -
      cumulative_g[static_cast<int64_t>(token) *
                       Q27_GDN_PREFILL_VALUE_HEADS +
                   head];
  const float raw_value = __bfloat162float(u[token_index]) - prediction[index];
  // The donor publishes the ungated residual for chunk_fwd_o, then applies
  // exp(g_last-g_t) only to the copy accumulated into the final state.
  token_major_v_new[token_index] = __float2bfloat16_rn(raw_value);
  packed_v_new[index] =
      __float2bfloat16_rn(raw_value * __expf(log_scale));
}

__global__ void ScaleState(float* state, const float* cumulative_g,
                           int last_token) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kRecurrentStateElements) return;
  const int head = index /
                   (Q27_GDN_PREFILL_HEAD_DIM * Q27_GDN_PREFILL_HEAD_DIM);
  state[index] *=
      __expf(cumulative_g[static_cast<int64_t>(last_token) *
                              Q27_GDN_PREFILL_VALUE_HEADS +
                          head]);
}

__global__ void StoreStateBf16(const float* input, __nv_bfloat16* output) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < kRecurrentStateElements)
    output[index] = __float2bfloat16_rn(input[index]);
}

__device__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset /= 2)
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  return value;
}

__global__ __launch_bounds__(kWarpThreads) void GatedRmsNorm128(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t valid_tokens) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t begin =
      static_cast<int64_t>(row) * Q27_GDN_PREFILL_HEAD_DIM;
  const int token = row / Q27_GDN_PREFILL_VALUE_HEADS;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
#pragma unroll
    for (int item = 0; item < 4; ++item)
      output[begin + lane + item * kWarpThreads] =
          __float2bfloat16_rn(0.0F);
    return;
  }
  float sum = 0.0F;
  float values[4];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const float value =
        __bfloat162float(input[begin + lane + item * kWarpThreads]);
    values[item] = value;
    sum += value * value;
  }
  sum = WarpReduceSum(sum);
  __shared__ float inverse_rms;
  if (lane == 0)
    inverse_rms = rsqrtf(sum / Q27_GDN_PREFILL_HEAD_DIM + 1.0e-6F);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = lane + item * kWarpThreads;
    const float z = __bfloat162float(gate[begin + column]);
    const float normalized = values[item] * inverse_rms *
                             __bfloat162float(weight[column]);
    const float silu_z = z / (1.0F + __expf(-z));
    output[begin + column] = __float2bfloat16_rn(normalized * silu_z);
  }
}

uint32_t Blocks(uint64_t elements) {
  return static_cast<uint32_t>((elements + kThreads - 1) / kThreads);
}

q27_gdn_prefill_status CheckLaunch(const char* prefix) {
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok() : CudaError(prefix, error);
}

}  // namespace

extern "C" q27_gdn_prefill_status q27_gdn_prefill_query(
    uint32_t tokens, q27_gdn_prefill_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_PREFILL_ABI_VERSION)
    return Invalid("invalid q27 GDN prefill layout query");
  if (tokens != Q27_GDN_PREFILL_TOKENS)
    return Unimplemented("only the fixed M=128 GDN prefill capsule is implemented");
  output->tokens = Q27_GDN_PREFILL_TOKENS;
  output->chunk_size = Q27_GDN_PREFILL_CHUNK;
  output->chunk_count = Q27_GDN_PREFILL_CHUNKS;
  output->reserved = 0;
  output->convolution_state_bytes = kConvStateBytes;
  output->recurrent_state_bytes = kRecurrentStateBytes;
  output->mixed_qkv_bytes = kMixedQkvBytes;
  output->qk_bytes = kQkBytes;
  output->value_bytes = kValueBytes;
  output->gate_input_bytes = kGateInputBytes;
  output->gate_output_bytes = kGateOutputBytes;
  output->chunk_states_bytes = kChunkStatesBytes;
  output->v_new_bytes = kValueBytes;
  output->chunk_scratch_bytes = kChunkScratchBytes;
  output->scratch_alignment = kScratchAlignment;
  return Ok();
}

extern "C" q27_gdn_prefill_status q27_gdn_prefill_causal_conv(
    const q27_gdn_prefill_conv_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_ABI_VERSION ||
      args->valid_tokens == 0 ||
      args->valid_tokens > Q27_GDN_PREFILL_TOKENS ||
      args->mixed_qkv_bf16 == nullptr || args->conv_weight_bf16 == nullptr ||
      args->convolution_state_bf16 == nullptr ||
      args->convolved_qkv_bf16 == nullptr ||
      args->mixed_qkv_bytes < kMixedQkvBytes ||
      args->conv_weight_bytes <
          static_cast<uint64_t>(Q27_GDN_PREFILL_CONV_WIDTH) *
              Q27_GDN_PREFILL_CONV_KERNEL * kBf16Bytes ||
      args->convolution_state_bytes < kConvStateBytes ||
      args->convolved_qkv_bytes < kMixedQkvBytes)
    return Invalid("invalid q27 GDN prefill convolution arguments");
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  CausalConv128<<<(Q27_GDN_PREFILL_CONV_WIDTH + kThreads - 1) / kThreads,
                  kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->mixed_qkv_bf16),
      static_cast<const __nv_bfloat16*>(args->conv_weight_bf16),
      static_cast<__nv_bfloat16*>(args->convolution_state_bf16),
      static_cast<__nv_bfloat16*>(args->convolved_qkv_bf16),
      args->valid_tokens);
  return CheckLaunch("q27 GDN prefill causal convolution: ");
}

extern "C" q27_gdn_prefill_status q27_gdn_prefill_prepare_gates(
    const q27_gdn_prefill_gate_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_ABI_VERSION ||
      args->valid_tokens == 0 ||
      args->valid_tokens > Q27_GDN_PREFILL_TOKENS ||
      args->projected_a_bf16 == nullptr || args->projected_b_bf16 == nullptr ||
      args->a_log_f32 == nullptr || args->dt_bias_f32 == nullptr ||
      args->cumulative_g_f32 == nullptr || args->beta_f32 == nullptr ||
      args->projected_a_bytes < kGateInputBytes ||
      args->projected_b_bytes < kGateInputBytes ||
      args->cumulative_g_bytes < kGateOutputBytes ||
      args->beta_bytes < kGateOutputBytes)
    return Invalid("invalid q27 GDN prefill gate arguments");
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  PrepareGates128<<<Q27_GDN_PREFILL_VALUE_HEADS, 1, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->projected_a_bf16),
      static_cast<const __nv_bfloat16*>(args->projected_b_bf16),
      args->a_log_f32, args->dt_bias_f32, args->cumulative_g_f32,
      args->beta_f32, args->valid_tokens);
  return CheckLaunch("q27 GDN prefill gate preparation: ");
}

extern "C" q27_gdn_prefill_status q27_gdn_prefill_chunk_state(
    const q27_gdn_prefill_chunk_state_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_ABI_VERSION ||
      args->valid_tokens == 0 ||
      args->valid_tokens > Q27_GDN_PREFILL_TOKENS ||
      args->k_bf16 == nullptr || args->w_bf16 == nullptr ||
      args->u_bf16 == nullptr || args->cumulative_g_f32 == nullptr ||
      args->recurrent_state_bf16 == nullptr ||
      args->chunk_states_bf16 == nullptr || args->v_new_bf16 == nullptr ||
      args->cublas_handle == nullptr ||
      !Aligned(args->scratch, kScratchAlignment) ||
      args->k_bytes < kQkBytes || args->w_bytes < kValueBytes ||
      args->u_bytes < kValueBytes ||
      args->cumulative_g_bytes < kGateOutputBytes ||
      args->recurrent_state_bytes < kRecurrentStateBytes ||
      args->chunk_states_bytes < kChunkStatesBytes ||
      args->v_new_bytes < kValueBytes ||
      args->scratch_bytes < kChunkScratchBytes)
    return Invalid("invalid q27 GDN prefill chunk-state arguments");

  cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(args->cublas_handle);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasPointerMode_t pointer_mode;
  cublasStatus_t blas = cublasGetPointerMode(handle, &pointer_mode);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN prefill get pointer mode: ", blas);
  if (pointer_mode != CUBLAS_POINTER_MODE_HOST)
    return Invalid("q27 GDN prefill requires cuBLAS host pointer mode");
  blas = cublasSetStream(handle, stream);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN prefill set stream: ", blas);

  auto* scratch = static_cast<uint8_t*>(args->scratch);
  auto* state_f32 = reinterpret_cast<float*>(scratch + kStateF32Offset);
  auto* rounded_state = reinterpret_cast<__nv_bfloat16*>(
      scratch + kRoundedStateOffset);
  auto* packed_k =
      reinterpret_cast<__nv_bfloat16*>(scratch + kPackedKOffset);
  auto* packed_w =
      reinterpret_cast<__nv_bfloat16*>(scratch + kPackedWOffset);
  auto* prediction = reinterpret_cast<float*>(scratch + kPredictionOffset);
  auto* packed_v_new =
      reinterpret_cast<__nv_bfloat16*>(scratch + kVNewChunkOffset);

  cudaError_t cuda =
      cudaMemsetAsync(args->chunk_states_bf16, 0, kChunkStatesBytes, stream);
  if (cuda != cudaSuccess)
    return CudaError("q27 GDN prefill clear chunk states: ", cuda);
  cuda = cudaMemsetAsync(args->v_new_bf16, 0, kValueBytes, stream);
  if (cuda != cudaSuccess)
    return CudaError("q27 GDN prefill clear padded v_new: ", cuda);

  LoadStateF32<<<Blocks(kRecurrentStateElements), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->recurrent_state_bf16),
      state_f32);
  q27_gdn_prefill_status launch =
      CheckLaunch("q27 GDN prefill load recurrent state: ");
  if (launch.code != Q27_GDN_PREFILL_OK) return launch;

  const float alpha = 1.0F;
  const float zero = 0.0F;
  const float one = 1.0F;
  const long long state_stride =
      static_cast<long long>(Q27_GDN_PREFILL_HEAD_DIM) *
      Q27_GDN_PREFILL_HEAD_DIM;
  const long long chunk_stride =
      static_cast<long long>(Q27_GDN_PREFILL_CHUNK) *
      Q27_GDN_PREFILL_HEAD_DIM;

  const int active_chunks =
      (args->valid_tokens + Q27_GDN_PREFILL_CHUNK - 1) /
      Q27_GDN_PREFILL_CHUNK;
  for (int chunk = 0; chunk < active_chunks; ++chunk) {
    const int token_offset = chunk * Q27_GDN_PREFILL_CHUNK;
    const int chunk_tokens =
        std::min(static_cast<int>(args->valid_tokens) - token_offset,
                 static_cast<int>(Q27_GDN_PREFILL_CHUNK));
    auto* saved_chunk =
        static_cast<__nv_bfloat16*>(args->chunk_states_bf16) +
        static_cast<uint64_t>(chunk) * kRecurrentStateElements;
    SaveRoundedState<<<Blocks(kRecurrentStateElements), kThreads, 0, stream>>>(
        state_f32, rounded_state, saved_chunk);
    PackChunkKW<<<Blocks(kPackedChunkElements), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args->k_bf16),
        static_cast<const __nv_bfloat16*>(args->w_bf16), token_offset,
        chunk_tokens, packed_k, packed_w);
    launch = CheckLaunch("q27 GDN prefill pack chunk/state: ");
    if (launch.code != Q27_GDN_PREFILL_OK) return launch;

    // prediction[BT,V] = W[BT,K] * BF16(state[V,K])^T, batched over
    // 48 heads. Row-major buffers are intentionally presented as the
    // equivalent transposed column-major matrices to cuBLAS.
    blas = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, Q27_GDN_PREFILL_HEAD_DIM,
        Q27_GDN_PREFILL_CHUNK, Q27_GDN_PREFILL_HEAD_DIM, &alpha,
        rounded_state, CUDA_R_16BF, Q27_GDN_PREFILL_HEAD_DIM, state_stride,
        packed_w, CUDA_R_16BF, Q27_GDN_PREFILL_HEAD_DIM, chunk_stride, &zero,
        prediction, CUDA_R_32F, Q27_GDN_PREFILL_HEAD_DIM, chunk_stride,
        Q27_GDN_PREFILL_VALUE_HEADS, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas != CUBLAS_STATUS_SUCCESS)
      return CublasError("q27 GDN prefill W/state GEMM: ", blas);

    FormVNew<<<Blocks(kPackedChunkElements), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args->u_bf16), prediction,
        args->cumulative_g_f32, token_offset, chunk_tokens, packed_v_new,
        static_cast<__nv_bfloat16*>(args->v_new_bf16));
    ScaleState<<<Blocks(kRecurrentStateElements), kThreads, 0, stream>>>(
        state_f32, args->cumulative_g_f32,
        token_offset + chunk_tokens - 1);
    launch = CheckLaunch("q27 GDN prefill form gated chunk update: ");
    if (launch.code != Q27_GDN_PREFILL_OK) return launch;

    // state[V,K] += v_new[BT,V]^T * K[BT,K].  In column-major view this is
    // state^T[K,V] += K^T[K,BT] * v_new[BT,V].
    blas = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, Q27_GDN_PREFILL_HEAD_DIM,
        Q27_GDN_PREFILL_HEAD_DIM, Q27_GDN_PREFILL_CHUNK, &alpha, packed_k,
        CUDA_R_16BF, Q27_GDN_PREFILL_HEAD_DIM, chunk_stride, packed_v_new,
        CUDA_R_16BF, Q27_GDN_PREFILL_HEAD_DIM, chunk_stride, &one, state_f32,
        CUDA_R_32F, Q27_GDN_PREFILL_HEAD_DIM, state_stride,
        Q27_GDN_PREFILL_VALUE_HEADS, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas != CUBLAS_STATUS_SUCCESS)
      return CublasError("q27 GDN prefill state update GEMM: ", blas);
  }

  StoreStateBf16<<<Blocks(kRecurrentStateElements), kThreads, 0, stream>>>(
      state_f32,
      static_cast<__nv_bfloat16*>(args->recurrent_state_bf16));
  return CheckLaunch("q27 GDN prefill store recurrent state: ");
}

extern "C" q27_gdn_prefill_status q27_gdn_prefill_gated_norm(
    const q27_gdn_prefill_norm_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_ABI_VERSION ||
      args->valid_tokens == 0 ||
      args->valid_tokens > Q27_GDN_PREFILL_TOKENS ||
      args->recurrent_output_bf16 == nullptr ||
      args->projected_z_bf16 == nullptr || args->norm_weight_bf16 == nullptr ||
      args->normalized_output_bf16 == nullptr ||
      args->recurrent_output_bytes < kValueBytes ||
      args->projected_z_bytes < kValueBytes ||
      args->norm_weight_bytes <
          static_cast<uint64_t>(Q27_GDN_PREFILL_HEAD_DIM) * kBf16Bytes ||
      args->normalized_output_bytes < kValueBytes)
    return Invalid("invalid q27 GDN prefill gated-norm arguments");
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  GatedRmsNorm128<<<Q27_GDN_PREFILL_TOKENS *
                        Q27_GDN_PREFILL_VALUE_HEADS,
                    kWarpThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->recurrent_output_bf16),
      static_cast<const __nv_bfloat16*>(args->projected_z_bf16),
      static_cast<const __nv_bfloat16*>(args->norm_weight_bf16),
      static_cast<__nv_bfloat16*>(args->normalized_output_bf16),
      args->valid_tokens);
  return CheckLaunch("q27 GDN prefill gated RMSNorm: ");
}

// SPDX-License-Identifier: Apache-2.0
// Fixed BF16 projection/QK-norm/NeoX-RoPE/O shell for pinned DFlash2 attention.

#include "q27_dflash2_attention.h"
#include "q27_dflash2_conv.h"
#include "q27_dflash2_flashinfer.h"
#include "q27_dflash2_model.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr uint32_t kTokens = Q27_DFLASH2_BLOCK_SIZE;
constexpr uint32_t kHidden = Q27_DFLASH2_HIDDEN_SIZE;
constexpr uint32_t kQColumns =
    Q27_DFLASH2_QUERY_HEADS * Q27_DFLASH2_HEAD_DIM;
constexpr uint32_t kKvColumns =
    Q27_DFLASH2_KV_HEADS * Q27_DFLASH2_HEAD_DIM;
constexpr uint32_t kRotaryPairs = Q27_DFLASH2_HEAD_DIM / 2;
constexpr uint32_t kNormThreads = Q27_DFLASH2_HEAD_DIM;
constexpr uint32_t kNormWarps = kNormThreads / 32;
constexpr float kRopeTheta = 10000000.0F;
constexpr float kDefaultEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 0.08838834764831845F;
constexpr uint32_t kWindowLeft = Q27_DFLASH2_SLIDING_WINDOW - 1;

constexpr uint64_t kInputBytes = kTokens * kHidden * 2ULL;
constexpr uint64_t kQBytes = Q27_DFLASH2_ATTENTION_Q_BYTES;
constexpr uint64_t kKvBytes = Q27_DFLASH2_ATTENTION_KV_BYTES;
constexpr uint64_t kContextBytes = Q27_DFLASH2_ATTENTION_CONTEXT_BYTES;
constexpr uint64_t kOutputBytes = Q27_DFLASH2_ATTENTION_OUTPUT_BYTES;
constexpr uint64_t kPositionBytes = kTokens * sizeof(uint64_t);
constexpr uint64_t kFrequencyBytes =
    Q27_DFLASH2_ATTENTION_ROPE_FREQUENCY_BYTES;
constexpr uint64_t kRopeCacheBytes = Q27_DFLASH2_ATTENTION_ROPE_CACHE_BYTES;
constexpr uint64_t kQWeightBytes =
    static_cast<uint64_t>(kQColumns) * kHidden * 2ULL;
constexpr uint64_t kKvWeightBytes =
    static_cast<uint64_t>(kKvColumns) * kHidden * 2ULL;
constexpr uint64_t kOWeightBytes =
    static_cast<uint64_t>(kHidden) * kQColumns * 2ULL;
constexpr uint64_t kNormWeightBytes = Q27_DFLASH2_HEAD_DIM * 2ULL;

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
  if (args == nullptr) return Invalid("DFlash2 attention arguments are null");
  if (args->struct_size < sizeof(Args)) {
    return Invalid("DFlash2 attention struct_size is too small");
  }
  if (args->abi_version != Q27_DFLASH2_ATTENTION_ABI_VERSION) {
    return Invalid("DFlash2 attention ABI version mismatch");
  }
  return Ok();
}

struct BufferRange {
  const void* data;
  uint64_t bytes;
};

bool InvalidRange(BufferRange range) {
  if (range.data == nullptr ||
      (reinterpret_cast<uintptr_t>(range.data) & 1U) != 0) {
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

__device__ float WarpSum(float value) {
#pragma unroll
  for (uint32_t offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

__global__ void InitializeRopeFrequencies(float* inverse_frequencies) {
  const uint32_t pair = threadIdx.x;
  if (pair < kRotaryPairs) {
    inverse_frequencies[pair] =
        expf(-logf(kRopeTheta) * static_cast<float>(pair) /
             static_cast<float>(kRotaryPairs));
  }
}

__global__ void BuildRopeCache(const uint64_t* positions,
                               const float* inverse_frequencies,
                               float* rope_cache) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTokens * kRotaryPairs; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kRotaryPairs;
    const uint32_t pair = index - token * kRotaryPairs;
    const float angle = static_cast<float>(positions[token]) *
                        inverse_frequencies[pair];
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    rope_cache[static_cast<uint64_t>(index) * 2] = cosine;
    rope_cache[static_cast<uint64_t>(index) * 2 + 1] = sine;
  }
}

__global__ void QkNormNeoXRope(__nv_bfloat16* q, __nv_bfloat16* k,
                               const __nv_bfloat16* q_gamma,
                               const __nv_bfloat16* k_gamma,
                               const float* rope_cache, float epsilon) {
  const uint32_t q_blocks = kTokens * Q27_DFLASH2_QUERY_HEADS;
  const bool is_q = blockIdx.x < q_blocks;
  const uint32_t local_block = is_q ? blockIdx.x : blockIdx.x - q_blocks;
  const uint32_t heads =
      is_q ? Q27_DFLASH2_QUERY_HEADS : Q27_DFLASH2_KV_HEADS;
  const uint32_t token = local_block / heads;
  const uint32_t head = local_block - token * heads;
  const uint32_t dimension = threadIdx.x;
  __nv_bfloat16* values = is_q ? q : k;
  const __nv_bfloat16* gamma = is_q ? q_gamma : k_gamma;
  const uint64_t base =
      (static_cast<uint64_t>(token) * heads + head) * Q27_DFLASH2_HEAD_DIM;

  float value = __bfloat162float(values[base + dimension]);
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
      rsqrtf(warp_sums[0] / static_cast<float>(Q27_DFLASH2_HEAD_DIM) +
             epsilon);

  __shared__ __nv_bfloat16 normalized[Q27_DFLASH2_HEAD_DIM];
  normalized[dimension] = __float2bfloat16_rn(
      value * inverse_rms * __bfloat162float(gamma[dimension]));
  __syncthreads();
  if (dimension < kRotaryPairs) {
    const float first = __bfloat162float(normalized[dimension]);
    const float second =
        __bfloat162float(normalized[dimension + kRotaryPairs]);
    const uint64_t rope =
        (static_cast<uint64_t>(token) * kRotaryPairs + dimension) * 2;
    const float cosine = rope_cache[rope];
    const float sine = rope_cache[rope + 1];
    values[base + dimension] =
        __float2bfloat16_rn(first * cosine - second * sine);
    values[base + dimension + kRotaryPairs] =
        __float2bfloat16_rn(second * cosine + first * sine);
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

bool ValidWeights(const q27_dflash2_layer_weights& weights) {
  return weights.q_proj.data != nullptr && weights.q_proj.bytes == kQWeightBytes &&
         weights.k_proj.data != nullptr &&
         weights.k_proj.bytes == kKvWeightBytes &&
         weights.v_proj.data != nullptr &&
         weights.v_proj.bytes == kKvWeightBytes &&
         weights.o_proj.data != nullptr && weights.o_proj.bytes == kOWeightBytes &&
         weights.q_norm.data != nullptr &&
         weights.q_norm.bytes == kNormWeightBytes &&
         weights.k_norm.data != nullptr &&
         weights.k_norm.bytes == kNormWeightBytes;
}

bool ValidState(const q27_dflash2_state_view* state) {
  return state != nullptr && state->struct_size >= sizeof(*state) &&
         state->abi_version == Q27_DFLASH2_ABI_VERSION &&
         state->key_cache_bf16 != nullptr && state->value_cache_bf16 != nullptr &&
         state->position_tags_u64 != nullptr &&
         state->committed_length <=
             Q27_DFLASH2_MAX_POSITION - Q27_DFLASH2_BLOCK_SIZE &&
         (reinterpret_cast<uintptr_t>(state->position_tags_u64) & 7U) == 0;
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_initialize_rope(
    const q27_dflash2_rope_init_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->inverse_frequencies_f32 == nullptr ||
      (reinterpret_cast<uintptr_t>(args->inverse_frequencies_f32) & 3U) != 0) {
    return Invalid("invalid DFlash2 RoPE frequency output");
  }
  InitializeRopeFrequencies<<<1, kRotaryPairs, 0,
                              static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->inverse_frequencies_f32);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("DFlash2 RoPE initialization", error);
}

extern "C" q27_dflash2_status q27_dflash2_attention_forward(
    const q27_dflash2_attention_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->weights == nullptr || !ValidWeights(*args->weights) ||
      args->layer_index >= Q27_DFLASH2_LAYERS || !ValidState(args->state) ||
      args->cublas_handle == nullptr || !std::isfinite(args->rms_epsilon) ||
      args->rms_epsilon < 0.0F) {
    return Invalid("invalid DFlash2 attention metadata or weights");
  }

  if (args->workspace_bytes < Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES ||
      (reinterpret_cast<uintptr_t>(args->workspace) & 15U) != 0) {
    return Invalid("invalid DFlash2 FlashInfer workspace");
  }
  const std::array<BufferRange, 10> runtime_buffers{{
      {args->input_bf16, kInputBytes},
      {args->positions_u64, kPositionBytes},
      {args->rope_inverse_frequencies_f32, kFrequencyBytes},
      {args->rope_cache_f32, kRopeCacheBytes},
      {args->q_bf16, kQBytes},
      {args->k_bf16, kKvBytes},
      {args->v_bf16, kKvBytes},
      {args->context_bf16, kContextBytes},
      {args->output_bf16, kOutputBytes},
      {args->workspace, Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES},
  }};
  if (!Disjoint(runtime_buffers)) {
    return Invalid("DFlash2 attention buffers are null, misaligned, or overlap");
  }
  if ((reinterpret_cast<uintptr_t>(args->positions_u64) & 7U) != 0 ||
      (reinterpret_cast<uintptr_t>(args->rope_inverse_frequencies_f32) & 3U) !=
          0 ||
      (reinterpret_cast<uintptr_t>(args->rope_cache_f32) & 3U) != 0) {
    return Invalid("DFlash2 attention typed buffers are misaligned");
  }
  const std::array<BufferRange, 6> weight_buffers{{
      {args->weights->q_proj.data, kQWeightBytes},
      {args->weights->k_proj.data, kKvWeightBytes},
      {args->weights->v_proj.data, kKvWeightBytes},
      {args->weights->o_proj.data, kOWeightBytes},
      {args->weights->q_norm.data, kNormWeightBytes},
      {args->weights->k_norm.data, kNormWeightBytes},
  }};
  if (!Disjoint(weight_buffers)) {
    return Invalid("DFlash2 attention weight buffers overlap");
  }
  const std::array<BufferRange, 3> state_buffers{{
      {args->state->key_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {args->state->value_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {args->state->position_tags_u64, Q27_DFLASH2_POSITION_TAG_BYTES},
  }};
  if (!Disjoint(state_buffers)) {
    return Invalid("DFlash2 attention persistent state buffers overlap");
  }
  for (BufferRange runtime : runtime_buffers) {
    for (BufferRange weight : weight_buffers) {
      if (Overlap(runtime, weight)) {
        return Invalid("DFlash2 attention scratch overlaps checkpoint weights");
      }
    }
    for (BufferRange state : state_buffers) {
      if (Overlap(runtime, state)) {
        return Invalid("DFlash2 attention scratch overlaps persistent state");
      }
    }
  }

  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("DFlash2 attention set stream", cublas_status);
  }
  status = ProjectRows(handle, args->input_bf16, args->weights->q_proj.data,
                       args->q_bf16, kTokens, kHidden, kQColumns,
                       "DFlash2 Q projection");
  if (status.code != Q27_DFLASH2_OK) return status;
  status = ProjectRows(handle, args->input_bf16, args->weights->k_proj.data,
                       args->k_bf16, kTokens, kHidden, kKvColumns,
                       "DFlash2 K projection");
  if (status.code != Q27_DFLASH2_OK) return status;
  status = ProjectRows(handle, args->input_bf16, args->weights->v_proj.data,
                       args->v_bf16, kTokens, kHidden, kKvColumns,
                       "DFlash2 V projection");
  if (status.code != Q27_DFLASH2_OK) return status;

  BuildRopeCache<<<2, 256, 0, stream>>>(
      args->positions_u64, args->rope_inverse_frequencies_f32,
      args->rope_cache_f32);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("DFlash2 RoPE cache launch", error);
  }
  constexpr uint32_t norm_blocks =
      kTokens * (Q27_DFLASH2_QUERY_HEADS + Q27_DFLASH2_KV_HEADS);
  const float epsilon =
      args->rms_epsilon > 0.0F ? args->rms_epsilon : kDefaultEpsilon;
  QkNormNeoXRope<<<norm_blocks, kNormThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(args->q_bf16),
      static_cast<__nv_bfloat16*>(args->k_bf16),
      static_cast<const __nv_bfloat16*>(args->weights->q_norm.data),
      static_cast<const __nv_bfloat16*>(args->weights->k_norm.data),
      args->rope_cache_f32, epsilon);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("DFlash2 QK norm/RoPE launch", error);
  }

  q27_dflash2_sliding_attention_call call{};
  call.struct_size = sizeof(call);
  call.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  call.layer_index = args->layer_index;
  call.token_count = kTokens;
  call.positions_u64 = args->positions_u64;
  call.q_bf16 = args->q_bf16;
  call.k_bf16 = args->k_bf16;
  call.v_bf16 = args->v_bf16;
  call.context_bf16 = args->context_bf16;
  call.state = args->state;
  call.workspace = args->workspace;
  call.workspace_bytes = args->workspace_bytes;
  call.scale = kAttentionScale;
  call.window_left = kWindowLeft;
  call.cuda_stream = args->cuda_stream;
  status = q27_dflash2_flashinfer_sliding_attention(&call, nullptr);
  if (status.code != Q27_DFLASH2_OK) return status;

  return ProjectRows(handle, args->context_bf16, args->weights->o_proj.data,
                     args->output_bf16, kTokens, kQColumns, kHidden,
                     "DFlash2 O projection");
}

extern "C" q27_dflash2_status q27_dflash2_attention_sublayer(
    const q27_dflash2_sublayer_call* call, void* user_data) {
  if (user_data != nullptr) {
    return Invalid("DFlash2 fixed attention user_data must be null");
  }
  if (call == nullptr || call->struct_size < sizeof(*call) ||
      call->abi_version != Q27_DFLASH2_MODEL_ABI_VERSION ||
      call->layer_index >= Q27_DFLASH2_LAYERS ||
      call->batch_size != 1 || call->token_count != kTokens ||
      call->weights == nullptr || call->positions_u64 == nullptr ||
      call->input_bf16 == nullptr || call->output_bf16 == nullptr ||
      !ValidState(call->state) || call->workspace == nullptr ||
      call->workspace_bytes <
          Q27_DFLASH2_ATTENTION_SUBLAYER_WORKSPACE_BYTES ||
      call->cublas_handle == nullptr ||
      (reinterpret_cast<uintptr_t>(call->positions_u64) & 7U) != 0 ||
      (reinterpret_cast<uintptr_t>(call->workspace) & 15U) != 0) {
    return Invalid("invalid fixed DFlash2 attention sublayer call");
  }
  const BufferRange outer_buffers[] = {
      {call->positions_u64, kPositionBytes},
      {call->input_bf16, kInputBytes},
      {call->output_bf16, kOutputBytes},
      {call->workspace, Q27_DFLASH2_ATTENTION_SUBLAYER_WORKSPACE_BYTES},
  };
  for (uint32_t left = 0;
       left < sizeof(outer_buffers) / sizeof(outer_buffers[0]); ++left) {
    if (InvalidRange(outer_buffers[left])) {
      return Invalid("invalid fixed DFlash2 attention outer buffer");
    }
    for (uint32_t right = left + 1;
         right < sizeof(outer_buffers) / sizeof(outer_buffers[0]); ++right) {
      if (InvalidRange(outer_buffers[right]) ||
          Overlap(outer_buffers[left], outer_buffers[right])) {
        return Invalid("fixed DFlash2 attention outer buffers overlap");
      }
    }
  }
  const BufferRange persistent_buffers[] = {
      {call->state->key_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {call->state->value_cache_bf16, Q27_DFLASH2_ONE_KV_CACHE_BYTES},
      {call->state->position_tags_u64, Q27_DFLASH2_POSITION_TAG_BYTES},
  };
  for (BufferRange persistent : persistent_buffers) {
    if (InvalidRange(persistent)) {
      return Invalid("invalid fixed DFlash2 attention persistent state");
    }
    for (BufferRange outer : outer_buffers) {
      if (Overlap(persistent, outer)) {
        return Invalid("fixed DFlash2 attention overlaps persistent state");
      }
    }
  }
  const q27_dflash2_layer_weights& weights = *call->weights;
  const BufferRange fixed_weights[] = {
      {weights.attention_conv_base.data, weights.attention_conv_base.bytes},
      {weights.attention_conv_projection.data,
       weights.attention_conv_projection.bytes},
      {weights.q_proj.data, weights.q_proj.bytes},
      {weights.k_proj.data, weights.k_proj.bytes},
      {weights.v_proj.data, weights.v_proj.bytes},
      {weights.o_proj.data, weights.o_proj.bytes},
      {weights.q_norm.data, weights.q_norm.bytes},
      {weights.k_norm.data, weights.k_norm.bytes},
  };
  for (BufferRange weight : fixed_weights) {
    if (InvalidRange(weight)) {
      return Invalid("invalid fixed DFlash2 attention weight buffer");
    }
    for (BufferRange outer : outer_buffers) {
      if (Overlap(weight, outer)) {
        return Invalid("fixed DFlash2 attention workspace overlaps weights");
      }
    }
  }

  uint8_t* workspace = static_cast<uint8_t*>(call->workspace);
  void* rope_frequencies =
      workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_ROPE_FREQUENCY_OFFSET;
  void* rope_cache =
      workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_ROPE_CACHE_OFFSET;
  void* coefficients =
      workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_CONV_COEFFICIENT_OFFSET;
  void* prepared = workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_PREPARED_OFFSET;
  void* q = workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_Q_OFFSET;
  void* k = workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_K_OFFSET;
  void* v = workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_V_OFFSET;
  void* context = workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_CONTEXT_OFFSET;
  void* raw_output =
      workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_RAW_OUTPUT_OFFSET;
  void* flashinfer =
      workspace + Q27_DFLASH2_ATTENTION_SUBLAYER_FLASHINFER_OFFSET;

  q27_dflash2_rope_init_args rope{};
  rope.struct_size = sizeof(rope);
  rope.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  rope.inverse_frequencies_f32 = static_cast<float*>(rope_frequencies);
  rope.cuda_stream = call->cuda_stream;
  q27_dflash2_status status = q27_dflash2_initialize_rope(&rope);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_conv_prepare_args prepare{};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = Q27_DFLASH2_CONV_ABI_VERSION;
  prepare.base_kernel = call->weights->attention_conv_base;
  prepare.kernel_projection = call->weights->attention_conv_projection;
  prepare.input_bf16 = call->input_bf16;
  prepare.coefficients_bf16 = coefficients;
  prepare.output_bf16 = prepared;
  prepare.cublas_handle = call->cublas_handle;
  prepare.cuda_stream = call->cuda_stream;
  status = q27_dflash2_conv_prepare(&prepare);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_attention_args attention{};
  attention.struct_size = sizeof(attention);
  attention.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  attention.layer_index = call->layer_index;
  attention.weights = call->weights;
  attention.input_bf16 = prepared;
  attention.positions_u64 = call->positions_u64;
  attention.rope_inverse_frequencies_f32 =
      static_cast<const float*>(rope_frequencies);
  attention.rope_cache_f32 = static_cast<float*>(rope_cache);
  attention.q_bf16 = q;
  attention.k_bf16 = k;
  attention.v_bf16 = v;
  attention.context_bf16 = context;
  attention.output_bf16 = raw_output;
  attention.state = call->state;
  attention.workspace = flashinfer;
  attention.workspace_bytes = Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES;
  attention.rms_epsilon = 1.0e-6F;
  attention.cublas_handle = call->cublas_handle;
  attention.cuda_stream = call->cuda_stream;
  status = q27_dflash2_attention_forward(&attention);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_conv_finish_args finish{};
  finish.struct_size = sizeof(finish);
  finish.abi_version = Q27_DFLASH2_CONV_ABI_VERSION;
  finish.base_kernel = call->weights->attention_conv_base;
  finish.input_bf16 = raw_output;
  finish.coefficients_bf16 = coefficients;
  finish.output_bf16 = call->output_bf16;
  finish.cuda_stream = call->cuda_stream;
  return q27_dflash2_conv_finish(&finish);
}

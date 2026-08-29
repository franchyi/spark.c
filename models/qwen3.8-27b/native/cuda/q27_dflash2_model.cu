// SPDX-License-Identifier: Apache-2.0
// Fixed-shape raw-C translation of the pinned SGLang DFlash2 model semantics.

#include "q27_dflash2_model.h"
#include "q27_dflash2_attention.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;
constexpr float kDefaultEpsilon = 1.0e-6F;
thread_local std::string g_error;

q27_dflash2_status Ok() { return {Q27_DFLASH2_OK, "ok"}; }

q27_dflash2_status Invalid(const char* message) {
  return {Q27_DFLASH2_INVALID_ARGUMENT, message};
}

q27_dflash2_status Unimplemented(const char* message) {
  return {Q27_DFLASH2_UNIMPLEMENTED, message};
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
  if (args == nullptr) return Invalid("DFlash2 model arguments must be non-null");
  if (args->struct_size < sizeof(Args)) {
    return Invalid("DFlash2 model argument struct_size is too small");
  }
  if (args->abi_version != Q27_DFLASH2_MODEL_ABI_VERSION) {
    return Invalid("DFlash2 model ABI version mismatch");
  }
  return Ok();
}

__device__ float WarpSum(float value) {
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

/* Standard RMSNorm: output = gamma * value * rsqrt(mean(value^2)+eps). */
__global__ void RmsNormRows(const __nv_bfloat16* input,
                            const __nv_bfloat16* residual,
                            const __nv_bfloat16* gamma,
                            __nv_bfloat16* output,
                            __nv_bfloat16* residual_output,
                            uint32_t columns, float epsilon) {
  const uint32_t row = blockIdx.x;
  const uint32_t thread = threadIdx.x;
  const uint64_t base = static_cast<uint64_t>(row) * columns;
  float square_sum = 0.0F;

  for (uint32_t column = thread; column < columns; column += blockDim.x) {
    float value = __bfloat162float(input[base + column]);
    if (residual != nullptr) {
      value += __bfloat162float(residual[base + column]);
    }
    const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
    if (residual_output != nullptr) residual_output[base + column] = rounded;
    const float norm_input = __bfloat162float(rounded);
    square_sum += norm_input * norm_input;
  }

  square_sum = WarpSum(square_sum);
  __shared__ float warp_sums[kWarps];
  const uint32_t lane = thread & 31U;
  const uint32_t warp = thread >> 5U;
  if (lane == 0) warp_sums[warp] = square_sum;
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < kWarps ? warp_sums[lane] : 0.0F;
    block_sum = WarpSum(block_sum);
    if (lane == 0) warp_sums[0] = block_sum;
  }
  __syncthreads();
  const float inverse_rms =
      rsqrtf(warp_sums[0] / static_cast<float>(columns) + epsilon);

  for (uint32_t column = thread; column < columns; column += blockDim.x) {
    const __nv_bfloat16 rounded =
        residual_output != nullptr
            ? residual_output[base + column]
            : __float2bfloat16_rn(__bfloat162float(input[base + column]));
    const float value = __bfloat162float(rounded) * inverse_rms *
                        __bfloat162float(gamma[column]);
    output[base + column] = __float2bfloat16_rn(value);
  }
}

/* One block scores one (batch, edge, predecessor-index, candidate-index). */
__global__ void ScoreSelector(
    const uint32_t* candidate_ids, const uint32_t* anchor_tokens,
    const float* unary_logits, const __nv_bfloat16* projected_hidden,
    const __nv_bfloat16* predecessor_codebook,
    const __nv_bfloat16* successor_codebook, float* scores,
    uint32_t* invalid_id_count) {
  const uint32_t score_index = blockIdx.x;
  const uint32_t candidate = score_index % Q27_DFLASH2_SELECTOR_TOP_K;
  const uint32_t predecessor_index =
      (score_index / Q27_DFLASH2_SELECTOR_TOP_K) %
      Q27_DFLASH2_SELECTOR_TOP_K;
  const uint32_t edge =
      (score_index /
       (Q27_DFLASH2_SELECTOR_TOP_K * Q27_DFLASH2_SELECTOR_TOP_K)) %
      Q27_DFLASH2_DRAFT_TOKENS;
  const uint32_t batch =
      score_index /
      (Q27_DFLASH2_DRAFT_TOKENS * Q27_DFLASH2_SELECTOR_TOP_K *
       Q27_DFLASH2_SELECTOR_TOP_K);
  const uint64_t edge_base =
      (static_cast<uint64_t>(batch) * Q27_DFLASH2_DRAFT_TOKENS + edge) *
      Q27_DFLASH2_SELECTOR_TOP_K;
  const uint32_t successor_id = candidate_ids[edge_base + candidate];
  const uint32_t predecessor_id =
      edge == 0
          ? anchor_tokens[batch]
          : candidate_ids[edge_base - Q27_DFLASH2_SELECTOR_TOP_K +
                          predecessor_index];
  if (predecessor_id >= Q27_DFLASH2_VOCAB_SIZE ||
      successor_id >= Q27_DFLASH2_VOCAB_SIZE) {
    if (threadIdx.x == 0) {
      atomicAdd(invalid_id_count, 1U);
      scores[score_index] = -CUDART_INF_F;
    }
    return;
  }
  const uint64_t hidden_base =
      (static_cast<uint64_t>(batch) * Q27_DFLASH2_DRAFT_TOKENS + edge) *
      Q27_DFLASH2_SELECTOR_RANK;
  const uint64_t predecessor_base =
      static_cast<uint64_t>(predecessor_id) * Q27_DFLASH2_SELECTOR_RANK;
  const uint64_t successor_base =
      static_cast<uint64_t>(successor_id) * Q27_DFLASH2_SELECTOR_RANK;

  float partial = 0.0F;
  for (uint32_t rank = threadIdx.x; rank < Q27_DFLASH2_SELECTOR_RANK;
       rank += blockDim.x) {
    /* PyTorch's BF16 elementwise predecessor*hidden rounds before einsum. */
    const float weighted_predecessor =
        __bfloat162float(predecessor_codebook[predecessor_base + rank]) *
        __bfloat162float(projected_hidden[hidden_base + rank]);
    const __nv_bfloat16 rounded_weighted =
        __float2bfloat16_rn(weighted_predecessor);
    partial += __bfloat162float(rounded_weighted) *
               __bfloat162float(successor_codebook[successor_base + rank]);
  }

  partial = WarpSum(partial);
  __shared__ float warp_sums[kWarps];
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? warp_sums[lane] : 0.0F;
    total = WarpSum(total);
    if (lane == 0) warp_sums[0] = total;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    /* torch.einsum(BF16,BF16) produces BF16 before FP32 unary promotion. */
    const float pair_score =
        __bfloat162float(__float2bfloat16_rn(warp_sums[0]));
    scores[score_index] = unary_logits[edge_base + candidate] + pair_score;
  }
}

q27_dflash2_status LaunchRmsNorm(const void* input, const void* residual,
                                 const void* gamma, void* output,
                                 void* residual_output, uint32_t rows,
                                 uint32_t columns, float epsilon,
                                 cudaStream_t stream) {
  if (rows == 0 || columns == 0 || input == nullptr || gamma == nullptr ||
      output == nullptr || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Invalid("invalid DFlash2 RMSNorm arguments");
  }
  RmsNormRows<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gamma),
      static_cast<__nv_bfloat16*>(output),
      static_cast<__nv_bfloat16*>(residual_output), columns, epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("DFlash2 RMSNorm launch", error);
}

q27_dflash2_status ProjectRows(cublasHandle_t handle, cudaStream_t stream,
                               const void* input_bf16,
                               const void* weight_bf16, void* output_bf16,
                               uint32_t rows, uint32_t input_columns,
                               uint32_t output_columns, const char* operation) {
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("DFlash2 cuBLAS set stream", status);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  /* Row-major C=A*W^T via column-major C^T=W*A^T. */
  status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns, rows, input_columns,
      &alpha, weight_bf16, CUDA_R_16BF, input_columns, input_bf16,
      CUDA_R_16BF, input_columns, &beta, output_bf16, CUDA_R_16BF,
      output_columns, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? Ok() : CublasError(operation, status);
}

bool DistinctForwardBuffers(const q27_dflash2_forward_args* args) {
  const void* input = args->input_embeddings_bf16;
  const void* normalized = args->normalized_bf16;
  const void* residual = args->residual_bf16;
  const void* sublayer = args->sublayer_output_bf16;
  const void* final_hidden = args->final_hidden_bf16;
  if (input == normalized || input == residual || input == sublayer ||
      residual == normalized || residual == sublayer ||
      normalized == sublayer) {
    return false;
  }
  return final_hidden == normalized ||
         (final_hidden != input && final_hidden != residual &&
          final_hidden != sublayer);
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_project_context(
    const q27_dflash2_context_projection_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->weights == nullptr || args->target_features_bf16 == nullptr ||
      args->scratch_bf16 == nullptr || args->context_hidden_bf16 == nullptr ||
      args->cublas_handle == nullptr || args->token_count == 0 ||
      args->token_count > 262144 ||
      !std::isfinite(args->rms_epsilon) || args->rms_epsilon < 0.0F ||
      args->scratch_bf16 == args->target_features_bf16 ||
      args->context_hidden_bf16 == args->target_features_bf16 ||
      args->context_hidden_bf16 == args->scratch_bf16) {
    return Invalid("invalid DFlash2 context projection arguments");
  }
  const q27_dflash2_weights& weights = *args->weights;
  if (weights.context_projection.data == nullptr ||
      weights.context_norm.data == nullptr) {
    return Invalid("DFlash2 context projection weights are null");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  status = ProjectRows(
      static_cast<cublasHandle_t>(args->cublas_handle), stream,
      args->target_features_bf16, weights.context_projection.data,
      args->scratch_bf16, args->token_count,
      Q27_DFLASH2_TARGET_FEATURES * Q27_DFLASH2_HIDDEN_SIZE,
      Q27_DFLASH2_HIDDEN_SIZE, "DFlash2 context projection");
  if (status.code != Q27_DFLASH2_OK) return status;
  const float epsilon = args->rms_epsilon > 0.0F ? args->rms_epsilon
                                                 : kDefaultEpsilon;
  return LaunchRmsNorm(args->scratch_bf16, nullptr, weights.context_norm.data,
                       args->context_hidden_bf16, nullptr, args->token_count,
                       Q27_DFLASH2_HIDDEN_SIZE, epsilon, stream);
}

extern "C" q27_dflash2_status q27_dflash2_project_selector_hidden(
    const q27_dflash2_selector_projection_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->weights == nullptr || args->hidden_bf16 == nullptr ||
      args->projected_hidden_bf16 == nullptr || args->cublas_handle == nullptr ||
      args->token_count == 0 ||
      args->token_count > Q27_DFLASH2_MAX_BATCH * Q27_DFLASH2_DRAFT_TOKENS ||
      args->hidden_bf16 == args->projected_hidden_bf16 ||
      args->weights->selector_hidden_projection.data == nullptr) {
    return Invalid("invalid DFlash2 selector projection arguments");
  }
  return ProjectRows(
      static_cast<cublasHandle_t>(args->cublas_handle),
      static_cast<cudaStream_t>(args->cuda_stream), args->hidden_bf16,
      args->weights->selector_hidden_projection.data,
      args->projected_hidden_bf16, args->token_count, Q27_DFLASH2_HIDDEN_SIZE,
      Q27_DFLASH2_SELECTOR_RANK, "DFlash2 selector hidden projection");
}

extern "C" q27_dflash2_status q27_dflash2_score_selector(
    const q27_dflash2_selector_score_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->weights == nullptr || args->candidate_ids == nullptr ||
      args->anchor_tokens == nullptr || args->unary_logits == nullptr ||
      args->projected_hidden_bf16 == nullptr || args->scores == nullptr ||
      args->invalid_id_count_u32 == nullptr ||
      args->batch_size == 0 || args->batch_size > Q27_DFLASH2_MAX_BATCH ||
      args->weights->selector_predecessor_codebook.data == nullptr ||
      args->weights->selector_successor_codebook.data == nullptr) {
    return Invalid("invalid DFlash2 selector score arguments");
  }
  const uint32_t blocks =
      args->batch_size * Q27_DFLASH2_DRAFT_TOKENS *
      Q27_DFLASH2_SELECTOR_TOP_K * Q27_DFLASH2_SELECTOR_TOP_K;
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cudaError_t error = cudaMemsetAsync(args->invalid_id_count_u32, 0,
                                      sizeof(*args->invalid_id_count_u32), stream);
  if (error != cudaSuccess) {
    return CudaError("DFlash2 selector id counter clear", error);
  }
  ScoreSelector<<<blocks, kThreads, 0,
                  stream>>>(
      args->candidate_ids, args->anchor_tokens, args->unary_logits,
      static_cast<const __nv_bfloat16*>(args->projected_hidden_bf16),
      static_cast<const __nv_bfloat16*>(
          args->weights->selector_predecessor_codebook.data),
      static_cast<const __nv_bfloat16*>(
          args->weights->selector_successor_codebook.data),
      args->scores, args->invalid_id_count_u32);
  error = cudaGetLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("DFlash2 selector score launch", error);
}

extern "C" q27_dflash2_status q27_dflash2_forward(
    const q27_dflash2_forward_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->mlp == nullptr) {
    return Unimplemented(
        "DFlash2 forward requires the fixed MLP dependency");
  }
  if (args->weights == nullptr || args->input_embeddings_bf16 == nullptr ||
      args->positions_u64 == nullptr || args->normalized_bf16 == nullptr ||
      args->residual_bf16 == nullptr || args->sublayer_output_bf16 == nullptr ||
      args->final_hidden_bf16 == nullptr || args->state == nullptr ||
      args->cublas_handle == nullptr || args->batch_size == 0 ||
      args->batch_size > Q27_DFLASH2_MAX_BATCH ||
      !std::isfinite(args->rms_epsilon) || args->rms_epsilon < 0.0F ||
      !DistinctForwardBuffers(args)) {
    return Invalid("invalid DFlash2 forward arguments");
  }
  const uint32_t tokens = args->batch_size * Q27_DFLASH2_BLOCK_SIZE;
  const float epsilon = args->rms_epsilon > 0.0F ? args->rms_epsilon
                                                 : kDefaultEpsilon;
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const void* norm_input = args->input_embeddings_bf16;
  const void* norm_residual = nullptr;

  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    const q27_dflash2_layer_weights& weights = args->weights->layers[layer];
    if (weights.input_norm.data == nullptr ||
        weights.post_attention_norm.data == nullptr) {
      return Invalid("DFlash2 forward norm weight is null");
    }
    status = LaunchRmsNorm(norm_input, norm_residual, weights.input_norm.data,
                           args->normalized_bf16, args->residual_bf16, tokens,
                           Q27_DFLASH2_HIDDEN_SIZE, epsilon, stream);
    if (status.code != Q27_DFLASH2_OK) return status;

    q27_dflash2_sublayer_call call{};
    call.struct_size = sizeof(call);
    call.abi_version = Q27_DFLASH2_MODEL_ABI_VERSION;
    call.layer_index = layer;
    call.batch_size = args->batch_size;
    call.token_count = tokens;
    call.weights = &weights;
    call.positions_u64 = args->positions_u64;
    call.input_bf16 = args->normalized_bf16;
    call.output_bf16 = args->sublayer_output_bf16;
    call.state = args->state;
    call.workspace = args->workspace;
    call.workspace_bytes = args->workspace_bytes;
    call.cublas_handle = args->cublas_handle;
    call.cuda_stream = args->cuda_stream;
    status = q27_dflash2_attention_sublayer(&call, nullptr);
    if (status.code != Q27_DFLASH2_OK) return status;

    status = LaunchRmsNorm(
        args->sublayer_output_bf16, args->residual_bf16,
        weights.post_attention_norm.data, args->normalized_bf16,
        args->residual_bf16, tokens, Q27_DFLASH2_HIDDEN_SIZE, epsilon, stream);
    if (status.code != Q27_DFLASH2_OK) return status;

    call.input_bf16 = args->normalized_bf16;
    call.output_bf16 = args->sublayer_output_bf16;
    status = args->mlp(&call, args->mlp_user_data);
    if (status.code != Q27_DFLASH2_OK) return status;

    norm_input = args->sublayer_output_bf16;
    norm_residual = args->residual_bf16;
  }

  if (args->weights->final_norm.data == nullptr) {
    return Invalid("DFlash2 final norm weight is null");
  }
  return LaunchRmsNorm(norm_input, norm_residual,
                       args->weights->final_norm.data,
                       args->final_hidden_bf16, args->residual_bf16, tokens,
                       Q27_DFLASH2_HIDDEN_SIZE, epsilon, stream);
}

// SPDX-License-Identifier: Apache-2.0
//
// Fixed batch-one DFlash2 target-head/top-16 reference. Candidate semantics
// follow SGLang c14312a66420b75ca9a11bf1817c4db1fa26b097 (Apache-2.0),
// DFlashDraftModel.compute_candidates. The deterministic reduction is a small
// raw CUDA replacement for the framework-level FlashInfer top_k call.

#include "q27_dflash2_topk.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kTopK = Q27_DFLASH2_TOPK_K;
constexpr uint32_t kSharedCandidates = kThreads * kTopK;
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

bool IsAligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool RangesOverlap(const void* left, uint64_t left_bytes, const void* right,
                   uint64_t right_bytes) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
  if (left_bytes > UINTPTR_MAX - left_begin ||
      right_bytes > UINTPTR_MAX - right_begin)
    return true;
  const uintptr_t left_end = left_begin + left_bytes;
  const uintptr_t right_end = right_begin + right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

__device__ __forceinline__ bool Better(float value, uint32_t token,
                                       float incumbent_value,
                                       uint32_t incumbent_token) {
  return value > incumbent_value ||
         (value == incumbent_value && token < incumbent_token);
}

__device__ __forceinline__ void Insert(float value, uint32_t token,
                                       float* values, uint32_t* tokens) {
  if (!Better(value, token, values[kTopK - 1], tokens[kTopK - 1])) return;
  uint32_t position = kTopK - 1;
  while (position != 0 &&
         Better(value, token, values[position - 1], tokens[position - 1])) {
    values[position] = values[position - 1];
    tokens[position] = tokens[position - 1];
    --position;
  }
  values[position] = value;
  tokens[position] = token;
}

__global__ void StableTop16(const float* logits, uint32_t* candidate_ids,
                            float* unary_logits) {
  const uint32_t row = blockIdx.x;
  float local_values[kTopK];
  uint32_t local_tokens[kTopK];
#pragma unroll
  for (uint32_t rank = 0; rank < kTopK; ++rank) {
    local_values[rank] = -CUDART_INF_F;
    local_tokens[rank] = UINT32_MAX;
  }

  const uint64_t row_base =
      static_cast<uint64_t>(row) * Q27_DFLASH2_VOCAB_SIZE;
  for (uint32_t token = threadIdx.x; token < Q27_DFLASH2_VOCAB_SIZE;
       token += blockDim.x) {
    float value = logits[row_base + token];
    if (isnan(value)) value = -CUDART_INF_F;
    /* torch.matmul(BF16,BF16) returns BF16 before .float() in SGLang. */
    value = __bfloat162float(__float2bfloat16_rn(value));
    Insert(value, token, local_values, local_tokens);
  }

  __shared__ float shared_values[kSharedCandidates];
  __shared__ uint32_t shared_tokens[kSharedCandidates];
  const uint32_t shared_base = threadIdx.x * kTopK;
#pragma unroll
  for (uint32_t rank = 0; rank < kTopK; ++rank) {
    shared_values[shared_base + rank] = local_values[rank];
    shared_tokens[shared_base + rank] = local_tokens[rank];
  }
  __syncthreads();

  /* Correctness reference: one deterministic merge after the parallel scan. */
  if (threadIdx.x == 0) {
    float best_values[kTopK];
    uint32_t best_tokens[kTopK];
#pragma unroll
    for (uint32_t rank = 0; rank < kTopK; ++rank) {
      best_values[rank] = -CUDART_INF_F;
      best_tokens[rank] = UINT32_MAX;
    }
    for (uint32_t thread = 0; thread < kThreads; ++thread) {
#pragma unroll
      for (uint32_t rank = 0; rank < kTopK; ++rank) {
        const uint32_t index = thread * kTopK + rank;
        Insert(shared_values[index], shared_tokens[index], best_values,
               best_tokens);
      }
    }
    const uint32_t output_base = row * kTopK;
#pragma unroll
    for (uint32_t rank = 0; rank < kTopK; ++rank) {
      candidate_ids[output_base + rank] = best_tokens[rank];
      unary_logits[output_base + rank] = best_values[rank];
    }
  }
}

q27_dflash2_status ValidateTopk(
    const q27_dflash2_topk_from_logits_args* args) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_DFLASH2_TOPK_ABI_VERSION)
    return Invalid("DFlash2 top-16 arguments or ABI are invalid");
  constexpr uint64_t output_elements =
      static_cast<uint64_t>(Q27_DFLASH2_TOPK_ROWS) * Q27_DFLASH2_TOPK_K;
  constexpr uint64_t output_bytes = output_elements * sizeof(uint32_t);
  if (!IsAligned(args->logits_f32, alignof(float)) ||
      args->logits_elements != Q27_DFLASH2_TOPK_LOGIT_ELEMENTS ||
      !IsAligned(args->candidate_ids_u32, alignof(uint32_t)) ||
      !IsAligned(args->unary_logits_f32, alignof(float)) ||
      RangesOverlap(args->logits_f32, Q27_DFLASH2_TOPK_LOGIT_BYTES,
                    args->candidate_ids_u32, output_bytes) ||
      RangesOverlap(args->logits_f32, Q27_DFLASH2_TOPK_LOGIT_BYTES,
                    args->unary_logits_f32, output_bytes) ||
      RangesOverlap(args->candidate_ids_u32, output_bytes,
                    args->unary_logits_f32, output_bytes))
    return Invalid("DFlash2 top-16 pointer, size, alignment, or alias is invalid");
  return Ok();
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_topk_from_logits(
    const q27_dflash2_topk_from_logits_args* args) {
  q27_dflash2_status status = ValidateTopk(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  StableTop16<<<Q27_DFLASH2_TOPK_ROWS, kThreads, 0,
                static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->logits_f32, args->candidate_ids_u32, args->unary_logits_f32);
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("DFlash2 stable top-16 launch", error);
}

extern "C" q27_dflash2_status q27_dflash2_lm_head_topk(
    const q27_dflash2_lm_head_topk_args* args) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_DFLASH2_TOPK_ABI_VERSION)
    return Invalid("DFlash2 LM-head/top-16 arguments or ABI are invalid");
  constexpr uint64_t hidden_bytes =
      static_cast<uint64_t>(Q27_DFLASH2_TOPK_HIDDEN_ELEMENTS) * 2ULL;
  constexpr uint64_t output_elements =
      static_cast<uint64_t>(Q27_DFLASH2_TOPK_ROWS) * Q27_DFLASH2_TOPK_K;
  constexpr uint64_t output_bytes = output_elements * sizeof(uint32_t);
  if (!IsAligned(args->hidden_bf16, alignof(__nv_bfloat16)) ||
      !IsAligned(args->lm_head_weight_bf16, alignof(__nv_bfloat16)) ||
      args->lm_head_weight_bytes != Q27_DFLASH2_TOPK_LM_HEAD_BYTES ||
      !IsAligned(args->logits_f32, alignof(float)) ||
      args->logits_elements != Q27_DFLASH2_TOPK_LOGIT_ELEMENTS ||
      !IsAligned(args->candidate_ids_u32, alignof(uint32_t)) ||
      !IsAligned(args->unary_logits_f32, alignof(float)) ||
      args->cublas_handle == nullptr ||
      RangesOverlap(args->hidden_bf16, hidden_bytes, args->logits_f32,
                    Q27_DFLASH2_TOPK_LOGIT_BYTES) ||
      RangesOverlap(args->lm_head_weight_bf16,
                    Q27_DFLASH2_TOPK_LM_HEAD_BYTES, args->logits_f32,
                    Q27_DFLASH2_TOPK_LOGIT_BYTES) ||
      RangesOverlap(args->logits_f32, Q27_DFLASH2_TOPK_LOGIT_BYTES,
                    args->candidate_ids_u32, output_bytes) ||
      RangesOverlap(args->logits_f32, Q27_DFLASH2_TOPK_LOGIT_BYTES,
                    args->unary_logits_f32, output_bytes) ||
      RangesOverlap(args->hidden_bf16, hidden_bytes, args->candidate_ids_u32,
                    output_bytes) ||
      RangesOverlap(args->hidden_bf16, hidden_bytes, args->unary_logits_f32,
                    output_bytes) ||
      RangesOverlap(args->lm_head_weight_bf16,
                    Q27_DFLASH2_TOPK_LM_HEAD_BYTES,
                    args->candidate_ids_u32, output_bytes) ||
      RangesOverlap(args->lm_head_weight_bf16,
                    Q27_DFLASH2_TOPK_LM_HEAD_BYTES,
                    args->unary_logits_f32, output_bytes) ||
      RangesOverlap(args->candidate_ids_u32, output_bytes,
                    args->unary_logits_f32, output_bytes))
    return Invalid(
        "DFlash2 LM-head/top-16 pointer, size, alignment, or alias is invalid");

  cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS)
    return CublasError("DFlash2 LM-head set cuBLAS stream", cublas_status);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  /* Row-major logits = hidden * lm_head^T via column-major transpose view. */
  cublas_status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, Q27_DFLASH2_VOCAB_SIZE,
      Q27_DFLASH2_TOPK_ROWS, Q27_DFLASH2_HIDDEN_SIZE, &alpha,
      args->lm_head_weight_bf16, CUDA_R_16BF, Q27_DFLASH2_HIDDEN_SIZE,
      args->hidden_bf16, CUDA_R_16BF, Q27_DFLASH2_HIDDEN_SIZE, &beta,
      args->logits_f32, CUDA_R_32F, Q27_DFLASH2_VOCAB_SIZE,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (cublas_status != CUBLAS_STATUS_SUCCESS)
    return CublasError("DFlash2 BF16 target LM-head", cublas_status);

  q27_dflash2_topk_from_logits_args topk = {};
  topk.struct_size = sizeof(topk);
  topk.abi_version = Q27_DFLASH2_TOPK_ABI_VERSION;
  topk.logits_f32 = args->logits_f32;
  topk.logits_elements = args->logits_elements;
  topk.candidate_ids_u32 = args->candidate_ids_u32;
  topk.unary_logits_f32 = args->unary_logits_f32;
  topk.cuda_stream = args->cuda_stream;
  return q27_dflash2_topk_from_logits(&topk);
}

// The softmax/top-k kernel is adapted from SGLang's Apache-2.0
// moe_topk_softmax.cuh at commit d91c3682b0b429e4c70df63cd57f819588ce29b0
// (source SHA-256 f9c8ee1f1e9af1037612418cda472b907c6455262c93a5d1e20764cf065fb55a).
// SGLang in turn credits vLLM v0.7.3 and TensorRT-LLM v0.7.1. This adapter
// retains the 512-expert warp arithmetic and tie-breaking, and removes
// TVM-FFI, sgl-kernel tensor wrappers, generic dispatch, and allocation.

#include "internal/moe_gate_backend.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kElementsPerLoad = 8;
constexpr int kValuesPerThread = 16;
constexpr int kLoadsPerThread = 2;
constexpr int kThreadsPerRow = 32;
constexpr int kRowsPerBlock = 4;

thread_local std::string g_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }

FlashStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

const char* CublasErrorName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "not supported";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "license error";
    default:
      return "unknown error";
  }
}

FlashStatus CublasError(const char* prefix, cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append(CublasErrorName(status));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
  T data[N];
};

__global__ __launch_bounds__(kWarpsPerBlock * kWarpSize)
void QwenMoeTopk512x10(const __nv_bfloat16* input, float* output,
                       int32_t* indices, int num_rows) {
  const int cta_base_row = static_cast<int>(blockIdx.x) * kRowsPerBlock;
  const int thread_row = cta_base_row + static_cast<int>(threadIdx.y);
  if (thread_row >= num_rows) return;

  const int lane = static_cast<int>(threadIdx.x);
  const __nv_bfloat16* row = input + thread_row * kExperts;
  const int first_element = lane * kElementsPerLoad;
  const __nv_bfloat16* read = row + first_element;

  using Access = AlignedArray<__nv_bfloat16, kElementsPerLoad>;
  __nv_bfloat16 temporary[kValuesPerThread];
  Access* local_vectors = reinterpret_cast<Access*>(temporary);
  const Access* row_vectors = reinterpret_cast<const Access*>(read);
#pragma unroll
  for (int load = 0; load < kLoadsPerThread; ++load) {
    local_vectors[load] = row_vectors[load * kThreadsPerRow];
  }

  float values[kValuesPerThread];
#pragma unroll
  for (int element = 0; element < kValuesPerThread; ++element) {
    values[element] = __bfloat162float(temporary[element]);
  }

  float row_max = values[0];
#pragma unroll
  for (int element = 1; element < kValuesPerThread; ++element) {
    row_max = max(row_max, values[element]);
  }
#pragma unroll
  for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
    row_max = max(row_max,
                  __shfl_xor_sync(0xffffffffU, row_max, mask, kThreadsPerRow));
  }

  float row_sum = 0.0F;
#pragma unroll
  for (int element = 0; element < kValuesPerThread; ++element) {
    values[element] = expf(values[element] - row_max);
    row_sum += values[element];
  }
#pragma unroll
  for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
    row_sum +=
        __shfl_xor_sync(0xffffffffU, row_sum, mask, kThreadsPerRow);
  }
  const float reciprocal_sum = 1.0F / row_sum;
#pragma unroll
  for (int element = 0; element < kValuesPerThread; ++element) {
    values[element] *= reciprocal_sum;
  }

  float selected_sum = 0.0F;
#pragma unroll
  for (int rank = 0; rank < kTopK; ++rank) {
    float maximum = values[0];
    int expert = first_element;
#pragma unroll
    for (int load = 0, column = first_element; load < kLoadsPerThread;
         ++load, column += kElementsPerLoad * kThreadsPerRow) {
#pragma unroll
      for (int element = 0; element < kElementsPerLoad; ++element) {
        const float candidate = values[load * kElementsPerLoad + element];
        if (candidate > maximum) {
          maximum = candidate;
          expert = column + element;
        }
      }
    }
#pragma unroll
    for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
      const float other_max =
          __shfl_xor_sync(0xffffffffU, maximum, mask, kThreadsPerRow);
      const int other_expert =
          __shfl_xor_sync(0xffffffffU, expert, mask, kThreadsPerRow);
      if (other_max > maximum ||
          (other_max == maximum && other_expert < expert)) {
        maximum = other_max;
        expert = other_expert;
      }
    }

    if (lane == 0) {
      const int output_index = thread_row * kTopK + rank;
      output[output_index] = maximum;
      indices[output_index] = expert;
      selected_sum += maximum;
    }

    if (rank + 1 < kTopK) {
      const int load_for_expert =
          expert / (kElementsPerLoad * kThreadsPerRow);
      const int lane_for_expert =
          (expert / kElementsPerLoad) % kThreadsPerRow;
      if (lane == lane_for_expert) {
        values[load_for_expert * kElementsPerLoad +
               expert % kElementsPerLoad] = -10000.0F;
      }
    }
  }

  if (lane == 0) {
    const float reciprocal_selected_sum = 1.0F / selected_sum;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
      output[thread_row * kTopK + rank] *= reciprocal_selected_sum;
    }
  }
}

}  // namespace

FlashStatus flash_sglang_cublas_moe_gate_cuda_launch(
    const FlashMoeGateArgs* args) {
  cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS router stream bind failed: ", status);
  }
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS router math-mode selection failed: ", status);
  }

  const int m = static_cast<int>(args->plan.num_experts);
  const int n = static_cast<int>(args->plan.num_tokens);
  const int k = static_cast<int>(args->plan.hidden_size);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  // Row-major Y[M,N] = X[M,K] * W[N,K]^T is the column-major operation
  // Y_col[N,M] = W_col[K,N]^T * X_col[K,M].
  status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
      args->router_weight, CUDA_R_16BF, k, args->hidden_states, CUDA_R_16BF,
      k, &beta, args->router_logits, CUDA_R_16BF, m, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS BF16 router projection failed: ", status);
  }

  const int blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
  const dim3 threads(kWarpSize, kWarpsPerBlock);
  QwenMoeTopk512x10<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->router_logits),
      args->topk_weights, args->topk_ids, n);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("SGLang MoE top-k launch failed: ", error);
  }
  return Ok();
}

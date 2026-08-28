// Qwen mHC arithmetic adapted from SGLang at commit
// d91c3682b0b429e4c70df63cd57f819588ce29b0:
// grouped_gemma_rmsnorm.cuh (group-local normalization) and hc_combine.cuh
// (FP32 gate reduction plus residual injection). Low-rank projections remain
// cuBLAS BF16. The persistent atomic Triton mix is deliberately not copied;
// this path follows SGLang's deterministic reference reduction.

#include "internal/mhc_backend.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kHc = 4;
constexpr int kHidden = 2560;
constexpr int kWidth = kHc * kHidden;
constexpr int kLowrank = 320;
constexpr int kNormThreads = kHidden / 16;
constexpr int kCombineThreads = 256;
constexpr int kElementThreads = 256;
constexpr int kVectorElements = 8;
thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

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
    default:
      return "unknown error";
  }
}

SparkServeStatus CublasError(const char* prefix, cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append(CublasErrorName(status));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

cublasStatus_t RowMajorBf16Linear(cublasHandle_t handle, int tokens,
                                  int output_columns, int input_columns,
                                  const void* input, const void* weight,
                                  void* output) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns, tokens,
      input_columns, &alpha, weight, CUDA_R_16BF, input_columns, input,
      CUDA_R_16BF, input_columns, &beta, output, CUDA_R_16BF, output_columns,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__device__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset /= 2) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

__global__ __launch_bounds__(kNormThreads) void SglangGroupedGemmaRmsNorm(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, float eps) {
  const int group = static_cast<int>(blockIdx.x);
  const int begin = group * kHidden;
  const int element_begin = static_cast<int>(threadIdx.x) * 16;
  float values[16];
  float sum = 0.0F;
#pragma unroll
  for (int element = 0; element < 16; ++element) {
    const float value =
        __bfloat162float(input[begin + element_begin + element]);
    values[element] = value;
    sum += value * value;
  }
  sum = WarpReduceSum(sum);
  __shared__ float warp_sums[5];
  __shared__ float norm;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < 5 ? warp_sums[lane] : 0.0F;
    total = WarpReduceSum(total);
    if (lane == 0) norm = rsqrtf(total / kHidden + eps);
  }
  __syncthreads();
#pragma unroll
  for (int element = 0; element < 16; ++element) {
    const int index = begin + element_begin + element;
    const float scale = 1.0F + __bfloat162float(weight[index % kWidth]);
    output[index] = __float2bfloat16_rn(values[element] * norm * scale);
  }
}

__global__ void ScaleByHc(const __nv_bfloat16* input,
                          __nv_bfloat16* output, int elements) {
  const int element = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  output[element] =
      __float2bfloat16_rn(__bfloat162float(input[element]) * 0.25F);
}

__global__ void SglangSiluInPlace(__nv_bfloat16* values, int elements) {
  const int element = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  const float value = __bfloat162float(values[element]);
  values[element] = __float2bfloat16_rn(value / (1.0F + expf(-value)));
}

__global__ void DeterministicMixMean(const __nv_bfloat16* normed,
                                     const __nv_bfloat16* gates,
                                     __nv_bfloat16* output, int tokens) {
  const int item = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int items = tokens * kHidden;
  if (item >= items) return;
  const int token = item / kHidden;
  const int column = item % kHidden;
  float sum = 0.0F;
#pragma unroll
  for (int branch = 0; branch < kHc; ++branch) {
    const int index = token * kWidth + branch * kHidden + column;
    const float gate_value = __bfloat162float(gates[index]);
    const float gate = 1.0F / (1.0F + expf(-gate_value));
    sum += gate * __bfloat162float(normed[index]);
  }
  output[item] = __float2bfloat16_rn(sum * 0.25F);
}

__global__ __launch_bounds__(kCombineThreads) void SglangHcCombine(
    const __nv_bfloat16* block_output, const __nv_bfloat16* residual,
    const __nv_bfloat16* normed_residual,
    const __nv_bfloat16* inject_weight, __nv_bfloat16* output) {
  const int token = static_cast<int>(blockIdx.x);
  const int row_begin = token * kWidth;
  float partial[kHc] = {};
#pragma unroll
  for (int chunk = 0; chunk < 5; ++chunk) {
    const int vector = static_cast<int>(threadIdx.x) + chunk * kCombineThreads;
    const int begin = row_begin + vector * kVectorElements;
#pragma unroll
    for (int branch = 0; branch < kHc; ++branch) {
      const int weight_begin = branch * kWidth + vector * kVectorElements;
#pragma unroll
      for (int pair = 0; pair < kVectorElements; pair += 2) {
        const float nx = __bfloat162float(normed_residual[begin + pair]);
        const float ny = __bfloat162float(normed_residual[begin + pair + 1]);
        const float wx = __bfloat162float(inject_weight[weight_begin + pair]);
        const float wy =
            __bfloat162float(inject_weight[weight_begin + pair + 1]);
        partial[branch] += nx * wx + ny * wy;
      }
    }
  }
#pragma unroll
  for (int branch = 0; branch < kHc; ++branch) {
    partial[branch] = WarpReduceSum(partial[branch]);
  }
  __shared__ float warp_sums[kHc][8];
  __shared__ float injection[kHc];
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  if (lane == 0) {
#pragma unroll
    for (int branch = 0; branch < kHc; ++branch) {
      warp_sums[branch][warp] = partial[branch];
    }
  }
  __syncthreads();
  if (threadIdx.x < kHc) {
    float total = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < 8; ++source_warp) {
      total += warp_sums[threadIdx.x][source_warp];
    }
    injection[threadIdx.x] = 2.0F / (1.0F + expf(-total * 0.25F));
  }
  __syncthreads();

#pragma unroll
  for (int chunk = 0; chunk < 5; ++chunk) {
    const int vector = static_cast<int>(threadIdx.x) + chunk * kCombineThreads;
    const int branch = vector / (kHidden / kVectorElements);
    const int column = (vector % (kHidden / kVectorElements)) * kVectorElements;
    const int residual_begin = row_begin + vector * kVectorElements;
#pragma unroll
    for (int element = 0; element < kVectorElements; ++element) {
      const float result =
          __bfloat162float(residual[residual_begin + element]) +
          injection[branch] *
              __bfloat162float(block_output[token * kHidden + column + element]);
      output[residual_begin + element] = __float2bfloat16_rn(result);
    }
  }
}

uint32_t Blocks(int elements) {
  return static_cast<uint32_t>((elements + kElementThreads - 1) /
                               kElementThreads);
}

}  // namespace

SparkServeStatus sparkserve_sglang_cublas_mhc_mix_cuda_launch(
    const SparkServeMhcArgs* args) {
  const int tokens = static_cast<int>(args->plan.num_tokens);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cublasStatus_t blas_status = cublasSetStream(handle, stream);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS mHC stream bind failed: ", blas_status);
  }
  blas_status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS mHC math-mode selection failed: ", blas_status);
  }

  SglangGroupedGemmaRmsNorm<<<tokens * kHc, kNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->hyper_input),
      static_cast<const __nv_bfloat16*>(args->norm_weight),
      static_cast<__nv_bfloat16*>(args->normed), args->plan.rms_norm_eps);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("mHC RMSNorm failed: ", error);

  blas_status = RowMajorBf16Linear(handle, tokens, kLowrank, kWidth,
                                   args->normed, args->mix_down_weight,
                                   args->mix_down);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS mHC down projection failed: ", blas_status);
  }
  const int lowrank_elements = tokens * kLowrank;
  ScaleByHc<<<Blocks(lowrank_elements), kElementThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->mix_down),
      static_cast<__nv_bfloat16*>(args->mix_activated), lowrank_elements);
  SglangSiluInPlace<<<Blocks(lowrank_elements), kElementThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(args->mix_activated), lowrank_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("mHC SiLU failed: ", error);

  blas_status = RowMajorBf16Linear(handle, tokens, kWidth, kLowrank,
                                   args->mix_activated, args->mix_up_weight,
                                   args->mix_up);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS mHC up projection failed: ", blas_status);
  }
  DeterministicMixMean<<<Blocks(tokens * kHidden), kElementThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->normed),
      static_cast<const __nv_bfloat16*>(args->mix_up),
      static_cast<__nv_bfloat16*>(args->mixed_output), tokens);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("mHC mix epilogue failed: ", error);
  return Ok();
}

SparkServeStatus sparkserve_sglang_cublas_mhc_combine_cuda_launch(
    const SparkServeMhcArgs* args) {
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  SglangHcCombine<<<args->plan.num_tokens, kCombineThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->block_output),
      static_cast<const __nv_bfloat16*>(args->hyper_input),
      static_cast<const __nv_bfloat16*>(args->normed),
      static_cast<const __nv_bfloat16*>(args->inject_weight),
      static_cast<__nv_bfloat16*>(args->combined_output));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("SGLang mHC combine failed: ", error);
  return Ok();
}

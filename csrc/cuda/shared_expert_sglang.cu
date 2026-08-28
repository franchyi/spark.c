// Shared-expert arithmetic adapted from SGLang at commit
// d91c3682b0b429e4c70df63cd57f819588ce29b0. The vectorized SiLU/multiply
// follows elementwise/activation.cuh (SHA-256 f1b56af7...), and the final
// sigmoid broadcast follows moe/triton_sigmoid_gate_mul.py (SHA-256
// 7c357dfb...). Framework tensor wrappers and allocation are removed.

#include "internal/shared_expert_backend.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kIntermediate = 640;
constexpr int kGateUp = 2 * kIntermediate;
constexpr int kVectorElements = 8;
constexpr int kThreads = 256;

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
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "license error";
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
                                  void* output, int output_row_stride) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns, tokens,
      input_columns, &alpha, weight, CUDA_R_16BF, input_columns, input,
      CUDA_R_16BF, input_columns, &beta, output, CUDA_R_16BF,
      output_row_stride, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__global__ void SglangSiluAndMul(const __nv_bfloat16* gate_up,
                                 __nv_bfloat16* activated, int tokens) {
  const int vectors_per_row = kIntermediate / kVectorElements;
  const int vector = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total_vectors = tokens * vectors_per_row;
  if (vector >= total_vectors) return;
  const int token = vector / vectors_per_row;
  const int offset = (vector % vectors_per_row) * kVectorElements;
  const auto* gate_vector = reinterpret_cast<const uint4*>(
      gate_up + token * kGateUp + offset);
  const auto* up_vector = reinterpret_cast<const uint4*>(
      gate_up + token * kGateUp + kIntermediate + offset);
  const uint4 gate_bits = *gate_vector;
  const uint4 up_bits = *up_vector;
  const auto* gate = reinterpret_cast<const __nv_bfloat16*>(&gate_bits);
  const auto* up = reinterpret_cast<const __nv_bfloat16*>(&up_bits);
  uint4 output_bits;
  auto* output = reinterpret_cast<__nv_bfloat16*>(&output_bits);
#pragma unroll
  for (int element = 0; element < kVectorElements; ++element) {
    const float gate_f32 = __bfloat162float(gate[element]);
    const float up_f32 = __bfloat162float(up[element]);
    const float silu = gate_f32 / (1.0F + expf(-gate_f32));
    output[element] = __float2bfloat16_rn(silu * up_f32);
  }
  *reinterpret_cast<uint4*>(activated + token * kIntermediate + offset) =
      output_bits;
}

__global__ void SglangSigmoidGateBroadcast(__nv_bfloat16* output,
                                           const __nv_bfloat16* gate,
                                           int tokens) {
  const int vectors_per_row = kHidden / kVectorElements;
  const int vector = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total_vectors = tokens * vectors_per_row;
  if (vector >= total_vectors) return;
  const int token = vector / vectors_per_row;
  const int offset = (vector % vectors_per_row) * kVectorElements;
  const float gate_f32 = __bfloat162float(gate[token]);
  const float sigmoid = 1.0F / (1.0F + expf(-gate_f32));
  auto* output_vector = reinterpret_cast<uint4*>(
      output + token * kHidden + offset);
  uint4 output_bits = *output_vector;
  auto* values = reinterpret_cast<__nv_bfloat16*>(&output_bits);
#pragma unroll
  for (int element = 0; element < kVectorElements; ++element) {
    values[element] =
        __float2bfloat16_rn(__bfloat162float(values[element]) * sigmoid);
  }
  *output_vector = output_bits;
}

}  // namespace

SparkServeStatus sparkserve_sglang_cublas_shared_expert_cuda_launch(
    const SparkServeSharedExpertArgs* args) {
  const int tokens = static_cast<int>(args->plan.num_tokens);
  cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS shared-expert stream bind failed: ", status);
  }
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS shared-expert math-mode selection failed: ",
                       status);
  }

  status = RowMajorBf16Linear(handle, tokens, kGateUp, kHidden,
                              args->hidden_states, args->gate_up_weight,
                              args->gate_up, kGateUp);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS shared merged gate/up projection failed: ",
                       status);
  }
  if (args->plan.output_mode == SPARKSERVE_SHARED_EXPERT_OUTPUT_GATED) {
    status = RowMajorBf16Linear(handle, tokens, 1, kHidden,
                                args->hidden_states, args->shared_gate_weight,
                                args->shared_gate, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return CublasError("cuBLAS shared scalar gate failed: ", status);
    }
  }

  const int activation_vectors = tokens * (kIntermediate / kVectorElements);
  SglangSiluAndMul<<<(activation_vectors + kThreads - 1) / kThreads, kThreads,
                       0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->gate_up),
      static_cast<__nv_bfloat16*>(args->activated), tokens);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("SGLang shared SiLU launch failed: ", error);
  }

  status = RowMajorBf16Linear(handle, tokens, kHidden, kIntermediate,
                              args->activated, args->down_weight, args->output,
                              kHidden);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cuBLAS shared down projection failed: ", status);
  }
  if (args->plan.output_mode == SPARKSERVE_SHARED_EXPERT_OUTPUT_UNGATED) {
    return Ok();
  }
  const int output_vectors = tokens * (kHidden / kVectorElements);
  SglangSigmoidGateBroadcast<<<(output_vectors + kThreads - 1) / kThreads,
                                kThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(args->output),
      static_cast<const __nv_bfloat16*>(args->shared_gate), tokens);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("SGLang shared sigmoid-gate launch failed: ", error);
  }
  return Ok();
}

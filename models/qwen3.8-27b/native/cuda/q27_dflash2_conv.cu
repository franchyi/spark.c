// SPDX-License-Identifier: Apache-2.0
// Fixed T=8/K=2/group=16 translation of SGLang DFlashGroupedConv semantics.

#include "q27_dflash2_conv.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr uint32_t kTokens = Q27_DFLASH2_BLOCK_SIZE;
constexpr uint32_t kHidden = Q27_DFLASH2_HIDDEN_SIZE;
constexpr uint32_t kProjection = Q27_DFLASH2_CONV_PROJECTION_SIZE;
constexpr uint32_t kGroups = Q27_DFLASH2_CONV_GROUPS;
constexpr uint32_t kThreads = 256;
constexpr uint64_t kBaseBytes = 2ULL * Q27_DFLASH2_CONV_TAPS * kHidden * 2ULL;
constexpr uint64_t kProjectionBytes =
    static_cast<uint64_t>(kProjection) * kHidden * 2ULL;
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
  if (args == nullptr) return Invalid("DFlash2 convolution arguments are null");
  if (args->struct_size < sizeof(Args)) {
    return Invalid("DFlash2 convolution struct_size is too small");
  }
  if (args->abi_version != Q27_DFLASH2_CONV_ABI_VERSION) {
    return Invalid("DFlash2 convolution ABI version mismatch");
  }
  return Ok();
}

bool RangesOverlap(const void* left, uint64_t left_bytes, const void* right,
                   uint64_t right_bytes) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
  if (left_bytes > std::numeric_limits<uintptr_t>::max() - left_begin ||
      right_bytes > std::numeric_limits<uintptr_t>::max() - right_begin) {
    return true;
  }
  const uintptr_t left_end = left_begin + static_cast<uintptr_t>(left_bytes);
  const uintptr_t right_end = right_begin + static_cast<uintptr_t>(right_bytes);
  return left_begin < right_end && right_begin < left_end;
}

bool MisalignedBf16(const void* pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & 1U) != 0;
}

__device__ __forceinline__ __nv_bfloat16 AddBf16(__nv_bfloat16 left,
                                                  __nv_bfloat16 right) {
  return __float2bfloat16_rn(__bfloat162float(left) + __bfloat162float(right));
}

__device__ __forceinline__ __nv_bfloat16 MultiplyBf16(
    __nv_bfloat16 left, __nv_bfloat16 right) {
  return __float2bfloat16_rn(__bfloat162float(left) * __bfloat162float(right));
}

template <uint32_t Side>
__global__ void GroupedConv(const __nv_bfloat16* input,
                            const __nv_bfloat16* coefficients,
                            const __nv_bfloat16* base,
                            __nv_bfloat16* output) {
  static_assert(Side < 2);
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kTokens * kHidden; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / kHidden;
    const uint32_t channel = index - token * kHidden;
    const uint32_t group = channel / Q27_DFLASH2_CONV_GROUP_SIZE;
    const uint64_t coefficient_base =
        static_cast<uint64_t>(token) * kProjection +
        Side * Q27_DFLASH2_CONV_TAPS * kGroups;
    const uint64_t base_offset =
        Side * Q27_DFLASH2_CONV_TAPS * kHidden;

    const __nv_bfloat16 tap0 = AddBf16(
        base[base_offset + channel], coefficients[coefficient_base + group]);
    __nv_bfloat16 value = MultiplyBf16(tap0, input[index]);
    if (token != 0) {
      const __nv_bfloat16 tap1 = AddBf16(
          base[base_offset + kHidden + channel],
          coefficients[coefficient_base + kGroups + group]);
      const __nv_bfloat16 previous =
          MultiplyBf16(tap1, input[index - kHidden]);
      value = AddBf16(value, previous);
    }
    output[index] = value;
  }
}

q27_dflash2_status LaunchGroupedConv(
    uint32_t side, const void* input_bf16, const void* coefficients_bf16,
    const void* base_bf16, void* output_bf16, cudaStream_t stream) {
  constexpr uint32_t blocks = (kTokens * kHidden + kThreads - 1) / kThreads;
  if (side == 0) {
    GroupedConv<0><<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input_bf16),
        static_cast<const __nv_bfloat16*>(coefficients_bf16),
        static_cast<const __nv_bfloat16*>(base_bf16),
        static_cast<__nv_bfloat16*>(output_bf16));
  } else {
    GroupedConv<1><<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input_bf16),
        static_cast<const __nv_bfloat16*>(coefficients_bf16),
        static_cast<const __nv_bfloat16*>(base_bf16),
        static_cast<__nv_bfloat16*>(output_bf16));
  }
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("DFlash2 grouped convolution", error);
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_conv_prepare(
    const q27_dflash2_conv_prepare_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->base_kernel.data == nullptr || args->base_kernel.bytes != kBaseBytes ||
      args->kernel_projection.data == nullptr ||
      args->kernel_projection.bytes != kProjectionBytes ||
      args->input_bf16 == nullptr || args->coefficients_bf16 == nullptr ||
      args->output_bf16 == nullptr || args->cublas_handle == nullptr ||
      MisalignedBf16(args->base_kernel.data) ||
      MisalignedBf16(args->kernel_projection.data) ||
      MisalignedBf16(args->input_bf16) ||
      MisalignedBf16(args->coefficients_bf16) ||
      MisalignedBf16(args->output_bf16) ||
      RangesOverlap(args->input_bf16, kTokens * kHidden * 2ULL,
                    args->coefficients_bf16,
                    Q27_DFLASH2_CONV_COEFFICIENT_BYTES) ||
      RangesOverlap(args->input_bf16, kTokens * kHidden * 2ULL,
                    args->output_bf16, kTokens * kHidden * 2ULL) ||
      RangesOverlap(args->coefficients_bf16,
                    Q27_DFLASH2_CONV_COEFFICIENT_BYTES, args->output_bf16,
                    kTokens * kHidden * 2ULL) ||
      RangesOverlap(args->input_bf16, kTokens * kHidden * 2ULL,
                    args->base_kernel.data, kBaseBytes) ||
      RangesOverlap(args->input_bf16, kTokens * kHidden * 2ULL,
                    args->kernel_projection.data, kProjectionBytes) ||
      RangesOverlap(args->coefficients_bf16,
                    Q27_DFLASH2_CONV_COEFFICIENT_BYTES,
                    args->base_kernel.data, kBaseBytes) ||
      RangesOverlap(args->coefficients_bf16,
                    Q27_DFLASH2_CONV_COEFFICIENT_BYTES,
                    args->kernel_projection.data, kProjectionBytes) ||
      RangesOverlap(args->output_bf16, kTokens * kHidden * 2ULL,
                    args->base_kernel.data, kBaseBytes) ||
      RangesOverlap(args->output_bf16, kTokens * kHidden * 2ULL,
                    args->kernel_projection.data, kProjectionBytes)) {
    return Invalid("invalid DFlash2 convolution prepare arguments");
  }

  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("DFlash2 convolution set stream", cublas_status);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  /* Row-major [8,5120] * [1280,5120]^T. */
  cublas_status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, kProjection, kTokens, kHidden, &alpha,
      args->kernel_projection.data, CUDA_R_16BF, kHidden, args->input_bf16,
      CUDA_R_16BF, kHidden, &beta, args->coefficients_bf16, CUDA_R_16BF,
      kProjection, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("DFlash2 convolution coefficient projection",
                       cublas_status);
  }
  return LaunchGroupedConv(0, args->input_bf16, args->coefficients_bf16,
                           args->base_kernel.data, args->output_bf16, stream);
}

extern "C" q27_dflash2_status q27_dflash2_conv_finish(
    const q27_dflash2_conv_finish_args* args) {
  q27_dflash2_status status = ValidateHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->base_kernel.data == nullptr || args->base_kernel.bytes != kBaseBytes ||
      args->input_bf16 == nullptr || args->coefficients_bf16 == nullptr ||
      args->output_bf16 == nullptr || MisalignedBf16(args->base_kernel.data) ||
      MisalignedBf16(args->input_bf16) ||
      MisalignedBf16(args->coefficients_bf16) ||
      MisalignedBf16(args->output_bf16) ||
      RangesOverlap(args->input_bf16, kTokens * kHidden * 2ULL,
                    args->coefficients_bf16,
                    Q27_DFLASH2_CONV_COEFFICIENT_BYTES) ||
      RangesOverlap(args->input_bf16, kTokens * kHidden * 2ULL,
                    args->output_bf16, kTokens * kHidden * 2ULL) ||
      RangesOverlap(args->coefficients_bf16,
                    Q27_DFLASH2_CONV_COEFFICIENT_BYTES, args->output_bf16,
                    kTokens * kHidden * 2ULL) ||
      RangesOverlap(args->input_bf16, kTokens * kHidden * 2ULL,
                    args->base_kernel.data, kBaseBytes) ||
      RangesOverlap(args->coefficients_bf16,
                    Q27_DFLASH2_CONV_COEFFICIENT_BYTES,
                    args->base_kernel.data, kBaseBytes) ||
      RangesOverlap(args->output_bf16, kTokens * kHidden * 2ULL,
                    args->base_kernel.data, kBaseBytes)) {
    return Invalid("invalid DFlash2 convolution finish arguments");
  }
  return LaunchGroupedConv(
      1, args->input_bf16, args->coefficients_bf16, args->base_kernel.data,
      args->output_bf16, static_cast<cudaStream_t>(args->cuda_stream));
}

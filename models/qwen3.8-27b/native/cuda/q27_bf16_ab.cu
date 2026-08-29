/* Lightweight BF16 gate projections for Qwen3.8-27B GDN decode. */

#include "q27_bf16_ab.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>

namespace {

thread_local std::string g_error;

q27_kernel_status Ok() { return {Q27_KERNEL_OK, "ok"}; }

q27_kernel_status Invalid(const char* message) {
  return {Q27_KERNEL_INVALID_ARGUMENT, message};
}

q27_kernel_status CublasError(const char* prefix, cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append(cublasGetStatusString(status));
  return {Q27_KERNEL_CUDA_ERROR, g_error.c_str()};
}

cublasStatus_t Project(cublasHandle_t handle, const void* input,
                       const void* weight, void* output) {
  constexpr int kHidden = 5120;
  constexpr int kHeads = 48;
  const float alpha = 1.0F;
  const float beta = 0.0F;

  // A row-major [48, 5120] checkpoint matrix is a column-major [5120, 48]
  // view to cuBLAS. Transposing that view computes W*x without repacking.
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kHeads, 1, kHidden,
                      &alpha, weight, CUDA_R_16BF, kHidden, input,
                      CUDA_R_16BF, kHidden, &beta, output, CUDA_R_16BF,
                      kHeads, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

}  // namespace

extern "C" q27_kernel_status q27_bf16_ab_project(
    const q27_bf16_ab_project_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_KERNEL_ABI_VERSION ||
      args->hidden_size != 5120 || args->value_heads != 48 ||
      args->input_bf16 == nullptr || args->weight_a_bf16 == nullptr ||
      args->weight_b_bf16 == nullptr || args->output_a_bf16 == nullptr ||
      args->output_b_bf16 == nullptr || args->cublas_handle == nullptr) {
    return Invalid("invalid q27 BF16 a/b projection arguments");
  }

  const cublasHandle_t handle =
      static_cast<cublasHandle_t>(args->cublas_handle);
  cublasStatus_t status =
      cublasSetStream(handle, static_cast<cudaStream_t>(args->cuda_stream));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("q27 BF16 a/b stream: ", status);
  }
  status = Project(handle, args->input_bf16, args->weight_a_bf16,
                   args->output_a_bf16);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("q27 BF16 in_proj_a: ", status);
  }
  status = Project(handle, args->input_bf16, args->weight_b_bf16,
                   args->output_b_bf16);
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("q27 BF16 in_proj_b: ", status);
}

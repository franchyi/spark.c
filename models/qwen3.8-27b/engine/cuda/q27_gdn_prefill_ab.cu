/* Fixed-M128 merged BF16 A/B projection for Qwen3.8-27B GDN prefill. */

#include "q27_gdn_prefill_ab.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr uint64_t kHiddenBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
    Q27_GDN_PREFILL_AB_HIDDEN * 2;
constexpr uint64_t kWeightBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_AB_MERGED_HEADS) *
    Q27_GDN_PREFILL_AB_HIDDEN * 2;
constexpr uint64_t kMergedBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
    Q27_GDN_PREFILL_AB_MERGED_HEADS * 2;
constexpr uint64_t kOutputBytes =
    static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
    Q27_GDN_PREFILL_AB_HEADS * 2;

thread_local std::string g_error;

q27_gdn_prefill_ab_status Ok() { return {Q27_GDN_PREFILL_AB_OK, "ok"}; }

q27_gdn_prefill_ab_status Invalid(const char* message) {
  return {Q27_GDN_PREFILL_AB_INVALID_ARGUMENT, message};
}

q27_gdn_prefill_ab_status CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_PREFILL_AB_CUDA_ERROR, g_error.c_str()};
}

q27_gdn_prefill_ab_status CublasError(const char* prefix,
                                      cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append("cuBLAS status ");
  g_error.append(std::to_string(static_cast<int>(status)));
  return {Q27_GDN_PREFILL_AB_CUBLAS_ERROR, g_error.c_str()};
}

__global__ void SplitAndMask(const __nv_bfloat16* merged,
                             __nv_bfloat16* output_a,
                             __nv_bfloat16* output_b,
                             uint32_t valid_tokens) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements =
      static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
      Q27_GDN_PREFILL_AB_HEADS;
  if (index >= elements) return;
  const int head = index % Q27_GDN_PREFILL_AB_HEADS;
  const int token = index / Q27_GDN_PREFILL_AB_HEADS;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    output_a[index] = __float2bfloat16_rn(0.0F);
    output_b[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const uint64_t merged_row =
      static_cast<uint64_t>(token) * Q27_GDN_PREFILL_AB_MERGED_HEADS;
  output_a[index] = merged[merged_row + head];
  output_b[index] = merged[merged_row + Q27_GDN_PREFILL_AB_HEADS + head];
}

}  // namespace

extern "C" q27_gdn_prefill_ab_status q27_gdn_prefill_ab_project(
    const q27_gdn_prefill_ab_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_AB_ABI_VERSION ||
      args->valid_tokens == 0 ||
      args->valid_tokens > Q27_GDN_PREFILL_AB_TOKENS ||
      args->normalized_hidden_bf16 == nullptr ||
      args->merged_weight_bf16 == nullptr ||
      args->merged_scratch_bf16 == nullptr ||
      args->projected_a_bf16 == nullptr || args->projected_b_bf16 == nullptr ||
      args->cublas_handle == nullptr ||
      args->normalized_hidden_bytes < kHiddenBytes ||
      args->merged_weight_bytes < kWeightBytes ||
      args->merged_scratch_bytes < kMergedBytes ||
      args->projected_a_bytes < kOutputBytes ||
      args->projected_b_bytes < kOutputBytes)
    return Invalid("invalid q27 GDN prefill A/B projection arguments");

  cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(args->cublas_handle);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasPointerMode_t mode;
  cublasStatus_t status = cublasGetPointerMode(handle, &mode);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN prefill A/B get pointer mode: ", status);
  if (mode != CUBLAS_POINTER_MODE_HOST)
    return Invalid("q27 GDN prefill A/B requires host pointer mode");
  status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN prefill A/B set stream: ", status);

  const float alpha = 1.0F;
  const float beta = 0.0F;
  // Row-major W[96,5120] and X[128,5120] are column-major transposed views.
  // C[96,128] column-major is the desired row-major [128,96] scratch.
  status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, Q27_GDN_PREFILL_AB_MERGED_HEADS,
      Q27_GDN_PREFILL_AB_TOKENS, Q27_GDN_PREFILL_AB_HIDDEN, &alpha,
      args->merged_weight_bf16, CUDA_R_16BF, Q27_GDN_PREFILL_AB_HIDDEN,
      args->normalized_hidden_bf16, CUDA_R_16BF, Q27_GDN_PREFILL_AB_HIDDEN,
      &beta, args->merged_scratch_bf16, CUDA_R_16BF,
      Q27_GDN_PREFILL_AB_MERGED_HEADS, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN prefill merged A/B GEMM: ", status);

  constexpr int kThreads = 256;
  constexpr uint64_t kElements =
      static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
      Q27_GDN_PREFILL_AB_HEADS;
  SplitAndMask<<<(kElements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->merged_scratch_bf16),
      static_cast<__nv_bfloat16*>(args->projected_a_bf16),
      static_cast<__nv_bfloat16*>(args->projected_b_bf16),
      args->valid_tokens);
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("q27 GDN prefill split/mask A/B: ", error);
}

#include "flash/qwen_decode_glue_api.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kHyper = kHidden * kStreams;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kThreads = 256;

thread_local std::string g_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }
FlashStatus Invalid(const char* message) {
  return {FLASH_STATUS_INVALID_ARGUMENT, message};
}

FlashStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

FlashStatus CublasError(const char* prefix, cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append(std::to_string(static_cast<int>(status)));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

__global__ void RepeatEmbedding(const __nv_bfloat16* input,
                                __nv_bfloat16* output) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < kHyper) output[index] = input[index % kHidden];
}

__global__ void AddHyper(const __nv_bfloat16* delta,
                         __nv_bfloat16* hidden) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < kHyper) {
    hidden[index] = __float2bfloat16_rn(__bfloat162float(hidden[index]) +
                                        __bfloat162float(delta[index]));
  }
}

// At sequence length one, every query head has one legal key. Softmax is one,
// so QSA attention is exactly the corresponding grouped KV value.
__global__ void ExpandSingleValue(const __nv_bfloat16* value,
                                  __nv_bfloat16* output) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kQueryHeads * kHeadDim) return;
  const int query_head = index / kHeadDim;
  const int column = index % kHeadDim;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  output[index] = value[kv_head * kHeadDim + column];
}

FlashStatus ValidateGlue(const FlashQwenDecodeGlueArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_DECODE_GLUE_ABI_VERSION ||
      args->input == nullptr || args->output == nullptr) {
    return Invalid("Qwen decode glue arguments are invalid");
  }
  return Ok();
}

}  // namespace

extern "C" FlashStatus flash_qwen_repeat_embedding_launch(
    const FlashQwenDecodeGlueArgs* args) {
  FlashStatus valid = ValidateGlue(args);
  if (valid.code != FLASH_STATUS_OK) return valid;
  RepeatEmbedding<<<(kHyper + kThreads - 1) / kThreads, kThreads, 0,
                     static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input),
      static_cast<__nv_bfloat16*>(args->output));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen embedding repeat failed: ", error);
}

extern "C" FlashStatus flash_qwen_add_hyper_launch(
    const FlashQwenDecodeGlueArgs* args) {
  FlashStatus valid = ValidateGlue(args);
  if (valid.code != FLASH_STATUS_OK) return valid;
  AddHyper<<<(kHyper + kThreads - 1) / kThreads, kThreads, 0,
             static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input),
      static_cast<__nv_bfloat16*>(args->output));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("Qwen PLE add failed: ", error);
}

extern "C" FlashStatus flash_qwen_qsa_single_value_launch(
    const FlashQwenDecodeGlueArgs* args) {
  FlashStatus valid = ValidateGlue(args);
  if (valid.code != FLASH_STATUS_OK) return valid;
  constexpr int kElements = kQueryHeads * kHeadDim;
  ExpandSingleValue<<<(kElements + kThreads - 1) / kThreads, kThreads, 0,
                      static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input),
      static_cast<__nv_bfloat16*>(args->output));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen single-token QSA failed: ", error);
}

extern "C" FlashStatus flash_qwen_lm_head_launch(
    const FlashQwenLmHeadArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_DECODE_GLUE_ABI_VERSION ||
      args->vocabulary == 0 || args->hidden_size == 0 ||
      args->hidden_states == nullptr || args->weight == nullptr ||
      args->logits == nullptr || args->cublas_handle == nullptr) {
    return Invalid("Qwen LM-head arguments are invalid");
  }
  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cublasStatus_t status =
      cublasSetStream(handle, static_cast<cudaStream_t>(args->cuda_stream));
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen LM-head stream bind failed: ", status);
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen LM-head math-mode failed: ", status);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(args->vocabulary), 1,
      static_cast<int>(args->hidden_size), &alpha, args->weight, CUDA_R_16BF,
      static_cast<int>(args->hidden_size), args->hidden_states, CUDA_R_16BF,
      static_cast<int>(args->hidden_size), &beta, args->logits, CUDA_R_32F,
      static_cast<int>(args->vocabulary), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("Qwen LM-head projection failed: ", status);
}

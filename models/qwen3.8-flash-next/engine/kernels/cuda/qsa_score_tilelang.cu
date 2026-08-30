// SGLang's QSA decode score is generated from mqa.py at commit
// d91c3682b0b429e4c70df63cd57f819588ce29b0 by TileLang 0.1.11
// (cd37ed5fc35ae7a60a1277c8eb49028174ac51e6). The generated MMA kernel and
// its MIT-licensed CUDA templates are included byte-for-byte; this file only
// replaces TVM-FFI with Flash's raw ABI and fixed stream ownership.

#include "internal/qsa_score_backend.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#define kernel_kernel flash_tilelang_qsa_score_kernel
#include "generated/qsa-score/generated/device_kernel.cu"
#undef kernel_kernel

namespace {

constexpr int kThreads = 128;
constexpr int kSubPages = 4;
constexpr size_t kDynamicSharedBytes = 18'432;

thread_local std::string g_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }

FlashStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

}  // namespace

FlashStatus flash_tilelang_qsa_score_cuda_launch(
    const FlashQsaScoreArgs* args) {
  const dim3 grid(args->plan.batch_size,
                  (args->plan.max_pages + kSubPages - 1) / kSubPages);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  flash_tilelang_qsa_score_kernel<<<grid, kThreads, kDynamicSharedBytes,
                                         stream>>>(
      args->context_lengths,
      static_cast<const bfloat16_t*>(args->key_cache), args->logits,
      args->page_table, static_cast<const bfloat16_t*>(args->query),
      args->score_scale, static_cast<int>(args->plan.batch_size),
      static_cast<int>(args->plan.max_model_len),
      static_cast<int>(args->plan.max_pages),
      static_cast<int>(args->plan.pages));
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("QSA TileLang score launch failed: ", error);
  }
  return Ok();
}

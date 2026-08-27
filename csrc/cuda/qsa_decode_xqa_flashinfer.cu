// Framework-free adapter for FlashInfer XQA at immutable commit
// 906181e3f4cf4bcc81835fb480db4011bbd80b62. The build links the upstream
// xqa/mha.cu specialization; this file only replaces Torch/TVM-FFI tensor
// wrappers with SparkServe's validated raw-pointer ABI.

#include "internal/qsa_decode_backend.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "mha.h"

namespace {

constexpr uint64_t kSemaphoreBytes = 8ULL * 1024 * 1024;

static_assert(HEAD_ELEMS == 256);
static_assert(HEAD_GRP_SIZE == 12);
static_assert(TOKENS_PER_PAGE == 64);
static_assert(INPUT_FP16 == 0);
static_assert(CACHE_ELEM_ENUM == 0);
static_assert(SPEC_DEC == 0);
static_assert(LOW_PREC_OUTPUT == 0);

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

}  // namespace

SparkServeStatus sparkserve_flashinfer_xqa_decode_cuda_launch(
    const SparkServeQsaDecodeArgs* args) {
  auto* workspace = static_cast<uint8_t*>(args->workspace);
  auto* semaphores = reinterpret_cast<uint32_t*>(workspace);
  void* scratch = workspace + kSemaphoreBytes;
  const float kv_scale = args->bmm2_scale;
  const float q_scale = args->bmm1_scale * 16.0f / kv_scale;
  const uint64_t stride_page =
      static_cast<uint64_t>(args->plan.page_size) * args->plan.num_kv_heads *
      args->plan.head_dim;
  const uint64_t stride_token =
      static_cast<uint64_t>(args->plan.num_kv_heads) * args->plan.head_dim;
  const uint64_t stride_head = args->plan.head_dim;
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);

  launchMHAFlashInfer(
      args->plan.multiprocessor_count, args->plan.num_kv_heads,
      /*slidingWinSize=*/0, q_scale, /*qScalePtr=*/nullptr,
      reinterpret_cast<OutputHead*>(args->output),
      reinterpret_cast<const InputHead*>(args->query),
      /*attentionSinks=*/nullptr,
      reinterpret_cast<GMemCacheHead*>(const_cast<void*>(args->packed_key)),
      reinterpret_cast<GMemCacheHead*>(const_cast<void*>(args->packed_value)),
      reinterpret_cast<const KVCachePageIndex*>(args->block_tables),
      args->plan.packed_row_stride,
      reinterpret_cast<const uint32_t*>(args->sequence_lengths),
      args->plan.batch_size, kv_scale, /*kvScalePtr=*/nullptr, semaphores,
      scratch, args->plan.enable_pdl != 0, stride_page, stride_token,
      stride_head, stream);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("FlashInfer XQA launch failed: ", error);
  return Ok();
}

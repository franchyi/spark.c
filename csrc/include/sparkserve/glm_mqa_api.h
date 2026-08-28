#ifndef SPARKSERVE_GLM_MQA_API_H_
#define SPARKSERVE_GLM_MQA_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_GLM_MQA_ABI_VERSION 1u
#define SPARKSERVE_GLM_MQA_GB10_SMS 48u
#define SPARKSERVE_GLM_MQA_SCHEDULE_WORDS 98u

// Decode-only GLM-5.3 sparse-index logits using the pinned SGLang DeepGEMM
// SM120 specialization. Query is FP8 E4M3 [batch,1,32,128]. Key pages are
// FP8 E4M3 [num_pages,64,128] and scales are FP32 [num_pages,64]. Query scale
// has already been folded into the FP32 per-head weights by index preparation.
// All pointers are device pointers; every allocation and the schedule buffer
// remains caller-owned and fixed-address.
typedef struct SparkServeGlmPagedMqaArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t batch_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t page_size;
  uint32_t num_pages;
  uint32_t num_sms;
  uint32_t max_context_len;
  uint32_t logits_stride;
  uint32_t block_table_stride;
  uint32_t reserved;
  const uint8_t* query_fp8;
  const uint8_t* key_cache_fp8;
  const float* scale_cache;
  const float* logit_weights;
  const uint32_t* context_lens;
  float* logits;
  const uint32_t* block_tables;
  uint32_t* schedule_metadata;
  uint64_t key_page_stride_bytes;
  uint64_t scale_page_stride_bytes;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmPagedMqaArgs;

SparkServeStatus sparkserve_glm_paged_mqa_validate(
    const SparkServeGlmPagedMqaArgs* args);

SparkServeStatus sparkserve_glm_paged_mqa_launch(
    const SparkServeGlmPagedMqaArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_GLM_MQA_API_H_

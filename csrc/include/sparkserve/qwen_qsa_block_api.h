#ifndef SPARKSERVE_QWEN_QSA_BLOCK_API_H_
#define SPARKSERVE_QWEN_QSA_BLOCK_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_QWEN_QSA_BLOCK_ABI_VERSION 1u

// Fixed Qwen3.8 Flash-Next attention frontend. cuBLAS owns the four BF16
// projections; the fused CUDA epilogue is adapted from SGLang's Apache-2.0
// fused_qk_rmsnorm_rope_gate Triton kernel.
typedef struct SparkServeQwenQsaProjectArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t tokens;
  uint32_t rotary_dim;
  uint64_t cos_sin_stride;
  const void* hidden_states;
  const void* q_weight;
  const void* k_weight;
  const void* v_weight;
  const void* index_qk_weight;
  const void* q_norm_weight;
  const void* k_norm_weight;
  const float* cos_sin_cache;
  const int64_t* positions;
  void* projected_q;
  void* projected_k;
  void* query;
  void* key;
  void* value;
  void* gate;
  void* index_qk;
  void* cublas_handle;
  void* cuda_stream;
} SparkServeQwenQsaProjectArgs;

// SGLang's sigmoid attention gate followed by the checkpoint output
// projection. `gated_output` is caller-owned fixed scratch.
typedef struct SparkServeQwenQsaFinishArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t tokens;
  uint32_t reserved;
  const void* attention_output;
  const void* gate;
  const void* out_weight;
  void* gated_output;
  void* output;
  void* cublas_handle;
  void* cuda_stream;
} SparkServeQwenQsaFinishArgs;

SparkServeStatus sparkserve_qwen_qsa_project_launch(
    const SparkServeQwenQsaProjectArgs* args);
SparkServeStatus sparkserve_qwen_qsa_finish_launch(
    const SparkServeQwenQsaFinishArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_QWEN_QSA_BLOCK_API_H_

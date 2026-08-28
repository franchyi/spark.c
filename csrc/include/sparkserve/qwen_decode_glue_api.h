#ifndef SPARKSERVE_QWEN_DECODE_GLUE_API_H_
#define SPARKSERVE_QWEN_DECODE_GLUE_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_QWEN_DECODE_GLUE_ABI_VERSION 1u

typedef struct SparkServeQwenDecodeGlueArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* input;
  void* output;
  void* cuda_stream;
} SparkServeQwenDecodeGlueArgs;

typedef struct SparkServeQwenLmHeadArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t vocabulary;
  uint32_t hidden_size;
  const void* hidden_states;
  const void* weight;
  float* logits;
  void* cublas_handle;
  void* cuda_stream;
} SparkServeQwenLmHeadArgs;

SparkServeStatus sparkserve_qwen_repeat_embedding_launch(
    const SparkServeQwenDecodeGlueArgs* args);
SparkServeStatus sparkserve_qwen_add_hyper_launch(
    const SparkServeQwenDecodeGlueArgs* args);
SparkServeStatus sparkserve_qwen_qsa_single_value_launch(
    const SparkServeQwenDecodeGlueArgs* args);
SparkServeStatus sparkserve_qwen_lm_head_launch(
    const SparkServeQwenLmHeadArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_QWEN_DECODE_GLUE_API_H_

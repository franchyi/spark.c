#ifndef FLASH_QWEN_DECODE_GLUE_API_H_
#define FLASH_QWEN_DECODE_GLUE_API_H_

#include <stdint.h>

#include "flash/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FLASH_QWEN_DECODE_GLUE_ABI_VERSION 1u

typedef struct FlashQwenDecodeGlueArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* input;
  void* output;
  void* cuda_stream;
} FlashQwenDecodeGlueArgs;

typedef struct FlashQwenLmHeadArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t vocabulary;
  uint32_t hidden_size;
  const void* hidden_states;
  const void* weight;
  float* logits;
  void* cublas_handle;
  void* cuda_stream;
} FlashQwenLmHeadArgs;

// Device-side greedy selection keeps the full vocabulary logits resident on
// the GPU.  Only the four-byte winning token crosses the coherent CPU alias.
typedef struct FlashQwenArgmaxArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t elements;
  uint32_t reserved;
  const float* values;
  uint32_t* output_index;
  void* cuda_stream;
} FlashQwenArgmaxArgs;

FlashStatus flash_qwen_repeat_embedding_launch(
    const FlashQwenDecodeGlueArgs* args);
FlashStatus flash_qwen_add_hyper_launch(
    const FlashQwenDecodeGlueArgs* args);
FlashStatus flash_qwen_qsa_single_value_launch(
    const FlashQwenDecodeGlueArgs* args);
FlashStatus flash_qwen_lm_head_launch(
    const FlashQwenLmHeadArgs* args);
FlashStatus flash_qwen_argmax_launch(
    const FlashQwenArgmaxArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FLASH_QWEN_DECODE_GLUE_API_H_

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
  uint32_t tokens;
  uint32_t vocabulary;
  uint32_t hidden_size;
  uint32_t reserved;
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
  uint32_t rows;
  uint32_t elements;
  uint32_t row_stride;
  uint32_t reserved;
  const float* values;
  uint32_t* output_index;
  void* cuda_stream;
} FlashQwenArgmaxArgs;

// Qwen4-Exp NEXTN input fusion. The checkpoint keeps two independent BF16
// projections: one for the candidate-token embedding and one shared across
// the four target hidden-state branches.
typedef struct FlashQwenMtpInputArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* embedding;
  const void* target_hidden;
  const void* embedding_norm_weight;
  const void* hidden_norm_weight;
  const void* embedding_fc_weight;
  const void* hidden_fc_weight;
  void* embedding_norm_scratch;
  void* embedding_projected_scratch;
  void* hidden_norm_scratch;
  void* output;
  void* cublas_handle;
  void* cuda_stream;
} FlashQwenMtpInputArgs;

// Fixed batch-one, top-10 BF16 routed expert path for the bundled MTP layer.
// Expert tensors retain their checkpoint layout:
// gate_up [512,1280,2560], down [512,2560,640].
typedef struct FlashQwenMtpExpertsArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* hidden_states;
  const int32_t* expert_ids;
  const float* expert_weights;
  const void* gate_up_weight;
  const void* down_weight;
  void* gate_up_scratch;
  void* activated_scratch;
  void* expert_output_scratch;
  void* output;
  void* cuda_stream;
} FlashQwenMtpExpertsArgs;

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
FlashStatus flash_qwen_mtp_input_launch(
    const FlashQwenMtpInputArgs* args);
FlashStatus flash_qwen_mtp_experts_launch(
    const FlashQwenMtpExpertsArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FLASH_QWEN_DECODE_GLUE_API_H_

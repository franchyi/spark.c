#ifndef Q27_GDN_PREFILL_LAYER_H_
#define Q27_GDN_PREFILL_LAYER_H_

#include "q27_prefill_fp8.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_PREFILL_LAYER_ABI_VERSION 1u

typedef struct q27_gdn_prefill_layer_status {
  int32_t code;
  const char* message;
} q27_gdn_prefill_layer_status;

enum {
  Q27_GDN_PREFILL_LAYER_OK = 0,
  Q27_GDN_PREFILL_LAYER_INVALID_ARGUMENT = 1,
  Q27_GDN_PREFILL_LAYER_CAPSULE_ERROR = 2,
  Q27_GDN_PREFILL_LAYER_CUDA_ERROR = 3,
};

typedef struct q27_gdn_prefill_layer_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t scratch_bytes;
  uint64_t scratch_alignment;
  uint64_t convolution_state_bytes;
  uint64_t recurrent_state_bytes;
  uint64_t qkvz_workspace_bytes;
  uint64_t sublayer_scratch_bytes;
} q27_gdn_prefill_layer_layout;

/* Thin full transformer-layer boundary around the validated GDN sublayer. */
typedef struct q27_gdn_prefill_layer_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t has_input_residual;
  const void* input_hidden_bf16;       /* [128,5120] */
  const void* input_residual_bf16;     /* optional [128,5120] */
  const void* input_norm_weight_bf16;  /* [5120] Gemma weight */
  const void* post_norm_weight_bf16;   /* [5120] Gemma weight */
  float norm_epsilon;
  uint32_t reserved;
  const void* qkvz_weight_fp8_e4m3;    /* [16384,5120] */
  uint64_t qkvz_weight_bytes;
  const float* qkvz_input_scale;
  const float* qkvz_weight_scale;
  const void* conv_weight_bf16;
  const void* merged_ab_weight_bf16;
  const float* a_log_f32;
  const float* dt_bias_f32;
  const void* gdn_norm_weight_bf16;
  const void* out_weight_fp8_e4m3;
  uint64_t out_weight_bytes;
  const float* out_input_scale;
  const float* out_weight_scale;
  void* convolution_state_bf16;
  uint64_t convolution_state_bytes;
  void* recurrent_state_bf16;
  uint64_t recurrent_state_bytes;
  void* normalized_output_bf16;        /* post-attention norm [128,5120] */
  void* residual_output_bf16;          /* post-attention residual [128,5120] */
  void* scratch;
  uint64_t scratch_bytes;
  q27_prefill_fp8_plan* qkvz_plan;
  q27_prefill_fp8_plan* output_plan;
  void* cublas_handle;
  void* cuda_stream;
} q27_gdn_prefill_layer_args;

q27_gdn_prefill_layer_status q27_gdn_prefill_layer_query(
    q27_gdn_prefill_layer_layout* output);
q27_gdn_prefill_layer_status q27_gdn_prefill_layer_forward(
    const q27_gdn_prefill_layer_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_LAYER_H_

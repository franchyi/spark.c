#ifndef Q27_GDN_PREFILL_SUBLAYER_H_
#define Q27_GDN_PREFILL_SUBLAYER_H_

#include "q27_prefill_fp8.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_PREFILL_SUBLAYER_ABI_VERSION 2u

typedef enum q27_gdn_prefill_sublayer_status_code {
  Q27_GDN_PREFILL_SUBLAYER_OK = 0,
  Q27_GDN_PREFILL_SUBLAYER_INVALID_ARGUMENT = 1,
  Q27_GDN_PREFILL_SUBLAYER_CAPSULE_ERROR = 2,
  Q27_GDN_PREFILL_SUBLAYER_CUDA_ERROR = 3,
} q27_gdn_prefill_sublayer_status_code;

typedef struct q27_gdn_prefill_sublayer_status {
  int32_t code;
  const char* message;
} q27_gdn_prefill_sublayer_status;

typedef struct q27_gdn_prefill_sublayer_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t scratch_bytes;
  uint64_t scratch_alignment;
  uint64_t convolution_state_bytes;
  uint64_t recurrent_state_bytes;
  uint64_t output_quantized_bytes;
  uint64_t output_workspace_bytes;
} q27_gdn_prefill_sublayer_layout;

/*
 * Fixed physical M=128 GDN prefill sublayer. QKV/Z and normalized_hidden are
 * already projected inputs for this model-specific sublayer; the final
 * validated batched FP8 output projection is included. Every pointer is
 * caller-owned and CUDA-visible. No allocation, framework, JIT, or M=1 tail
 * path is permitted.
 */
typedef struct q27_gdn_prefill_sublayer_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* normalized_hidden_bf16; /* [128,5120] */
  uint64_t normalized_hidden_bytes;
  const void* projected_qkv_bf16;     /* [128,10240] */
  uint64_t projected_qkv_bytes;
  const void* projected_z_bf16;       /* [128,6144] */
  uint64_t projected_z_bytes;
  const void* conv_weight_bf16;       /* [10240,4] */
  uint64_t conv_weight_bytes;
  const void* merged_ab_weight_bf16;  /* [96,5120] */
  uint64_t merged_ab_weight_bytes;
  const float* a_log_f32;             /* [48] */
  const float* dt_bias_f32;           /* [48] */
  const void* norm_weight_bf16;       /* [128] */
  const void* out_weight_fp8_e4m3;    /* [5120,6144] */
  uint64_t out_weight_bytes;
  const float* out_input_scale;
  const float* out_weight_scale;
  void* convolution_state_bf16;       /* [10240,3], updated */
  uint64_t convolution_state_bytes;
  void* recurrent_state_bf16;         /* [48,128,128], updated */
  uint64_t recurrent_state_bytes;
  void* output_hidden_bf16;           /* [128,5120] */
  uint64_t output_hidden_bytes;
  void* scratch;
  uint64_t scratch_bytes;
  q27_prefill_fp8_plan* output_plan;
  void* cublas_handle;
  void* cuda_stream;
  uint32_t verify_t8_gdn;
  uint32_t reserved2;
  void* checkpoint_convolution_bf16; /* [8,10240,3], verify only */
  uint64_t checkpoint_convolution_bytes;
  void* checkpoint_recurrent_bf16;   /* [8,48,128,128], verify only */
  uint64_t checkpoint_recurrent_bytes;
  const int32_t* state_index_i32;     /* device scalar zero, verify only */
} q27_gdn_prefill_sublayer_args;

q27_gdn_prefill_sublayer_status q27_gdn_prefill_sublayer_query(
    q27_gdn_prefill_sublayer_layout* output);
q27_gdn_prefill_sublayer_status q27_gdn_prefill_sublayer_forward(
    const q27_gdn_prefill_sublayer_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_SUBLAYER_H_

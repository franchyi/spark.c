#ifndef Q27_GDN_PREFILL_M512_H_
#define Q27_GDN_PREFILL_M512_H_

#include "q27_prefill_fp8.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_PREFILL_M512_ABI_VERSION 1u

enum {
  Q27_GDN_PREFILL_M512_TOKENS = 512,
  Q27_GDN_PREFILL_M512_CHUNK_TOKENS = 128,
  Q27_GDN_PREFILL_M512_CHUNKS = 4,
  Q27_GDN_PREFILL_M512_HIDDEN = 5120,
  Q27_GDN_PREFILL_M512_QKVZ = 16384,
  Q27_GDN_PREFILL_M512_QKV = 10240,
  Q27_GDN_PREFILL_M512_VALUE = 6144,
};

typedef enum q27_gdn_prefill_m512_status_code {
  Q27_GDN_PREFILL_M512_OK = 0,
  Q27_GDN_PREFILL_M512_INVALID_ARGUMENT = 1,
  Q27_GDN_PREFILL_M512_CAPSULE_ERROR = 2,
  Q27_GDN_PREFILL_M512_CUDA_ERROR = 3,
} q27_gdn_prefill_m512_status_code;

typedef struct q27_gdn_prefill_m512_status {
  int32_t code;
  const char* message;
} q27_gdn_prefill_m512_status;

typedef struct q27_gdn_prefill_m512_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t scratch_bytes;
  uint64_t scratch_alignment;
  uint64_t convolution_state_bytes;
  uint64_t recurrent_state_bytes;
  uint64_t qkvz_workspace_bytes;
  uint64_t output_workspace_bytes;
  uint64_t pre_output_bf16_bytes;
  uint64_t shared_bytes;
} q27_gdn_prefill_m512_layout;

/*
 * Fixed physical M=512 Qwen3.8 GDN transformer layer.
 *
 * The hot call performs one M512 input norm and one M512 FP8 QKVZ projection,
 * then executes up to four ordered, existing M128 recurrent chunks against
 * the same live convolution/recurrent state.  Their gated-normalized
 * [128,6144] results are collected into one [512,6144] tile and consumed by
 * one M512 FP8 output projection.  The final post-attention norm is M512.
 * Rows valid_tokens..511 are padding and cannot mutate state.  The QKVZ and
 * output weights are each read by exactly one projection, never once per
 * recurrent chunk.
 *
 * All pointers are caller-owned CUDA-visible storage.  qkvz_plan must have
 * shape (M,N,K)=(512,16384,5120), and output_plan must have
 * (512,5120,6144).  The call is allocation-free and has no M=1 fallback.
 */
typedef struct q27_gdn_prefill_m512_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t has_input_residual;
  const void* input_hidden_bf16;       /* [512,5120] */
  const void* input_residual_bf16;     /* optional [512,5120] */
  const void* input_norm_weight_bf16;  /* [5120] Gemma weight */
  const void* post_norm_weight_bf16;   /* [5120] Gemma weight */
  float norm_epsilon;
  uint32_t reserved;
  const void* qkvz_weight_fp8_e4m3;    /* [16384,5120] */
  uint64_t qkvz_weight_bytes;
  const float* qkvz_input_scale;
  const float* qkvz_weight_scale;
  const void* conv_weight_bf16;        /* [10240,4] */
  const void* merged_ab_weight_bf16;   /* [96,5120] */
  const float* a_log_f32;              /* [48] */
  const float* dt_bias_f32;            /* [48] */
  const void* gdn_norm_weight_bf16;    /* [128] */
  const void* out_weight_fp8_e4m3;     /* [5120,6144] */
  uint64_t out_weight_bytes;
  const float* out_input_scale;
  const float* out_weight_scale;
  void* convolution_state_bf16;        /* [10240,3], updated */
  uint64_t convolution_state_bytes;
  void* recurrent_state_bf16;          /* [48,128,128], updated */
  uint64_t recurrent_state_bytes;
  void* normalized_output_bf16;        /* [512,5120] */
  void* residual_output_bf16;          /* [512,5120] */
  void* scratch;
  uint64_t scratch_bytes;
  q27_prefill_fp8_plan* qkvz_plan;
  q27_prefill_fp8_plan* output_plan;
  void* cublas_handle;
  void* cuda_stream;
} q27_gdn_prefill_m512_args;

q27_gdn_prefill_m512_status q27_gdn_prefill_m512_query(
    q27_gdn_prefill_m512_layout* output);
q27_gdn_prefill_m512_status q27_gdn_prefill_m512_forward(
    const q27_gdn_prefill_m512_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_M512_H_

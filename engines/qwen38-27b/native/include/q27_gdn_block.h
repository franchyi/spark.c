#ifndef SPARKSERVE_Q27_GDN_BLOCK_H_
#define SPARKSERVE_Q27_GDN_BLOCK_H_

#include "q27_bf16_ab.h"
#include "q27_gdn.h"
#include "q27_kernels.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_BLOCK_ABI_VERSION 1u

typedef enum q27_gdn_block_status_code {
  Q27_GDN_BLOCK_OK = 0,
  Q27_GDN_BLOCK_INVALID_ARGUMENT = 1,
  Q27_GDN_BLOCK_PROJECTION_ERROR = 2,
  Q27_GDN_BLOCK_RECURRENT_ERROR = 3,
} q27_gdn_block_status_code;

typedef struct q27_gdn_block_status {
  int32_t code;
  const char* message;
} q27_gdn_block_status;

/*
 * Fixed decode layout for one Qwen3.8-27B GDN layer. State is caller-owned
 * and allocated once per layer. Scratch is caller-owned and may be shared by
 * layers whose launches cannot overlap.
 */
typedef struct q27_gdn_block_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t scratch_bytes;
  uint64_t scratch_alignment;
  uint64_t convolution_state_bytes_per_slot;
  uint64_t recurrent_state_bytes_per_slot;
} q27_gdn_block_layout;

/*
 * One allocation-free, synchronization-free decode attention sublayer:
 *
 *   FP8 QKV + FP8 Z + BF16 A/B
 *     -> causal conv + FlashInfer GDN + gated RMSNorm
 *     -> FP8 output projection
 *
 * All pointers are device pointers except cublas_handle and cuda_stream. FP8
 * weights are raw E4M3 row-major matrices. Every scale is a device FP32
 * scalar exactly as stored by the checkpoint. `a_log_f32` and `dt_bias_f32`
 * are the load-time converted vectors consumed by the recurrent kernel.
 */
typedef struct q27_gdn_block_decode_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t state_slots;
  uint32_t reserved;

  const void* normalized_hidden_bf16; /* BF16 [5120]. */

  const void* qkv_weight_fp8_e4m3;    /* E4M3 [10240,5120]. */
  const float* qkv_input_scale;
  const float* qkv_weight_scale;
  const void* z_weight_fp8_e4m3;      /* E4M3 [6144,5120]. */
  const float* z_input_scale;
  const float* z_weight_scale;
  const void* a_weight_bf16;          /* BF16 [48,5120]. */
  const void* b_weight_bf16;          /* BF16 [48,5120]. */

  const void* conv_weight_bf16;       /* BF16 [10240,4]. */
  const void* norm_weight_bf16;       /* BF16 [128]. */
  const float* a_log_f32;             /* FP32 [48]. */
  const float* dt_bias_f32;           /* FP32 [48]. */

  const void* out_weight_fp8_e4m3;    /* E4M3 [5120,6144]. */
  const float* out_input_scale;
  const float* out_weight_scale;

  void* convolution_state_bf16;
  void* recurrent_state_bf16;
  const int32_t* state_indices_i32;   /* INT32 [1]. */

  void* scratch;
  uint64_t scratch_bytes;
  void* output_bf16;                  /* BF16 [5120]. */
  void* cublas_handle;
  void* cuda_stream;
} q27_gdn_block_decode_args;

q27_gdn_block_status q27_gdn_block_query_layout(
    q27_gdn_block_layout* output);
q27_gdn_block_status q27_gdn_block_decode(
    const q27_gdn_block_decode_args* args);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SPARKSERVE_Q27_GDN_BLOCK_H_ */

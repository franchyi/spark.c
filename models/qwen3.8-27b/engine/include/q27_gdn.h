#ifndef Q27_GDN_H_
#define Q27_GDN_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_ABI_VERSION 1u

enum {
  Q27_GDN_HIDDEN_SIZE = 5120,
  Q27_GDN_QK_HEADS = 16,
  Q27_GDN_VALUE_HEADS = 48,
  Q27_GDN_HEAD_DIM = 128,
  Q27_GDN_QK_WIDTH = 2048,
  Q27_GDN_VALUE_WIDTH = 6144,
  Q27_GDN_CONV_WIDTH = 10240,
  Q27_GDN_CONV_KERNEL = 4,
  Q27_GDN_CONV_HISTORY = 3,
};

typedef enum q27_gdn_status_code {
  Q27_GDN_OK = 0,
  Q27_GDN_INVALID_ARGUMENT = 1,
  Q27_GDN_CUDA_ERROR = 2,
  Q27_GDN_FLASHINFER_ERROR = 3,
} q27_gdn_status_code;

typedef struct q27_gdn_status {
  int32_t code;
  const char* message;
} q27_gdn_status;

typedef struct q27_gdn_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t convolution_state_bytes_per_slot;
  uint64_t recurrent_state_bytes_per_slot;
  uint64_t projected_qkv_bytes;
  uint64_t recurrent_output_bytes;
} q27_gdn_layout;

/*
 * Convert the checkpoint's BF16 A_log/dt_bias vectors to the FP32 vectors
 * consumed by the pinned FlashInfer recurrence. This is a model-load
 * operation, not part of decode or CUDA graph replay.
 */
typedef struct q27_gdn_convert_parameters_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* a_log_bf16;     /* BF16 [48], device pointer. */
  const void* dt_bias_bf16;   /* BF16 [48], device pointer. */
  float* a_log_f32;           /* FP32 [48], device pointer. */
  float* dt_bias_f32;         /* FP32 [48], device pointer. */
  void* cuda_stream;
} q27_gdn_convert_parameters_args;

/*
 * One Qwen3.8-27B linear-attention decode step after projection.
 *
 * The three FP8 projections (QKV, Z, and output) deliberately remain outside
 * this ABI. The two small A/B projections are BF16. All pointers are device
 * pointers, all storage is caller-owned, and the launch performs no allocation
 * or synchronization. The fixed arithmetic is:
 *
 *   causal-conv(Q|K|V) -> FlashInfer GDN recurrence -> gated RMSNorm
 *
 * `projected_qkv_bf16` is [Q(2048),K(2048),V(6144)]. The convolution and
 * recurrent state pools are indexed by `state_indices_i32[0]`, which is a
 * device INT32 pointer. Active indices must be in [0,state_slots).
 */
typedef struct q27_gdn_decode_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t state_slots;
  uint32_t reserved;
  const void* projected_qkv_bf16; /* BF16 [10240]. */
  const void* projected_z_bf16;   /* BF16 [6144]. */
  const void* projected_a_bf16;   /* BF16 [48]. */
  const void* projected_b_bf16;   /* BF16 [48]. */
  const void* conv_weight_bf16;   /* BF16 [10240,4]. */
  const void* norm_weight_bf16;   /* BF16 [128]. */
  const float* a_log_f32;         /* FP32 [48]. */
  const float* dt_bias_f32;       /* FP32 [48]. */
  void* convolution_state_bf16;   /* BF16 [state_slots,10240,3]. */
  void* recurrent_state_bf16;     /* BF16 [state_slots,48,128,128]. */
  const int32_t* state_indices_i32; /* INT32 [1]. */
  void* convolved_qkv_bf16;       /* BF16 [10240] scratch. */
  void* recurrent_output_bf16;    /* BF16 [6144] scratch. */
  void* normalized_output_bf16;   /* BF16 [6144], for FP8 out projection. */
  void* cuda_stream;
} q27_gdn_decode_args;

q27_gdn_status q27_gdn_query_layout(q27_gdn_layout* output);
q27_gdn_status q27_gdn_convert_parameters(
    const q27_gdn_convert_parameters_args* args);
q27_gdn_status q27_gdn_decode(const q27_gdn_decode_args* args);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* Q27_GDN_H_ */

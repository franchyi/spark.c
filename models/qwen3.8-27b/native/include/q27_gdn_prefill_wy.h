#ifndef Q27_GDN_PREFILL_WY_H_
#define Q27_GDN_PREFILL_WY_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_PREFILL_WY_ABI_VERSION 1u

enum {
  Q27_GDN_WY_TOKENS = 128,
  Q27_GDN_WY_CHUNK = 64,
  Q27_GDN_WY_CHUNKS = 2,
  Q27_GDN_WY_QK_HEADS = 16,
  Q27_GDN_WY_VALUE_HEADS = 48,
  Q27_GDN_WY_DIM = 128,
};

typedef enum q27_gdn_prefill_wy_status_code {
  Q27_GDN_PREFILL_WY_OK = 0,
  Q27_GDN_PREFILL_WY_INVALID_ARGUMENT = 1,
  Q27_GDN_PREFILL_WY_CUDA_ERROR = 2,
  Q27_GDN_PREFILL_WY_CUBLAS_ERROR = 3,
} q27_gdn_prefill_wy_status_code;

typedef struct q27_gdn_prefill_wy_status {
  int32_t code;
  const char* message;
} q27_gdn_prefill_wy_status;

typedef struct q27_gdn_prefill_wy_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t qk_bytes;
  uint64_t value_bytes;
  uint64_t solved_a_bytes;
  uint64_t intra_scratch_bytes;
  uint64_t output_scratch_bytes;
  uint64_t scratch_alignment;
} q27_gdn_prefill_wy_layout;

/* BF16 L2 normalization for Q/K [128,16,128], epsilon 1e-6. */
typedef struct q27_gdn_prefill_l2norm_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* input_bf16;
  uint64_t input_bytes;
  void* output_bf16;
  uint64_t output_bytes;
  void* cuda_stream;
} q27_gdn_prefill_l2norm_args;

/*
 * Exact formula contract of c427 chunk_gated_delta_rule_fwd_intra.
 * k is normalized [128,16,128], v [128,48,128], cumulative_g/beta
 * [128,48] FP32 (beta is BF16-rounded FP32). Outputs w/u are
 * [128,48,128] BF16 and solved_a is physical [2,48,64,64] BF16.
 */
typedef struct q27_gdn_prefill_intra_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* k_bf16;
  uint64_t k_bytes;
  const void* v_bf16;
  uint64_t v_bytes;
  const float* cumulative_g_f32;
  uint64_t cumulative_g_bytes;
  const float* beta_f32;
  uint64_t beta_bytes;
  void* solved_a_bf16;
  uint64_t solved_a_bytes;
  void* w_bf16;
  uint64_t w_bytes;
  void* u_bf16;
  uint64_t u_bytes;
  void* scratch;
  uint64_t scratch_bytes;
  void* cublas_handle;
  void* cuda_stream;
} q27_gdn_prefill_intra_args;

/*
 * c427 chunk_fwd_o contract. q/k are normalized [128,16,128], v_new is the
 * ungated donor residual [128,48,128], and chunk_states is the BF16 state
 * before each chunk [2,48,128,128]. Output is [128,48,128] BF16. The fixed
 * scale is 1/sqrt(128).
 */
typedef struct q27_gdn_prefill_output_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* q_bf16;
  uint64_t q_bytes;
  const void* k_bf16;
  uint64_t k_bytes;
  const void* v_new_bf16;
  uint64_t v_new_bytes;
  const void* chunk_states_bf16;
  uint64_t chunk_states_bytes;
  const float* cumulative_g_f32;
  uint64_t cumulative_g_bytes;
  void* recurrent_output_bf16;
  uint64_t recurrent_output_bytes;
  void* scratch;
  uint64_t scratch_bytes;
  void* cublas_handle;
  void* cuda_stream;
} q27_gdn_prefill_output_args;

q27_gdn_prefill_wy_status q27_gdn_prefill_wy_query(
    q27_gdn_prefill_wy_layout* output);
q27_gdn_prefill_wy_status q27_gdn_prefill_l2norm(
    const q27_gdn_prefill_l2norm_args* args);
q27_gdn_prefill_wy_status q27_gdn_prefill_intra(
    const q27_gdn_prefill_intra_args* args);
q27_gdn_prefill_wy_status q27_gdn_prefill_recurrent_output(
    const q27_gdn_prefill_output_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_WY_H_

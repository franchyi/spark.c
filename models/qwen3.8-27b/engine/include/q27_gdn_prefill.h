#ifndef Q27_GDN_PREFILL_H_
#define Q27_GDN_PREFILL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_PREFILL_ABI_VERSION 2u

enum {
  Q27_GDN_PREFILL_TOKENS = 128,
  Q27_GDN_PREFILL_CHUNK = 64,
  Q27_GDN_PREFILL_CHUNKS = 2,
  Q27_GDN_PREFILL_QK_HEADS = 16,
  Q27_GDN_PREFILL_VALUE_HEADS = 48,
  Q27_GDN_PREFILL_HEAD_DIM = 128,
  Q27_GDN_PREFILL_CONV_WIDTH = 10240,
  Q27_GDN_PREFILL_CONV_KERNEL = 4,
  Q27_GDN_PREFILL_CONV_HISTORY = 3,
};

/*
 * Every launch keeps the fixed 128-row physical allocation. valid_tokens is
 * the logical row count in [1,128]; rows at or above it are zero padding and
 * cannot update convolution/recurrent state.
 */

typedef enum q27_gdn_prefill_status_code {
  Q27_GDN_PREFILL_OK = 0,
  Q27_GDN_PREFILL_INVALID_ARGUMENT = 1,
  Q27_GDN_PREFILL_CUDA_ERROR = 2,
  Q27_GDN_PREFILL_CUBLAS_ERROR = 3,
  Q27_GDN_PREFILL_UNIMPLEMENTED = 4,
} q27_gdn_prefill_status_code;

typedef struct q27_gdn_prefill_status {
  int32_t code;
  const char* message;
} q27_gdn_prefill_status;

typedef struct q27_gdn_prefill_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t tokens;
  uint32_t chunk_size;
  uint32_t chunk_count;
  uint32_t reserved;
  uint64_t convolution_state_bytes;
  uint64_t recurrent_state_bytes;
  uint64_t mixed_qkv_bytes;
  uint64_t qk_bytes;
  uint64_t value_bytes;
  uint64_t gate_input_bytes;
  uint64_t gate_output_bytes;
  uint64_t chunk_states_bytes;
  uint64_t v_new_bytes;
  uint64_t chunk_scratch_bytes;
  uint64_t scratch_alignment;
} q27_gdn_prefill_layout;

/*
 * Fixed M=128 causal convolution. Input/output are [128,10240] BF16;
 * weights are [10240,4] BF16; state is [10240,3] BF16 and is updated in
 * place to the final three-token window. One kernel owns the full chunk.
 */
typedef struct q27_gdn_prefill_conv_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* mixed_qkv_bf16;
  uint64_t mixed_qkv_bytes;
  const void* conv_weight_bf16;
  uint64_t conv_weight_bytes;
  void* convolution_state_bf16;
  uint64_t convolution_state_bytes;
  void* convolved_qkv_bf16;
  uint64_t convolved_qkv_bytes;
  void* cuda_stream;
} q27_gdn_prefill_conv_args;

/*
 * Compute SGLang's log-space forget gate and beta, then the chunk-local
 * cumulative log gate used by its Triton recurrence. a/b are [128,48] BF16;
 * A_log/dt_bias are [48] FP32; outputs are [128,48] FP32. Beta values are
 * BF16-rounded then represented in FP32, matching fused_gdn_gating.py. The
 * cumulative sum resets at token 64.
 */
typedef struct q27_gdn_prefill_gate_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* projected_a_bf16;
  uint64_t projected_a_bytes;
  const void* projected_b_bf16;
  uint64_t projected_b_bytes;
  const float* a_log_f32;
  const float* dt_bias_f32;
  float* cumulative_g_f32;
  uint64_t cumulative_g_bytes;
  float* beta_f32;
  uint64_t beta_bytes;
  void* cuda_stream;
} q27_gdn_prefill_gate_args;

/*
 * Highest-cost BF16-state chunk recurrence translated from SGLang's Triton
 * chunk_delta_h kernel. Inputs are the exact outputs of its intra-chunk stage:
 *   k [128,16,128] BF16
 *   w [128,48,128] BF16
 *   u [128,48,128] BF16
 *   cumulative_g [128,48] FP32 (chunk-local log cumulative sum)
 * recurrent_state is [48,128,128] BF16 in [H,V,K] order and is updated in
 * place. chunk_states is [2,48,128,128] BF16 (state before each chunk), and
 * v_new is the ungated [128,48,128] BF16 residual for the following
 * chunk-output kernel; the state update uses a separate gated scratch copy.
 *
 * The cuBLAS handle must use host pointer mode, be warmed, and not be shared
 * concurrently. scratch is caller-owned, fixed-address, and 256-byte aligned.
 */
typedef struct q27_gdn_prefill_chunk_state_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* k_bf16;
  uint64_t k_bytes;
  const void* w_bf16;
  uint64_t w_bytes;
  const void* u_bf16;
  uint64_t u_bytes;
  const float* cumulative_g_f32;
  uint64_t cumulative_g_bytes;
  void* recurrent_state_bf16;
  uint64_t recurrent_state_bytes;
  void* chunk_states_bf16;
  uint64_t chunk_states_bytes;
  void* v_new_bf16;
  uint64_t v_new_bytes;
  void* scratch;
  uint64_t scratch_bytes;
  void* cublas_handle;
  void* cuda_stream;
} q27_gdn_prefill_chunk_state_args;

/* Gated RMSNorm over [128,48,128] BF16, weight [128] BF16. */
typedef struct q27_gdn_prefill_norm_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* recurrent_output_bf16;
  uint64_t recurrent_output_bytes;
  const void* projected_z_bf16;
  uint64_t projected_z_bytes;
  const void* norm_weight_bf16;
  uint64_t norm_weight_bytes;
  void* normalized_output_bf16;
  uint64_t normalized_output_bytes;
  void* cuda_stream;
} q27_gdn_prefill_norm_args;

q27_gdn_prefill_status q27_gdn_prefill_query(
    uint32_t tokens, q27_gdn_prefill_layout* output);
q27_gdn_prefill_status q27_gdn_prefill_causal_conv(
    const q27_gdn_prefill_conv_args* args);
q27_gdn_prefill_status q27_gdn_prefill_prepare_gates(
    const q27_gdn_prefill_gate_args* args);
q27_gdn_prefill_status q27_gdn_prefill_chunk_state(
    const q27_gdn_prefill_chunk_state_args* args);
q27_gdn_prefill_status q27_gdn_prefill_gated_norm(
    const q27_gdn_prefill_norm_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_H_

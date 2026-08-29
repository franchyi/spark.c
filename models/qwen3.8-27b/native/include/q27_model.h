#ifndef Q27_MODEL_H_
#define Q27_MODEL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_MODEL_ABI_VERSION 1u

enum {
  Q27_MODEL_LAYERS = 64,
  Q27_MODEL_GDN_LAYERS = 48,
  Q27_MODEL_ATTENTION_LAYERS = 16,
};

typedef enum q27_model_status_code {
  Q27_MODEL_OK = 0,
  Q27_MODEL_INVALID_ARGUMENT = 1,
  Q27_MODEL_OUT_OF_MEMORY = 2,
  Q27_MODEL_CUDA_ERROR = 3,
  Q27_MODEL_KERNEL_ERROR = 4,
} q27_model_status_code;

typedef struct q27_model_status {
  int32_t code;
  const char* message;
} q27_model_status;

/*
 * All fields are device-visible addresses. Rust derives them from the strict,
 * revision-locked safetensors plan and q27 scale sidecar. Unused GDN or full
 * attention fields must be null; the fixed layer index selects the branch.
 */
typedef struct q27_model_layer_weights {
  const void* input_norm_bf16;
  const void* post_attention_norm_bf16;

  const void* mlp_gate_weight_fp4;
  const void* mlp_gate_scales_fp8_128x4;
  const float* mlp_gate_alpha;
  const float* mlp_hidden_scale_inv;
  const void* mlp_up_weight_fp4;
  const void* mlp_up_scales_fp8_128x4;
  const float* mlp_up_alpha;
  const void* mlp_down_weight_fp4;
  const void* mlp_down_scales_fp8_128x4;
  const float* mlp_down_alpha;
  const float* mlp_activated_scale_inv;

  const void* gdn_qkv_weight_fp8;
  const float* gdn_qkv_input_scale;
  const float* gdn_qkv_weight_scale;
  const void* gdn_z_weight_fp8;
  const float* gdn_z_input_scale;
  const float* gdn_z_weight_scale;
  const void* gdn_a_weight_bf16;
  const void* gdn_b_weight_bf16;
  const void* gdn_conv_weight_bf16;
  const void* gdn_norm_weight_bf16;
  const void* gdn_a_log_bf16;
  const void* gdn_dt_bias_bf16;
  const void* gdn_out_weight_fp8;
  const float* gdn_out_input_scale;
  const float* gdn_out_weight_scale;

  const void* attention_q_weight_fp8;
  const float* attention_q_input_scale;
  const float* attention_q_weight_scale;
  const void* attention_k_weight_fp8;
  const float* attention_k_input_scale;
  const float* attention_k_weight_scale;
  const void* attention_v_weight_fp8;
  const float* attention_v_input_scale;
  const float* attention_v_weight_scale;
  const void* attention_o_weight_fp8;
  const float* attention_o_input_scale;
  const float* attention_o_weight_scale;
  const void* attention_q_norm_bf16;
  const void* attention_k_norm_bf16;
} q27_model_layer_weights;

typedef struct q27_model_weights {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* embedding_bf16;
  const void* final_norm_bf16;
  const void* lm_head_bf16;
  const q27_model_layer_weights* layers;
  uint32_t layer_count;
  uint32_t reserved;
} q27_model_weights;

typedef struct q27_model_options {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t context_capacity;
  uint32_t device_id;
} q27_model_options;

typedef struct q27_model_stats {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t resident_weight_bytes;
  uint64_t state_bytes;
  uint64_t scratch_bytes;
  uint32_t context_capacity;
  uint32_t position;
  uint64_t last_decode_us;
} q27_model_stats;

typedef struct q27_model q27_model;

/* One model, one slot. No allocation occurs in prefill or decode. */
q27_model_status q27_model_create(const q27_model_weights* weights,
                                  const q27_model_options* options,
                                  q27_model** output);
q27_model_status q27_model_reset(q27_model* model);
/*
 * Advance one prompt token and its recurrent/KV state without producing
 * logits. Work is ordered on the model stream; deferred CUDA errors are
 * reported by the next synchronizing call (normally decode_greedy).
 */
q27_model_status q27_model_consume_token(q27_model* model, uint32_t token);
/*
 * Reset and prefill one host sequence in physical M=128 tiles.
 * Only one 128-token host tile is copied at a time. Intermediate tiles skip
 * the LM head; the final tile returns the first greedy completion token.
 */
q27_model_status q27_model_prefill_greedy(q27_model* model,
                                          const uint32_t* host_tokens,
                                          uint32_t count,
                                          uint32_t* output_token);
q27_model_status q27_model_decode_greedy(q27_model* model, uint32_t token,
                                         uint32_t* output_token);
/* Diagnostic-only post-step copy; the decode hot path never performs it. */
q27_model_status q27_model_copy_logits(const q27_model* model,
                                       float* host_logits,
                                       uint32_t elements);
q27_model_status q27_model_get_stats(const q27_model* model,
                                     q27_model_stats* output);
q27_model_status q27_model_destroy(q27_model* model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_MODEL_H_

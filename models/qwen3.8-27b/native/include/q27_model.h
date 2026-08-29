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
  Q27_MODEL_DFLASH2_BLOCK_SIZE = 8,
  Q27_MODEL_DFLASH2_TARGET_FEATURES = 5,
  Q27_MODEL_DFLASH2_HIDDEN_SIZE = 5120,
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

/*
 * Host-visible result of one greedy DFlash2 target transaction. target_top1
 * contains all eight target predictions. committed_tokens contains the
 * accepted draft prefix followed by the target bonus; entries at or after
 * commit_length are zero. target_features_bf16 is a model-owned device view
 * with fixed layout [8,5,5120] BF16 and remains valid until the next verify
 * transaction or model destruction.
 */
typedef struct q27_model_dflash2_verify_result {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t base_position;
  uint32_t new_position;
  uint32_t accept_length;  /* Accepted draft tokens, 0..7. */
  uint32_t commit_length;  /* Consumed target rows / emitted tokens, 1..8. */
  uint32_t bonus_token;
  uint32_t reserved;
  uint32_t target_top1[Q27_MODEL_DFLASH2_BLOCK_SIZE];
  uint32_t committed_tokens[Q27_MODEL_DFLASH2_BLOCK_SIZE];
  const void* target_features_bf16;
  uint64_t target_features_bytes;
} q27_model_dflash2_verify_result;

#define Q27_MODEL_DFLASH2_PROFILE_ABI_VERSION 1u

/*
 * Optional CUDA-event timings for the most recent successful target verify.
 * Events are created once with the model when Q27_DFLASH2_PROFILE=1. All
 * values remain zero and valid remains false in the default runtime.
 */
typedef struct q27_model_dflash2_profile_stats {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t total_us;
  uint64_t snapshot_us;
  uint64_t speculative_pass_us;
  uint64_t speculative_result_sync_us;
  uint64_t rollback_us;
  uint64_t committed_replay_us;
  uint64_t committed_result_sync_us;
  uint32_t enabled;
  uint32_t valid;
} q27_model_dflash2_profile_stats;

/*
 * Borrowed view published once per target prompt tile. The callback runs on
 * the model owner thread after the tile has been enqueued. It may enqueue
 * draft context projection/KV work on the supplied stream and cuBLAS handle;
 * it must not allocate, synchronize, or retain target_features_bf16.
 */
typedef struct q27_model_dflash2_feature_batch {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* target_features_bf16; /* [token_count,5,5120], device BF16. */
  uint32_t token_count;
  uint32_t first_position;
  void* cublas_handle;
  void* cuda_stream;
} q27_model_dflash2_feature_batch;

typedef q27_model_status (*q27_model_dflash2_feature_sink)(
    const q27_model_dflash2_feature_batch* batch, void* user_data);

/* Read-only target tensors shared with the model-specific DFlash2 capsule. */
typedef struct q27_model_dflash2_runtime_view {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* embedding_bf16; /* [248320,5120]. */
  const void* lm_head_bf16;   /* Resident, cuBLAS-aligned [248320,5120]. */
  uint32_t vocabulary;
  uint32_t hidden_size;
} q27_model_dflash2_runtime_view;

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
 * Reset and prefill one host sequence largest-first with M512 then M128
 * prompt lanes. The experimental M2048 lane is selected only when
 * Q27_PREFILL_M2048=1; its first GB10 promotion canary did not clear M512.
 * Only one tile is copied at a time. Intermediate tiles skip the LM head; the
 * final tile returns the first greedy completion token.
 */
q27_model_status q27_model_prefill_greedy(q27_model* model,
                                          const uint32_t* host_tokens,
                                          uint32_t count,
                                          uint32_t* output_token);
/*
 * Target prefill plus streaming DFlash2 feature publication. Each tile is
 * consumed by sink before its model-owned feature buffer is reused. A null
 * sink is rejected; plain target-only callers use q27_model_prefill_greedy.
 */
q27_model_status q27_model_prefill_dflash2(
    q27_model* model, const uint32_t* host_tokens, uint32_t count,
    q27_model_dflash2_feature_sink sink, void* sink_user_data,
    uint32_t* output_token);
q27_model_status q27_model_get_dflash2_runtime_view(
    const q27_model* model, q27_model_dflash2_runtime_view* output);
q27_model_status q27_model_decode_greedy(q27_model* model, uint32_t token,
                                         uint32_t* output_token);
/*
 * Fixed greedy target verification for candidates [anchor,draft x7]. The
 * call snapshots live GDN state, performs one valid_tokens=8 target forward,
 * publishes all target top-1 rows and five post-layer feature taps, computes
 * greedy acceptance, restores the snapshot, then reruns candidates[0..commit)
 * to make only the accepted target prefix live. Attention KV rows are
 * append-only and the rejected physical rows remain hidden by model position.
 *
 * The transaction allocates nothing but synchronizes to return host-visible
 * acceptance metadata. It is batch-one and temperature-zero by design.
 */
q27_model_status q27_model_dflash2_verify(
    q27_model* model,
    const uint32_t host_candidates[Q27_MODEL_DFLASH2_BLOCK_SIZE],
    q27_model_dflash2_verify_result* output);
q27_model_status q27_model_get_dflash2_profile_stats(
    const q27_model* model, q27_model_dflash2_profile_stats* output);
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

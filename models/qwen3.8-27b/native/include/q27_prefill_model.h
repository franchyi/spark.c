#ifndef Q27_PREFILL_MODEL_H_
#define Q27_PREFILL_MODEL_H_

#include <stdint.h>

#include "q27_prefill_attention_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_PREFILL_MODEL_ABI_VERSION 3u

enum {
  Q27_PREFILL_MODEL_LAYERS = 64,
  Q27_PREFILL_MODEL_GDN_LAYERS = 48,
  Q27_PREFILL_MODEL_ATTENTION_LAYERS = 16,
  Q27_PREFILL_MODEL_TOKENS = 128,
  Q27_PREFILL_MODEL_M512_TOKENS = 512,
  Q27_PREFILL_MODEL_M2048_TOKENS = 2048,
  Q27_PREFILL_MODEL_M4096_TOKENS = 4096,
  Q27_PREFILL_MODEL_M8192_TOKENS = 8192,
  Q27_PREFILL_MODEL_HIDDEN = 5120,
  Q27_PREFILL_MODEL_VOCAB = 248320,
  Q27_PREFILL_MODEL_GDN = 0,
  Q27_PREFILL_MODEL_ATTENTION = 1,
  Q27_PREFILL_MODEL_OUTPUT_NONE = 0,
  Q27_PREFILL_MODEL_OUTPUT_LAST = 1,
  Q27_PREFILL_MODEL_OUTPUT_ALL_ROWS = 2,
};

typedef enum q27_prefill_model_status_code {
  Q27_PREFILL_MODEL_OK = 0,
  Q27_PREFILL_MODEL_INVALID_ARGUMENT = 1,
  Q27_PREFILL_MODEL_CUDA_ERROR = 2,
  Q27_PREFILL_MODEL_CAPSULE_ERROR = 3,
  Q27_PREFILL_MODEL_INTERNAL_ERROR = 4,
} q27_prefill_model_status_code;

typedef struct q27_prefill_model_status {
  int32_t code;
  const char* message;
} q27_prefill_model_status;

typedef struct q27_prefill_model_config {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t cache_capacity;
  uint32_t fast_accum;
  uint64_t fp8_workspace_bytes;
} q27_prefill_model_config;

/* Caller-owned hot-call arena. Every offset is 256-byte aligned. */
typedef struct q27_prefill_model_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t scratch_bytes;
  uint64_t scratch_alignment;
  uint64_t hidden_offset;
  uint64_t normalized_offset;
  uint64_t residual_a_offset;
  uint64_t residual_b_offset;
  uint64_t last_hidden_offset;
  uint64_t logits_offset;
  uint64_t argmax_values_offset;
  uint64_t argmax_indices_offset;
  uint64_t invalid_count_offset;
  uint64_t shared_offset;
  uint64_t shared_bytes;
  uint64_t gdn_state_bytes_per_layer;
  uint64_t gdn_conv_bytes_per_layer;
  uint64_t attention_cache_bytes_per_layer;
} q27_prefill_model_layout;

typedef struct q27_prefill_model_mlp_weights {
  const void* gate_up_weight_fp4_e2m1;       /* [34816,5120] packed */
  uint64_t gate_up_weight_bytes;
  const void* gate_up_scales_e4m3_128x4;
  uint64_t gate_up_scale_bytes;
  const float* hidden_global_scale_inv;
  const float* gate_up_alpha;
  const void* down_weight_fp4_e2m1;          /* [5120,17408] packed */
  uint64_t down_weight_bytes;
  const void* down_scales_e4m3_128x4;
  uint64_t down_scale_bytes;
  const float* activated_global_scale_inv;
  const float* down_alpha;
} q27_prefill_model_mlp_weights;

/* Fused/merged tensors are load-time products; the hot path never merges. */
typedef struct q27_prefill_model_gdn_layer {
  const void* qkvz_weight_fp8_e4m3;          /* [16384,5120] */
  uint64_t qkvz_weight_bytes;
  const float* qkvz_input_scale;
  const float* qkvz_weight_scale;
  const void* conv_weight_bf16;              /* [10240,4] */
  const void* merged_ab_weight_bf16;         /* [96,5120] */
  const float* a_log_f32;                    /* [48] */
  const float* dt_bias_f32;                  /* [48] */
  const void* gdn_norm_weight_bf16;          /* [128] */
  const void* out_weight_fp8_e4m3;           /* [5120,6144] */
  uint64_t out_weight_bytes;
  const float* out_input_scale;
  const float* out_weight_scale;
  void* convolution_state_bf16;              /* private to this layer */
  uint64_t convolution_state_bytes;
  void* recurrent_state_bf16;                /* private to this layer */
  uint64_t recurrent_state_bytes;
} q27_prefill_model_gdn_layer;

typedef struct q27_prefill_model_attention_layer {
  q27_prefill_attention_layer_weights weights;
  const int32_t* block_table_i32;
  uint32_t block_table_entries;
  uint32_t reserved;
  void* key_cache_fp8_e4m3;                  /* private to this layer */
  void* value_cache_fp8_e4m3;                /* private to this layer */
  float key_cache_scale;
  float value_cache_scale;
} q27_prefill_model_attention_layer;

typedef struct q27_prefill_model_layer {
  uint32_t kind;
  uint32_t reserved0;
  const void* input_norm_bf16;
  const void* post_attention_norm_bf16;
  q27_prefill_model_mlp_weights mlp;
  q27_prefill_model_gdn_layer gdn;
  q27_prefill_model_attention_layer attention;
} q27_prefill_model_layer;

typedef struct q27_prefill_model_plan q27_prefill_model_plan;

/*
 * One exact fixed-M128 target tile. valid_tokens masks the final prompt tile.
 * Layer i is attention iff (i+1)%4==0. The call performs embedding once,
 * all 64 target layers plus their dense MLPs, final norm, last-valid-row LM
 * head, and deterministic argmax. It allocates and synchronizes nothing.
 *
 * If target_features_bf16 is non-null, the call publishes the logical
 * post-layer BF16 hidden state after zero-based target layers 5, 19, 33, 47,
 * and 61. The fixed layout is [valid_tokens,5,5120], token-major. A logical
 * post-layer state is the BF16-rounded sum of the layer MLP output and its
 * residual, exactly matching the state observed before the next target layer.
 */
typedef struct q27_prefill_model_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t committed_tokens;
  const uint32_t* token_ids_u32;             /* device [valid_tokens] */
  const void* embedding_bf16;                /* [248320,5120] */
  const void* final_norm_bf16;               /* [5120] */
  const void* lm_head_bf16;                  /* [248320,5120] */
  const q27_prefill_model_layer* layers;      /* host-visible [64] */
  uint32_t layer_count;
  uint32_t produce_output;                   /* Q27_PREFILL_MODEL_OUTPUT_* */
  const float* rope_cos_sin_f32;
  uint64_t rope_row_stride_elements;
  uint32_t rope_position_capacity;
  uint32_t reserved2;
  void* output_token_i32;                    /* device scalar */
  void* scratch;
  uint64_t scratch_bytes;
  void* cuda_stream;
  void* output_top1_i32;                     /* device [valid_tokens], ALL_ROWS */
  uint64_t output_top1_bytes;
  void* target_features_bf16;                /* optional [valid_tokens,5,5120] */
  uint64_t target_features_bytes;
  uint32_t verify_t8_gdn;
  uint32_t reserved3;
  void* gdn_checkpoint_convolution_bf16;      /* [48,8,10240,3] */
  uint64_t gdn_checkpoint_convolution_bytes;
  void* gdn_checkpoint_recurrent_bf16;        /* [48,8,48,128,128] */
  uint64_t gdn_checkpoint_recurrent_bytes;
  const int32_t* gdn_state_index_i32;         /* device scalar zero */
} q27_prefill_model_args;

q27_prefill_model_status q27_prefill_model_query(
    const q27_prefill_model_config* config, q27_prefill_model_layout* output);
/*
 * M512 creates a distinct fixed-shape plan/layout. Use it for full prompt
 * chunks, preserving the M128 plan/forward above for the final or verifier
 * chunk. Both lanes update the same caller-owned per-layer KV/GDN state.
 */
q27_prefill_model_status q27_prefill_model_query_m512(
    const q27_prefill_model_config* config, q27_prefill_model_layout* output);
q27_prefill_model_status q27_prefill_model_query_m2048(
    const q27_prefill_model_config* config, q27_prefill_model_layout* output);
q27_prefill_model_status q27_prefill_model_query_m4096(
    const q27_prefill_model_config* config, q27_prefill_model_layout* output);
q27_prefill_model_status q27_prefill_model_query_m8192(
    const q27_prefill_model_config* config, q27_prefill_model_layout* output);
q27_prefill_model_status q27_prefill_model_plan_create(
    const q27_prefill_model_config* config, q27_prefill_model_plan** output);
q27_prefill_model_status q27_prefill_model_plan_create_m512(
    const q27_prefill_model_config* config, q27_prefill_model_plan** output);
q27_prefill_model_status q27_prefill_model_plan_create_m2048(
    const q27_prefill_model_config* config, q27_prefill_model_plan** output);
q27_prefill_model_status q27_prefill_model_plan_create_m4096(
    const q27_prefill_model_config* config, q27_prefill_model_plan** output);
q27_prefill_model_status q27_prefill_model_plan_create_m8192(
    const q27_prefill_model_config* config, q27_prefill_model_plan** output);
void q27_prefill_model_plan_destroy(q27_prefill_model_plan* plan);
q27_prefill_model_status q27_prefill_model_forward(
    q27_prefill_model_plan* plan, const q27_prefill_model_args* args);
q27_prefill_model_status q27_prefill_model_forward_m512(
    q27_prefill_model_plan* plan, const q27_prefill_model_args* args);
q27_prefill_model_status q27_prefill_model_forward_m2048(
    q27_prefill_model_plan* plan, const q27_prefill_model_args* args);
q27_prefill_model_status q27_prefill_model_forward_m4096(
    q27_prefill_model_plan* plan, const q27_prefill_model_args* args);
q27_prefill_model_status q27_prefill_model_forward_m8192(
    q27_prefill_model_plan* plan, const q27_prefill_model_args* args);

/* Device diagnostics inside a valid arena, readable after stream completion. */
uint32_t* q27_prefill_model_invalid_count(
    const q27_prefill_model_layout* layout, void* scratch);
const float* q27_prefill_model_logits(
    const q27_prefill_model_layout* layout, const void* scratch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_PREFILL_MODEL_H_

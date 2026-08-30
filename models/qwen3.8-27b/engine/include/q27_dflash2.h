#ifndef Q27_DFLASH2_H_
#define Q27_DFLASH2_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Revision-locked native contract for:
 *   z-lab/Qwen3.8-27B-DFlash2
 *   revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
 *
 * This is deliberately not a generic speculative-decoding ABI.  It fixes the
 * model dimensions and the eight-token DFlash block used by the pinned Spark
 * recipe.  All tensor addresses below are device-visible unless a field says
 * otherwise.  The capsule owns no Python, Torch, SGLang, or vLLM objects.
 */
#define Q27_DFLASH2_ABI_VERSION 1u

enum {
  /* Personal Spark v1: one request and one persistent draft-cache slot. */
  Q27_DFLASH2_MAX_BATCH = 1,
  Q27_DFLASH2_BLOCK_SIZE = 8,
  Q27_DFLASH2_DRAFT_TOKENS = 7,
  Q27_DFLASH2_LAYERS = 5,
  Q27_DFLASH2_TARGET_FEATURES = 5,
  Q27_DFLASH2_HIDDEN_SIZE = 5120,
  Q27_DFLASH2_INTERMEDIATE_SIZE = 17408,
  Q27_DFLASH2_QUERY_HEADS = 32,
  Q27_DFLASH2_KV_HEADS = 8,
  Q27_DFLASH2_HEAD_DIM = 128,
  Q27_DFLASH2_CONV_TAPS = 2,
  Q27_DFLASH2_CONV_GROUP_SIZE = 16,
  Q27_DFLASH2_CONV_GROUPS = 320,
  Q27_DFLASH2_CONV_PROJECTION_SIZE = 1280,
  Q27_DFLASH2_SELECTOR_RANK = 256,
  Q27_DFLASH2_SELECTOR_TOP_K = 16,
  Q27_DFLASH2_VOCAB_SIZE = 248320,
  Q27_DFLASH2_MASK_TOKEN_ID = 248070,
  Q27_DFLASH2_EOS_TOKEN_ID = 248044,
  Q27_DFLASH2_SLIDING_WINDOW = 2048,
  Q27_DFLASH2_MAX_POSITION = 262144,
};

#define Q27_DFLASH2_ONE_KV_CACHE_BYTES                                    \
  (5ULL * 2048ULL * 8ULL * 128ULL * 2ULL)
#define Q27_DFLASH2_POSITION_TAG_BYTES (2048ULL * 8ULL)

typedef enum q27_dflash2_status_code {
  Q27_DFLASH2_OK = 0,
  Q27_DFLASH2_INVALID_ARGUMENT = 1,
  Q27_DFLASH2_INCOMPATIBLE_CHECKPOINT = 2,
  Q27_DFLASH2_CUDA_ERROR = 3,
  Q27_DFLASH2_UNIMPLEMENTED = 4,
} q27_dflash2_status_code;

typedef struct q27_dflash2_status {
  int32_t code;
  const char* message;
} q27_dflash2_status;

/* A strict row-major BF16 checkpoint tensor and its exact payload size. */
typedef struct q27_dflash2_weight_view {
  const void* data;
  uint64_t bytes;
} q27_dflash2_weight_view;

typedef struct q27_dflash2_layer_weights {
  /* [5120] */
  q27_dflash2_weight_view input_norm;
  /* [2, 2, 5120], [1280, 5120] */
  q27_dflash2_weight_view attention_conv_base;
  q27_dflash2_weight_view attention_conv_projection;

  /* q:[4096,5120], k/v:[1024,5120], o:[5120,4096] */
  q27_dflash2_weight_view q_proj;
  q27_dflash2_weight_view k_proj;
  q27_dflash2_weight_view v_proj;
  q27_dflash2_weight_view o_proj;
  /* [128] per-head RMSNorm weights */
  q27_dflash2_weight_view q_norm;
  q27_dflash2_weight_view k_norm;

  /* [5120] */
  q27_dflash2_weight_view post_attention_norm;
  /* [2, 2, 5120], [1280, 5120] */
  q27_dflash2_weight_view mlp_conv_base;
  q27_dflash2_weight_view mlp_conv_projection;
  /* gate/up:[17408,5120], down:[5120,17408] */
  q27_dflash2_weight_view mlp_gate;
  q27_dflash2_weight_view mlp_up;
  q27_dflash2_weight_view mlp_down;
} q27_dflash2_layer_weights;

typedef struct q27_dflash2_weights {
  uint32_t struct_size;
  uint32_t abi_version;

  /* fc:[5120, 5*5120], hidden_norm/final_norm:[5120] */
  q27_dflash2_weight_view context_projection;
  q27_dflash2_weight_view context_norm;
  q27_dflash2_weight_view final_norm;

  q27_dflash2_layer_weights layers[Q27_DFLASH2_LAYERS];

  /* DFlash2-only: [256,5120], [248320,256], [248320,256]. */
  q27_dflash2_weight_view selector_hidden_projection;
  q27_dflash2_weight_view selector_predecessor_codebook;
  q27_dflash2_weight_view selector_successor_codebook;
} q27_dflash2_weights;

/*
 * Persistent one-slot state.  K/V layout is
 * [layer=5, ring_position=2048, kv_head=8, head_dim=128] BF16.  A position
 * tag distinguishes valid entries after ring wrap.  The target Q27 recurrent
 * state and its eight verify snapshots remain target-capsule state, not draft
 * state.
 */
typedef struct q27_dflash2_state_view {
  uint32_t struct_size;
  uint32_t abi_version;
  void* key_cache_bf16;
  void* value_cache_bf16;
  uint64_t* position_tags_u64;
  void* workspace;
  uint64_t workspace_bytes;
  uint64_t committed_length;
} q27_dflash2_state_view;

/*
 * Target features captured after target layers 5,19,33,47,61, in that exact
 * order. hidden_bf16 is [token_count, 5, 5120]; positions_u64 is [token_count].
 */
typedef struct q27_dflash2_target_features {
  const void* hidden_bf16;
  const uint64_t* positions_u64;
  uint32_t token_count;
  uint32_t reserved;
} q27_dflash2_target_features;

/*
 * Build [bonus, MASK x 7], absolute positions, and ring-cache slots.  Inputs
 * and outputs are contiguous device arrays with the documented shapes.
 */
typedef struct q27_dflash2_prepare_block_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const uint32_t* bonus_tokens;     /* [batch] */
  const uint64_t* prefix_lengths;   /* [batch] */
  uint32_t* block_tokens;           /* [batch, 8] */
  uint64_t* positions;              /* [batch, 8] */
  uint32_t* cache_slots;            /* [batch, 8] */
  uint32_t batch_size;
  uint32_t reserved;
  void* cuda_stream;
} q27_dflash2_prepare_block_args;

/*
 * Greedy DFlash2 selector path. candidate_ids is [batch,7,16] and scores is
 * [batch,7,16,16] FP32.  For slot zero only predecessor row zero is used;
 * later slots use the previously selected top-k index as predecessor row.
 * Equal scores choose the lower top-k index, matching the pinned Triton walk.
 */
typedef struct q27_dflash2_selector_walk_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const uint32_t* candidate_ids;
  const float* scores;
  uint32_t* draft_tokens;       /* [batch, 7] */
  uint32_t* selected_indices;   /* [batch, 7], optional */
  uint32_t batch_size;
  uint32_t reserved;
  void* cuda_stream;
} q27_dflash2_selector_walk_args;

/*
 * candidates is [batch,8]: anchor at column zero, proposed tokens at 1..7.
 * target_top1 is the target's greedy result for all eight verify rows.
 * committed_tokens contains the accepted draft prefix followed by the target
 * bonus; columns at/after commit_length are zero.
 */
typedef struct q27_dflash2_accept_greedy_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const uint32_t* candidates;
  const uint32_t* target_top1;
  const uint64_t* prefix_lengths;
  uint32_t* accept_lengths;     /* [batch], 0..7 */
  uint32_t* commit_lengths;     /* [batch], 1..8 */
  uint32_t* bonus_tokens;       /* [batch] */
  uint32_t* committed_tokens;   /* [batch, 8] */
  uint64_t* new_lengths;        /* [batch] */
  uint32_t batch_size;
  uint32_t reserved;
  void* cuda_stream;
} q27_dflash2_accept_greedy_args;

/* Strictly validate all pointers and byte sizes; this performs no CUDA work. */
q27_dflash2_status q27_dflash2_validate_weights(
    const q27_dflash2_weights* weights);

/* Small graph-safe control launches; no allocation and no synchronization. */
q27_dflash2_status q27_dflash2_prepare_block(
    const q27_dflash2_prepare_block_args* args);
q27_dflash2_status q27_dflash2_selector_walk_greedy(
    const q27_dflash2_selector_walk_args* args);
q27_dflash2_status q27_dflash2_accept_greedy(
    const q27_dflash2_accept_greedy_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_H_

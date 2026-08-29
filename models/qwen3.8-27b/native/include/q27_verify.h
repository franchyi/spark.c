#ifndef Q27_VERIFY_H_
#define Q27_VERIFY_H_

#include "q27_model.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fixed target-verification ABI for the pinned Qwen3.8-27B DFlash2 recipe.
 * This is intentionally not a general speculative-decoding interface.
 *
 * The eight candidate rows are laid out as [anchor, draft x 7].  The target
 * predicts one token after each row.  Greedy verification accepts consecutive
 * candidates[:, 1:] == target_top1[:, :-1], then appends target_top1[accept].
 *
 * All tensor addresses are device-visible.  Control calls allocate nothing,
 * never synchronize, and are safe to place in a fixed-address CUDA graph.
 */
#define Q27_VERIFY_ABI_VERSION 1u

enum {
  Q27_VERIFY_BLOCK_SIZE = 8,
  Q27_VERIFY_DRAFT_TOKENS = 7,
  Q27_VERIFY_MAX_REQUESTS = 8,
  Q27_VERIFY_TARGET_FEATURES = 5,
  Q27_VERIFY_HIDDEN_SIZE = 5120,
  Q27_VERIFY_VOCAB_SIZE = 248320,
  Q27_VERIFY_GDN_LAYERS = 48,
  Q27_VERIFY_ATTENTION_LAYERS = 16,
  Q27_VERIFY_GDN_CONV_WIDTH = 10240,
  Q27_VERIFY_GDN_CONV_HISTORY = 3,
  Q27_VERIFY_GDN_VALUE_HEADS = 48,
  Q27_VERIFY_GDN_HEAD_DIM = 128,
  Q27_VERIFY_KV_HEADS = 4,
  Q27_VERIFY_KV_HEAD_DIM = 256,
};

#define Q27_VERIFY_GDN_CONV_BYTES_PER_REQUEST                              \
  (Q27_VERIFY_GDN_LAYERS * Q27_VERIFY_GDN_CONV_WIDTH *                    \
   Q27_VERIFY_GDN_CONV_HISTORY * 2ULL)
#define Q27_VERIFY_GDN_RECURRENT_BYTES_PER_REQUEST                         \
  (Q27_VERIFY_GDN_LAYERS * Q27_VERIFY_GDN_VALUE_HEADS *                   \
   Q27_VERIFY_GDN_HEAD_DIM * Q27_VERIFY_GDN_HEAD_DIM * 2ULL)
#define Q27_VERIFY_GDN_STATE_BYTES_PER_REQUEST                             \
  (Q27_VERIFY_GDN_CONV_BYTES_PER_REQUEST +                                \
   Q27_VERIFY_GDN_RECURRENT_BYTES_PER_REQUEST)
#define Q27_VERIFY_ONE_KV_ROW_BYTES                                        \
  (Q27_VERIFY_ATTENTION_LAYERS * Q27_VERIFY_KV_HEADS *                    \
   Q27_VERIFY_KV_HEAD_DIM)

typedef enum q27_verify_status_code {
  Q27_VERIFY_OK = 0,
  Q27_VERIFY_INVALID_ARGUMENT = 1,
  Q27_VERIFY_CUDA_ERROR = 2,
  Q27_VERIFY_UNIMPLEMENTED = 3,
} q27_verify_status_code;

typedef enum q27_verify_device_error_code {
  Q27_VERIFY_DEVICE_OK = 0,
  Q27_VERIFY_DEVICE_TOKEN_OUT_OF_RANGE = 1,
  Q27_VERIFY_DEVICE_CONTEXT_OVERFLOW = 2,
  Q27_VERIFY_DEVICE_COMMIT_LENGTH_OUT_OF_RANGE = 3,
  Q27_VERIFY_DEVICE_CHECKPOINT_LENGTH_MISMATCH = 4,
} q27_verify_device_error_code;

typedef struct q27_verify_status {
  int32_t code;
  const char* message;
} q27_verify_status;

/* Exact allocation sizes for a request batch and dense target KV capacity. */
typedef struct q27_verify_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t request_count;
  uint32_t context_capacity;
  uint64_t live_convolution_bytes;
  uint64_t live_recurrent_bytes;
  uint64_t one_key_cache_bytes;
  uint64_t one_value_cache_bytes;
  uint64_t base_convolution_bytes;
  uint64_t base_recurrent_bytes;
  uint64_t checkpoint_convolution_bytes;
  uint64_t checkpoint_recurrent_bytes;
  uint64_t base_length_bytes;
  uint64_t checkpoint_length_bytes;
} q27_verify_layout;

/*
 * Live target state, request-major.  GDN state layouts are:
 *   convolution: [request, 48, 10240, 3] BF16
 *   recurrent:   [request, 48, 48, 128, 128] BF16
 * Dense FP8 K/V layouts are:
 *   [request, 16, context_capacity, 4, 256]
 * lengths_u64 is [request] and is the only visibility boundary for KV rows.
 */
typedef struct q27_verify_target_state {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t request_count;
  uint32_t context_capacity;
  void* convolution_state_bf16;
  uint64_t convolution_state_bytes;
  void* recurrent_state_bf16;
  uint64_t recurrent_state_bytes;
  void* key_cache_fp8_e4m3;
  uint64_t key_cache_bytes;
  void* value_cache_fp8_e4m3;
  uint64_t value_cache_bytes;
  uint64_t* lengths_u64;
} q27_verify_target_state;

/*
 * Base is the pre-verify state.  Checkpoints are request-major with eight
 * states per request; slot i is the state after target row i.  A T=8 GDN
 * kernel should write these intermediate states directly.  The explicit
 * checkpoint snapshot call is also available for a development reference.
 */
typedef struct q27_verify_journal {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t request_count;
  uint32_t reserved;
  void* base_convolution_bf16;
  uint64_t base_convolution_bytes;
  void* base_recurrent_bf16;
  uint64_t base_recurrent_bytes;
  uint64_t* base_lengths_u64;
  uint64_t base_length_bytes;
  void* checkpoint_convolution_bf16;
  uint64_t checkpoint_convolution_bytes;
  void* checkpoint_recurrent_bf16;
  uint64_t checkpoint_recurrent_bytes;
  uint64_t* checkpoint_lengths_u64;
  uint64_t checkpoint_length_bytes;
} q27_verify_journal;

typedef struct q27_verify_state_args {
  uint32_t struct_size;
  uint32_t abi_version;
  q27_verify_target_state* state;
  q27_verify_journal* journal;
  void* cuda_stream;
} q27_verify_state_args;

typedef struct q27_verify_snapshot_checkpoint_args {
  uint32_t struct_size;
  uint32_t abi_version;
  q27_verify_target_state* state;
  q27_verify_journal* journal;
  uint32_t checkpoint_index; /* 0..7, after the corresponding target row. */
  uint32_t reserved;
  void* cuda_stream;
} q27_verify_snapshot_checkpoint_args;

/*
 * candidates and target_top1 are [request,8].  The output contract is the
 * same as pinned SGLang DFlash2 greedy verification.  device_error_u32 is a
 * required device scalar, cleared by the launch and filled asynchronously.
 */
typedef struct q27_verify_accept_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const uint32_t* candidates;
  const uint32_t* target_top1;
  const uint64_t* base_lengths_u64;
  uint32_t* accept_lengths_u32;
  uint32_t* commit_lengths_u32;
  uint32_t* bonus_tokens_u32;
  uint32_t* committed_tokens_u32;
  uint64_t* new_lengths_u64;
  uint32_t* device_error_u32;
  uint32_t request_count;
  uint32_t context_capacity;
  void* cuda_stream;
} q27_verify_accept_args;

/*
 * Select checkpoint commit_lengths[request]-1 and restore it to live GDN
 * state.  KV is append-only: rejected rows stay physically present but the
 * live logical length hides them.  device_error_u32 is cleared by the call.
 */
typedef struct q27_verify_commit_args {
  uint32_t struct_size;
  uint32_t abi_version;
  q27_verify_target_state* state;
  q27_verify_journal* journal;
  const uint32_t* commit_lengths_u32;
  uint32_t* device_error_u32;
  void* cuda_stream;
} q27_verify_commit_args;

/*
 * Fixed-T=8 target-forward seam.  It is fully shape-validated today but
 * returns Q27_VERIFY_UNIMPLEMENTED until the M=8 kernel set listed in
 * native/tools/Q27_VERIFY.md is linked.  Scratch is caller-owned and stable.
 */
typedef struct q27_verify_forward_t8_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_model_weights* weights;
  const uint32_t* candidates_u32; /* [request,8]. */
  q27_verify_target_state* state;
  q27_verify_journal* journal;
  uint32_t* target_top1_u32;      /* [request,8]. */
  void* target_features_bf16;     /* [request,8,5,5120]. */
  uint32_t* device_error_u32;
  void* scratch;
  uint64_t scratch_bytes;
  void* cuda_stream;
} q27_verify_forward_t8_args;

q27_verify_status q27_verify_query_layout(uint32_t request_count,
                                          uint32_t context_capacity,
                                          q27_verify_layout* output);
q27_verify_status q27_verify_validate_state(
    const q27_verify_target_state* state,
    const q27_verify_journal* journal);

/* Allocation-free, synchronization-free, graph-safe control launches. */
q27_verify_status q27_verify_snapshot_base(const q27_verify_state_args* args);
q27_verify_status q27_verify_snapshot_checkpoint(
    const q27_verify_snapshot_checkpoint_args* args);
q27_verify_status q27_verify_accept_greedy(
    const q27_verify_accept_args* args);
q27_verify_status q27_verify_rollback(const q27_verify_state_args* args);
q27_verify_status q27_verify_commit(const q27_verify_commit_args* args);

q27_verify_status q27_verify_forward_t8(
    const q27_verify_forward_t8_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_VERIFY_H_

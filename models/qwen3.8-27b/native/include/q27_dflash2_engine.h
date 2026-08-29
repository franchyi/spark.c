#ifndef Q27_DFLASH2_ENGINE_H_
#define Q27_DFLASH2_ENGINE_H_

#include <stdint.h>

#include "q27_dflash2.h"
#include "q27_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_DFLASH2_ENGINE_ABI_VERSION 1u

typedef struct q27_dflash2_engine q27_dflash2_engine;

/*
 * Fixed batch-one owner configuration. The target capsule is constructed and
 * destroyed by this engine. Weight descriptor structs are copied at create;
 * their device-visible checkpoint payloads must remain alive until destroy.
 */
typedef struct q27_dflash2_engine_create_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_model_weights* target_weights;
  const q27_model_options* target_options;
  const q27_dflash2_weights* draft_weights;
  float rms_epsilon;
  uint32_t reserved;
} q27_dflash2_engine_create_args;

/*
 * One completed DFlash2 verify transaction. emitted_tokens contains the
 * accepted draft prefix followed by the target bonus; entries at or after
 * emitted_count are zero. anchor_token is the already-emitted token consumed
 * as target row zero and is not repeated in emitted_tokens.
 */
typedef struct q27_dflash2_engine_block_result {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t base_position;
  uint32_t new_position;
  uint32_t anchor_token;
  uint32_t accepted_draft_tokens;
  uint32_t emitted_count;
  uint32_t bonus_token;
  uint32_t proposed_tokens[Q27_DFLASH2_BLOCK_SIZE];
  uint32_t target_top1[Q27_DFLASH2_BLOCK_SIZE];
  uint32_t emitted_tokens[Q27_DFLASH2_BLOCK_SIZE];
  uint64_t elapsed_us;
} q27_dflash2_engine_block_result;

typedef struct q27_dflash2_engine_stats {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t target_resident_weight_bytes;
  uint64_t draft_checkpoint_weight_bytes;
  uint64_t state_bytes;   /* Target plus draft owner state. */
  uint64_t scratch_bytes; /* Target plus draft owner scratch. */
  uint64_t prompt_tokens;
  uint64_t verify_calls;
  uint64_t proposed_draft_tokens;
  uint64_t accepted_draft_tokens;
  uint64_t emitted_tokens;
  uint64_t last_prefill_us;
  uint64_t last_block_us;
  uint32_t context_capacity;
  uint32_t position;
  uint32_t next_anchor_token;
  uint32_t ready_to_decode;
} q27_dflash2_engine_stats;

q27_dflash2_status q27_dflash2_engine_create(
    const q27_dflash2_engine_create_args* args,
    q27_dflash2_engine** output);

/* Reset the sole target/draft slot. No request may be in flight. */
q27_dflash2_status q27_dflash2_engine_reset(q27_dflash2_engine* engine);

/*
 * Reset and prefill one complete host prompt through the target. Target
 * feature tiles are projected and committed to the draft KV ring before this
 * synchronizing call returns. first_token is the target's first greedy token
 * and becomes the internal anchor for decode_block.
 */
q27_dflash2_status q27_dflash2_engine_prefill(
    q27_dflash2_engine* engine, const uint32_t* host_tokens, uint32_t count,
    uint32_t* first_token);

/* Fixed greedy block=8 transaction; allocation-free after create. */
q27_dflash2_status q27_dflash2_engine_decode_block(
    q27_dflash2_engine* engine, q27_dflash2_engine_block_result* output);

q27_dflash2_status q27_dflash2_engine_get_stats(
    const q27_dflash2_engine* engine, q27_dflash2_engine_stats* output);

q27_dflash2_status q27_dflash2_engine_destroy(
    q27_dflash2_engine* engine);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_ENGINE_H_

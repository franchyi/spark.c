#ifndef Q27_H_
#define Q27_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_ABI_VERSION 1u

typedef struct q27_engine q27_engine;

typedef enum q27_status_code {
  Q27_STATUS_OK = 0,
  Q27_STATUS_INVALID_ARGUMENT = 1,
  Q27_STATUS_INCOMPATIBLE_CHECKPOINT = 2,
  Q27_STATUS_OUT_OF_MEMORY = 3,
  Q27_STATUS_CUDA_ERROR = 4,
  Q27_STATUS_INTERNAL = 5,
} q27_status_code;

typedef struct q27_status {
  int32_t code;
  const char* message;
} q27_status;

enum {
  Q27_DISABLE_CUDA_GRAPHS = 1u << 0,
  Q27_DISABLE_MTP = 1u << 1,
};

typedef struct q27_config {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* checkpoint_path;
  uint32_t max_slots;
  uint32_t context_capacity;
  uint32_t flags;
  uint32_t device_id;
  uint64_t workspace_limit_bytes;
} q27_config;

typedef struct q27_stats {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t mapped_weight_bytes;
  uint64_t resident_state_bytes;
  uint64_t kv_bytes;
  uint64_t workspace_bytes;
  uint64_t decode_steps;
  uint64_t verified_tokens;
  uint64_t last_step_us;
  uint32_t graph_replay_enabled;
  uint32_t mtp_enabled;
} q27_stats;

/*
 * The ABI is model-level by design. Per-kernel donor ABIs remain fixture-only;
 * Rust never mirrors the 64-layer launch list and never owns CUDA addresses.
 */
q27_status q27_open(const q27_config* config, q27_engine** output);
q27_status q27_slot_reset(q27_engine* engine, uint32_t slot);
q27_status q27_prefill(q27_engine* engine, uint32_t slot,
                       const uint32_t* tokens, uint32_t token_count,
                       float* last_logits);
q27_status q27_decode(q27_engine* engine, uint32_t slot, uint32_t token,
                      float* logits);
q27_status q27_verify(q27_engine* engine, uint32_t slot,
                      const uint32_t* draft_tokens, uint32_t token_count,
                      float* logits);
q27_status q27_state_snapshot(q27_engine* engine, uint32_t slot,
                              uint32_t checkpoint);
q27_status q27_state_restore(q27_engine* engine, uint32_t slot,
                             uint32_t checkpoint);
q27_status q27_get_stats(const q27_engine* engine, q27_stats* output);
q27_status q27_close(q27_engine* engine);

#ifdef __cplusplus
}
#endif

#endif

#ifndef Q27_GDN_PREFILL_C427_H_
#define Q27_GDN_PREFILL_C427_H_

#include "q27_gdn_prefill_c427_aot.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_C427_GDN_PREFILL_ABI_VERSION 1u

enum {
  Q27_C427_GDN_PREFILL_T512 = 512,
  Q27_C427_GDN_PREFILL_T2048 = 2048,
};

typedef struct q27_c427_gdn_prefill q27_c427_gdn_prefill;

/*
 * Exact c427 BF16-state recurrence, after convolution, split, gating and Q/K
 * normalization. State is updated in place. Output is the ungated core
 * [T,48,128]; the caller retains Qwen's final Z-gated RMSNorm and projection.
 */
typedef struct q27_c427_gdn_prefill_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t token_count;
  uint32_t reserved;
  const void* q_normalized_bf16;
  uint64_t q_bytes;
  const void* k_normalized_bf16;
  uint64_t k_bytes;
  const void* v_bf16;
  uint64_t v_bytes;
  const float* g_log_f32;
  uint64_t g_bytes;
  const float* beta_f32;
  uint64_t beta_bytes;
  void* state_bf16;
  uint64_t state_bytes;
  void* output_bf16;
  uint64_t output_bytes;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_c427_gdn_prefill_args;

/* Load the pinned selected cubins once. This is the only allocating call. */
q27_c427_gdn_aot_status q27_c427_gdn_prefill_create(
    const char* artifact_directory, q27_c427_gdn_prefill** output);

q27_c427_gdn_aot_status q27_c427_gdn_prefill_workspace_bytes(
    uint32_t token_count, uint64_t* output_bytes);

/* Six asynchronous launches: metadata plus the five exact c427 FLA stages. */
q27_c427_gdn_aot_status q27_c427_gdn_prefill_forward(
    q27_c427_gdn_prefill* capsule,
    const q27_c427_gdn_prefill_args* args);

void q27_c427_gdn_prefill_destroy(q27_c427_gdn_prefill* capsule);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_C427_H_

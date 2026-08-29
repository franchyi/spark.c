#ifndef Q27_GDN_PREFILL_C427_PREPARE_H_
#define Q27_GDN_PREFILL_C427_PREPARE_H_

#include "q27_gdn_prefill_c427_aot.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_C427_GDN_PREPARE_ABI_VERSION 1u

typedef struct q27_c427_gdn_prepare q27_c427_gdn_prepare;

/*
 * Exact pinned-c427 prompt preparation after the QKVZ projection. The
 * convolution state is updated in place from valid rows only. All output
 * rows in [valid_tokens, token_count) are zeroed by the capsule.
 */
typedef struct q27_c427_gdn_prepare_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t token_count;
  uint32_t valid_tokens;
  const void* fused_qkvz_bf16;
  uint64_t fused_qkvz_bytes;
  const void* conv_weight_bf16;
  uint64_t conv_weight_bytes;
  void* convolution_state_bf16;
  uint64_t convolution_state_bytes;
  void* q_normalized_bf16;
  uint64_t q_bytes;
  void* k_normalized_bf16;
  uint64_t k_bytes;
  void* v_bf16;
  uint64_t v_bytes;
  void* z_bf16;
  uint64_t z_bytes;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_c427_gdn_prepare_args;

q27_c427_gdn_aot_status q27_c427_gdn_prepare_create(
    const char* artifact_directory, q27_c427_gdn_prepare** output);

q27_c427_gdn_aot_status q27_c427_gdn_prepare_workspace_bytes(
    uint32_t token_count, uint64_t* output_bytes);

/* Metadata plus five asynchronous pinned-c427 launches. */
q27_c427_gdn_aot_status q27_c427_gdn_prepare_forward(
    q27_c427_gdn_prepare* capsule,
    const q27_c427_gdn_prepare_args* args);

void q27_c427_gdn_prepare_destroy(q27_c427_gdn_prepare* capsule);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_C427_PREPARE_H_

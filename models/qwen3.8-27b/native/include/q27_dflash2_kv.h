#ifndef Q27_DFLASH2_KV_H_
#define Q27_DFLASH2_KV_H_

#include <stdint.h>

#include "q27_dflash2_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_DFLASH2_KV_ABI_VERSION 1u
#define Q27_DFLASH2_KV_MAX_CHUNK_TOKENS 2048u
#define Q27_DFLASH2_KV_SCRATCH_BYTES_PER_TOKEN (8ULL * 128ULL * 2ULL)
#define Q27_DFLASH2_KV_ROPE_CACHE_BYTES_PER_TOKEN (64ULL * 2ULL * 4ULL)

/*
 * Invalidate all 2,048 ring tags without clearing K/V payloads and reset the
 * host-visible committed length. Enqueue on the same stream used by subsequent
 * materialization/attention work; the call allocates and synchronizes nothing.
 */
typedef struct q27_dflash2_kv_reset_args {
  uint32_t struct_size;
  uint32_t abi_version;
  q27_dflash2_state_view* state;
  void* cuda_stream;
} q27_dflash2_kv_reset_args;

/*
 * Strict batch-one, contiguous-chunk context-KV reference. context_hidden is
 * the BF16 [token_count,5120] result of q27_dflash2_project_context. For each
 * of the five draft layers this call performs K-only and V-only BF16 GEMMs,
 * per-head K RMSNorm, full-dimension NeoX RoPE, and a fused K/V ring write
 * [layer, absolute_position%2048, 8,128] K/V plus the absolute position tag.
 *
 * token_count is 1..2048 so one call never has duplicate ring destinations.
 * Arbitrary-length prefill is processed in increasing chunks by advancing
 * context_hidden and first_position. The later chunks intentionally overwrite
 * expired ring slots. The caller updates state->committed_length only after all
 * chunks have been enqueued on the same stream.
 *
 * k_scratch_bf16 and v_scratch_bf16 each require token_count*8*128 BF16;
 * rope_cache_f32 requires token_count*64*(cos,sin) FP32. The 64-element inverse
 * frequency table is initialized once with q27_dflash2_initialize_rope.
 */
typedef struct q27_dflash2_kv_materialize_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_dflash2_weights* weights;
  const void* context_hidden_bf16;
  uint64_t first_position;
  uint32_t token_count;
  float rms_epsilon;
  const float* rope_inverse_frequencies_f32;
  float* rope_cache_f32;
  void* k_scratch_bf16;
  void* v_scratch_bf16;
  q27_dflash2_state_view* state;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_kv_materialize_args;

q27_dflash2_status q27_dflash2_reset_kv(
    const q27_dflash2_kv_reset_args* args);
q27_dflash2_status q27_dflash2_materialize_context_kv(
    const q27_dflash2_kv_materialize_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_KV_H_

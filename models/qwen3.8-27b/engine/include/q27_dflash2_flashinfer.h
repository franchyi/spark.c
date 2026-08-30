#ifndef Q27_DFLASH2_FLASHINFER_H_
#define Q27_DFLASH2_FLASHINFER_H_

#include <stdint.h>

#include "q27_dflash2_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_DFLASH2_FLASHINFER_ABI_VERSION 1u
#define Q27_DFLASH2_FLASHINFER_HISTORY_TOKENS 2047ULL
#define Q27_DFLASH2_FLASHINFER_MAX_KV_TOKENS \
  (Q27_DFLASH2_FLASHINFER_HISTORY_TOKENS + 8ULL)
#define Q27_DFLASH2_FLASHINFER_ONE_STAGING_BYTES \
  (Q27_DFLASH2_FLASHINFER_MAX_KV_TOKENS * 8ULL * 128ULL * 2ULL)
#define Q27_DFLASH2_FLASHINFER_INVALID_COUNT_OFFSET \
  (2ULL * Q27_DFLASH2_FLASHINFER_ONE_STAGING_BYTES)
#define Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES \
  (Q27_DFLASH2_FLASHINFER_INVALID_COUNT_OFFSET + 4ULL)

/*
 * Exact pinned-FlashInfer implementation of q27_dflash2_sliding_attention_hook.
 * The state ring contains committed context only. This hook gathers its last
 * min(committed_length,2047) tagged rows plus the current eight ephemeral K/V
 * rows into caller workspace, then runs FlashInfer's BF16 single-prefill kernel
 * with Q=8, 32/8 GQA, D=128, causal mask, scale=1/sqrt(128), and
 * window_left=2047. It never commits the ephemeral mask-block K/V to live state;
 * accepted context is materialized later from target features by q27_dflash2_kv.
 *
 * `call->workspace` must contain Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES
 * device-visible bytes. The final u32 at
 * Q27_DFLASH2_FLASHINFER_INVALID_COUNT_OFFSET is cleared then incremented for
 * any missing/stale history tag or non-contiguous position. A nonzero value is
 * a failed proposal invariant and must be checked before acceptance. The hook
 * allocates and synchronizes nothing and has no framework runtime dependency.
 */
q27_dflash2_status q27_dflash2_flashinfer_sliding_attention(
    const q27_dflash2_sliding_attention_call* call, void* user_data);

/* Device pointer into a workspace previously passed to the hook. */
uint32_t* q27_dflash2_flashinfer_invalid_count(void* workspace,
                                               uint64_t workspace_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_FLASHINFER_H_

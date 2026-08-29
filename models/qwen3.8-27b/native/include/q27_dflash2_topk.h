#ifndef Q27_DFLASH2_TOPK_H_
#define Q27_DFLASH2_TOPK_H_

#include "q27_dflash2.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_DFLASH2_TOPK_ABI_VERSION 1u

enum {
  Q27_DFLASH2_TOPK_ROWS = Q27_DFLASH2_DRAFT_TOKENS,
  Q27_DFLASH2_TOPK_K = Q27_DFLASH2_SELECTOR_TOP_K,
};

#define Q27_DFLASH2_TOPK_HIDDEN_ELEMENTS                                  \
  (Q27_DFLASH2_TOPK_ROWS * Q27_DFLASH2_HIDDEN_SIZE)
#define Q27_DFLASH2_TOPK_LOGIT_ELEMENTS                                   \
  (Q27_DFLASH2_TOPK_ROWS * Q27_DFLASH2_VOCAB_SIZE)
#define Q27_DFLASH2_TOPK_LM_HEAD_BYTES                                    \
  (Q27_DFLASH2_VOCAB_SIZE * Q27_DFLASH2_HIDDEN_SIZE * 2ULL)
#define Q27_DFLASH2_TOPK_LOGIT_BYTES                                      \
  (Q27_DFLASH2_TOPK_LOGIT_ELEMENTS * 4ULL)

/*
 * Deterministic top-16 reference over caller-owned FP32 GEMM scratch.
 * Before comparison every value is rounded FP32 -> BF16 -> FP32, matching the
 * pinned SGLang dense-BF16 lm_head output. Results are sorted by descending
 * rounded value, then ascending token id. NaN is treated as negative infinity.
 */
typedef struct q27_dflash2_topk_from_logits_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const float* logits_f32;       /* [7,248320]. */
  uint64_t logits_elements;
  uint32_t* candidate_ids_u32;   /* [7,16]. */
  float* unary_logits_f32;       /* [7,16], BF16 values promoted to FP32. */
  void* cuda_stream;
} q27_dflash2_topk_from_logits_args;

/*
 * Correctness-first target LM-head path. hidden_bf16 is exactly draft final
 * hidden block rows 1..7 (row zero is the anchor and is excluded), flattened
 * as [7,5120]. lm_head_weight_bf16 is the pinned target's row-major
 * [248320,5120] BF16 head. logits_f32 is caller-owned [7,248320] scratch.
 *
 * The cuBLAS handle must use host scalar pointer mode, be warmed before graph
 * capture, and not be shared concurrently with a different stream.
 */
typedef struct q27_dflash2_lm_head_topk_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* hidden_bf16;
  const void* lm_head_weight_bf16;
  uint64_t lm_head_weight_bytes;
  float* logits_f32;
  uint64_t logits_elements;
  uint32_t* candidate_ids_u32;
  float* unary_logits_f32;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_lm_head_topk_args;

q27_dflash2_status q27_dflash2_topk_from_logits(
    const q27_dflash2_topk_from_logits_args* args);
q27_dflash2_status q27_dflash2_lm_head_topk(
    const q27_dflash2_lm_head_topk_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_TOPK_H_

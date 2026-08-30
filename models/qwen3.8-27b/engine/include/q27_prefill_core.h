#ifndef Q27_PREFILL_CORE_H_
#define Q27_PREFILL_CORE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_PREFILL_CORE_ABI_VERSION 1u

enum {
  Q27_PREFILL_CORE_TOKENS = 128,
  Q27_PREFILL_CORE_M512_TOKENS = 512,
  Q27_PREFILL_CORE_M2048_TOKENS = 2048,
  Q27_PREFILL_CORE_M4096_TOKENS = 4096,
  Q27_PREFILL_CORE_M8192_TOKENS = 8192,
  Q27_PREFILL_CORE_HIDDEN = 5120,
  Q27_PREFILL_CORE_VOCAB = 248320,
};

typedef enum q27_prefill_core_status_code {
  Q27_PREFILL_CORE_OK = 0,
  Q27_PREFILL_CORE_INVALID_ARGUMENT = 1,
  Q27_PREFILL_CORE_CUDA_ERROR = 2,
} q27_prefill_core_status_code;

typedef struct q27_prefill_core_status {
  int32_t code;
  const char* message;
} q27_prefill_core_status;

/*
 * Gather valid_tokens device token IDs into a fixed [128,5120] BF16 tile.
 * Padding rows are zeroed. invalid_token_count_u32 is a device scalar cleared
 * by the call and incremented once per out-of-range valid token.
 */
typedef struct q27_prefill_embedding_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const uint32_t* token_ids_u32;
  const void* embedding_bf16;
  void* output_bf16;
  uint32_t* invalid_token_count_u32;
  void* cuda_stream;
} q27_prefill_embedding_args;

/*
 * Fixed [128,5120] Gemma RMSNorm. For valid rows, BF16-round input+residual
 * (or input alone), publish that value to residual_output, then multiply by
 * rsqrt(mean(x^2)+epsilon) * (1+checkpoint_weight). Padding rows are zeroed.
 * All tensor pointers are disjoint caller-owned device buffers.
 */
typedef struct q27_prefill_norm_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t has_residual;
  const void* input_bf16;
  const void* residual_bf16;
  const void* checkpoint_weight_bf16;
  void* output_bf16;
  void* residual_output_bf16;
  float epsilon;
  uint32_t reserved;
  void* cuda_stream;
} q27_prefill_norm_args;

q27_prefill_core_status q27_prefill_embedding(
    const q27_prefill_embedding_args* args);
q27_prefill_core_status q27_prefill_norm(
    const q27_prefill_norm_args* args);

/*
 * M=512 variants of the same operations and argument ABI.  The caller must
 * provide fixed [512,5120] output tiles; rows valid_tokens..511 are zeroed.
 * Keeping separate symbols prevents changing the established M=128 contract.
 */
q27_prefill_core_status q27_prefill_embedding_m512(
    const q27_prefill_embedding_args* args);
q27_prefill_core_status q27_prefill_norm_m512(
    const q27_prefill_norm_args* args);

/* Fixed M=2048 prompt lane; semantics are identical to the M512 lane. */
q27_prefill_core_status q27_prefill_embedding_m2048(
    const q27_prefill_embedding_args* args);
q27_prefill_core_status q27_prefill_norm_m2048(
    const q27_prefill_norm_args* args);

/* Fixed M=4096 prompt lane; semantics are identical to the M2048 lane. */
q27_prefill_core_status q27_prefill_embedding_m4096(
    const q27_prefill_embedding_args* args);
q27_prefill_core_status q27_prefill_norm_m4096(
    const q27_prefill_norm_args* args);

/* Fixed M=8192 prompt lane; semantics are identical to the M2048 lane. */
q27_prefill_core_status q27_prefill_embedding_m8192(
    const q27_prefill_embedding_args* args);
q27_prefill_core_status q27_prefill_norm_m8192(
    const q27_prefill_norm_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_PREFILL_CORE_H_

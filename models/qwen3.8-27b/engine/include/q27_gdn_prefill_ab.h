#ifndef Q27_GDN_PREFILL_AB_H_
#define Q27_GDN_PREFILL_AB_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_PREFILL_AB_ABI_VERSION 1u

enum {
  Q27_GDN_PREFILL_AB_TOKENS = 128,
  Q27_GDN_PREFILL_AB_HIDDEN = 5120,
  Q27_GDN_PREFILL_AB_HEADS = 48,
  Q27_GDN_PREFILL_AB_MERGED_HEADS = 96,
};

typedef enum q27_gdn_prefill_ab_status_code {
  Q27_GDN_PREFILL_AB_OK = 0,
  Q27_GDN_PREFILL_AB_INVALID_ARGUMENT = 1,
  Q27_GDN_PREFILL_AB_CUDA_ERROR = 2,
  Q27_GDN_PREFILL_AB_CUBLAS_ERROR = 3,
} q27_gdn_prefill_ab_status_code;

typedef struct q27_gdn_prefill_ab_status {
  int32_t code;
  const char* message;
} q27_gdn_prefill_ab_status;

/*
 * merged_weight is a load-time concatenation [A;B] in row-major
 * [96,5120] BF16. One fixed M=128 GEMM writes merged_scratch [128,96], then
 * a CUDA split publishes separate A/B [128,48]. Rows >= valid_tokens are
 * forced to zero without changing the physical GEMM shape.
 */
typedef struct q27_gdn_prefill_ab_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t reserved;
  const void* normalized_hidden_bf16;
  uint64_t normalized_hidden_bytes;
  const void* merged_weight_bf16;
  uint64_t merged_weight_bytes;
  void* merged_scratch_bf16;
  uint64_t merged_scratch_bytes;
  void* projected_a_bf16;
  uint64_t projected_a_bytes;
  void* projected_b_bf16;
  uint64_t projected_b_bytes;
  void* cublas_handle;
  void* cuda_stream;
} q27_gdn_prefill_ab_args;

q27_gdn_prefill_ab_status q27_gdn_prefill_ab_project(
    const q27_gdn_prefill_ab_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_AB_H_

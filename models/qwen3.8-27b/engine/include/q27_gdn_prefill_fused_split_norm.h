#ifndef Q27_GDN_PREFILL_FUSED_SPLIT_NORM_H_
#define Q27_GDN_PREFILL_FUSED_SPLIT_NORM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_GDN_PREFILL_FUSED_SPLIT_NORM_ABI_VERSION 1u

enum {
  Q27_GDN_FUSED_TOKENS = 128,
  Q27_GDN_FUSED_QK_HEADS = 16,
  Q27_GDN_FUSED_VALUE_HEADS = 48,
  Q27_GDN_FUSED_HEAD_DIM = 128,
  Q27_GDN_FUSED_QKV_WIDTH = 10240,
  Q27_GDN_FUSED_Z_WIDTH = 6144,
  Q27_GDN_FUSED_QKVZ_WIDTH = 16384,
  Q27_GDN_FUSED_CONV_KERNEL = 4,
  Q27_GDN_FUSED_CONV_HISTORY = 3,
};

typedef enum q27_gdn_fused_split_norm_status_code {
  Q27_GDN_FUSED_SPLIT_NORM_OK = 0,
  Q27_GDN_FUSED_SPLIT_NORM_INVALID_ARGUMENT = 1,
  Q27_GDN_FUSED_SPLIT_NORM_CUDA_ERROR = 2,
} q27_gdn_fused_split_norm_status_code;

typedef struct q27_gdn_fused_split_norm_status {
  int32_t code;
  const char* message;
} q27_gdn_fused_split_norm_status;

/*
 * Fixed-M128 projection-boundary fusion for the Qwen3.8-27B GDN layer.
 *
 * fused_qkvz is a BF16 [source_rows,16384] output of the model's fused QKVZ
 * projection. source_row selects one physical M128 slice without first
 * materializing a private copy. The first 10240 features enter the donor-exact
 * width-4 causal convolution; features 10240..16383 are the raw Z gate.
 * Outputs are:
 *
 *   q_normalized [128,16,128] BF16
 *   k_normalized [128,16,128] BF16
 *   value        [128,48,128] BF16, after causal convolution
 *   projected_z  [128,48,128] BF16, copied before convolution
 *
 * Q/K L2 normalization occurs after causal convolution, with epsilon 1e-6.
 * This compatibility fallback uses a 128-thread reduction tree and can differ
 * from the pinned c427 Triton reduction by 1--2 BF16 ULP.
 * Convolution products round to BF16 before FP32 accumulation. Rows at or
 * above valid_tokens produce zero Q/K/V and cannot update state; Z remains a
 * byte-exact split of the projection buffer, matching the current unfused
 * layer boundary. convolution_state is [10240,3] BF16 and is updated in
 * place. All buffers must be distinct, device-accessible, and at least
 * 2-byte aligned. The call performs no allocation or synchronization.
 */
typedef struct q27_gdn_fused_split_norm_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t source_row;
  const void* fused_qkvz_bf16;
  uint64_t fused_qkvz_bytes;
  const void* conv_weight_bf16;
  uint64_t conv_weight_bytes;
  void* convolution_state_bf16;
  uint64_t convolution_state_bytes;
  void* q_normalized_bf16;
  uint64_t q_normalized_bytes;
  void* k_normalized_bf16;
  uint64_t k_normalized_bytes;
  void* value_bf16;
  uint64_t value_bytes;
  void* projected_z_bf16;
  uint64_t projected_z_bytes;
  void* cuda_stream;
} q27_gdn_fused_split_norm_args;

q27_gdn_fused_split_norm_status q27_gdn_fused_split_norm(
    const q27_gdn_fused_split_norm_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_FUSED_SPLIT_NORM_H_

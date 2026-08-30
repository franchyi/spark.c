#ifndef Q27_GDN_VERIFY_T8_H_
#define Q27_GDN_VERIFY_T8_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Qwen3.8-27B batch-one/T=8 GDN verification state seam.
 *
 * This ABI is deliberately separate from the fixed-M128 prompt-prefill ABI.
 * Verification must use the pinned FlashInfer sequential BF16-state MTP
 * arithmetic, leave the live state unchanged, and publish h_1..h_8 so the
 * accepted row can be selected without replaying the target model.
 */
#define Q27_GDN_VERIFY_T8_ABI_VERSION 1u

enum {
  Q27_GDN_VERIFY_TOKENS = 8,
  Q27_GDN_VERIFY_GDN_LAYERS = 48,
  Q27_GDN_VERIFY_QK_HEADS = 16,
  Q27_GDN_VERIFY_VALUE_HEADS = 48,
  Q27_GDN_VERIFY_HEAD_DIM = 128,
  Q27_GDN_VERIFY_QKV_WIDTH = 10240,
  Q27_GDN_VERIFY_VALUE_WIDTH = 6144,
  Q27_GDN_VERIFY_CONV_HISTORY = 3,
  Q27_GDN_VERIFY_CONV_KERNEL = 4,
};

#define Q27_GDN_VERIFY_CONV_STATE_BYTES_PER_LAYER \
  (Q27_GDN_VERIFY_QKV_WIDTH * Q27_GDN_VERIFY_CONV_HISTORY * 2ULL)
#define Q27_GDN_VERIFY_RECURRENT_STATE_BYTES_PER_LAYER                  \
  (Q27_GDN_VERIFY_VALUE_HEADS * Q27_GDN_VERIFY_HEAD_DIM *              \
   Q27_GDN_VERIFY_HEAD_DIM * 2ULL)
#define Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER                     \
  (Q27_GDN_VERIFY_TOKENS * Q27_GDN_VERIFY_CONV_STATE_BYTES_PER_LAYER)
#define Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER                \
  (Q27_GDN_VERIFY_TOKENS * Q27_GDN_VERIFY_RECURRENT_STATE_BYTES_PER_LAYER)

typedef enum q27_gdn_verify_t8_status_code {
  Q27_GDN_VERIFY_T8_OK = 0,
  Q27_GDN_VERIFY_T8_INVALID_ARGUMENT = 1,
  Q27_GDN_VERIFY_T8_CUDA_ERROR = 2,
  Q27_GDN_VERIFY_T8_ARTIFACT_ERROR = 3,
} q27_gdn_verify_t8_status_code;

typedef struct q27_gdn_verify_t8_status {
  int32_t code;
  const char* message;
} q27_gdn_verify_t8_status;

/*
 * Run the donor-exact BF16 width-four causal convolution over exactly eight
 * rows. projected_qkv is [8,10240] BF16. convolved_qkv has the same layout.
 * live_convolution_state is read-only [10240,3]. checkpoint_convolution is
 * [8,10240,3], where row t contains the state after input row t. No live
 * state is mutated, and the call allocates and synchronizes nothing.
 */
typedef struct q27_gdn_verify_t8_conv_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* projected_qkv_bf16;
  uint64_t projected_qkv_bytes;
  const void* conv_weight_bf16;
  uint64_t conv_weight_bytes;
  const void* live_convolution_state_bf16;
  uint64_t live_convolution_state_bytes;
  void* convolved_qkv_bf16;
  uint64_t convolved_qkv_bytes;
  void* checkpoint_convolution_bf16;
  uint64_t checkpoint_convolution_bytes;
  void* cuda_stream;
} q27_gdn_verify_t8_conv_args;

/*
 * Invoke the separately exported pinned FlashInfer BF16-state T=8 artifact.
 * convolved_qkv is [8,10240] with Q/K/V at offsets 0/2048/4096. projected
 * A/B are [8,48]. live_recurrent_state is read-only [48,128,128]. Output is
 * [8,48,128], and checkpoint_recurrent is [8,48,128,128] containing h_1..h_8.
 * state_index_i32 is a device scalar equal to zero. The AOT specialization
 * has intermediate-state caching enabled and final live-state write disabled.
 */
typedef struct q27_gdn_verify_t8_recurrent_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* convolved_qkv_bf16;
  uint64_t convolved_qkv_bytes;
  const void* projected_a_bf16;
  uint64_t projected_a_bytes;
  const void* projected_b_bf16;
  uint64_t projected_b_bytes;
  const float* a_log_f32;
  const float* dt_bias_f32;
  const void* live_recurrent_state_bf16;
  uint64_t live_recurrent_state_bytes;
  const int32_t* state_index_i32;
  void* recurrent_output_bf16;
  uint64_t recurrent_output_bytes;
  void* checkpoint_recurrent_bf16;
  uint64_t checkpoint_recurrent_bytes;
  void* cuda_stream;
} q27_gdn_verify_t8_recurrent_args;

/*
 * Select one accepted row from layer-major journals:
 *   convolution [48,8,10240,3]
 *   recurrent   [48,8,48,128,128]
 * and publish it to the 48 live layer states. selected_row_u32 and
 * device_error_u32 are device scalars. Invalid rows write no live state.
 */
typedef struct q27_gdn_verify_t8_commit_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* checkpoint_convolution_bf16;
  uint64_t checkpoint_convolution_bytes;
  const void* checkpoint_recurrent_bf16;
  uint64_t checkpoint_recurrent_bytes;
  void* live_convolution_state_bf16;
  uint64_t live_convolution_state_bytes;
  void* live_recurrent_state_bf16;
  uint64_t live_recurrent_state_bytes;
  const uint32_t* selected_row_u32;
  uint32_t* device_error_u32;
  void* cuda_stream;
} q27_gdn_verify_t8_commit_args;

q27_gdn_verify_t8_status q27_gdn_verify_t8_convolve(
    const q27_gdn_verify_t8_conv_args* args);
q27_gdn_verify_t8_status q27_gdn_verify_t8_recurrent(
    const q27_gdn_verify_t8_recurrent_args* args);
q27_gdn_verify_t8_status q27_gdn_verify_t8_commit(
    const q27_gdn_verify_t8_commit_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_VERIFY_T8_H_

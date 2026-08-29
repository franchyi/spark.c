#ifndef Q27_GDN_PREFILL_FLASHINFER_SM121_H_
#define Q27_GDN_PREFILL_FLASHINFER_SM121_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Experimental Python-free seam for FlashInfer's actual SM121 GDN-prefill
 * donor.  It is intentionally separate from the production BF16-state path:
 * this donor requires FP32 recurrent state and therefore is not byte-identical
 * to the Mia/SGLang --mamba-ssm-dtype=bfloat16 reference.
 */
#define Q27_GDN_PREFILL_SM121_ABI_VERSION 1u

enum {
  Q27_GDN_PREFILL_SM121_MAX_TOKENS = 2048,
  Q27_GDN_PREFILL_SM121_QK_HEADS = 16,
  Q27_GDN_PREFILL_SM121_VALUE_HEADS = 48,
  Q27_GDN_PREFILL_SM121_HEAD_DIM = 128,
};

#define Q27_GDN_PREFILL_SM121_STATE_BYTES \
  (Q27_GDN_PREFILL_SM121_VALUE_HEADS * Q27_GDN_PREFILL_SM121_HEAD_DIM * \
   Q27_GDN_PREFILL_SM121_HEAD_DIM * 4ULL)

typedef enum q27_gdn_prefill_sm121_status_code {
  Q27_GDN_PREFILL_SM121_OK = 0,
  Q27_GDN_PREFILL_SM121_INVALID_ARGUMENT = 1,
  Q27_GDN_PREFILL_SM121_CUDA_ERROR = 2,
  Q27_GDN_PREFILL_SM121_ARTIFACT_ERROR = 3,
} q27_gdn_prefill_sm121_status_code;

typedef struct q27_gdn_prefill_sm121_status {
  int32_t code;
  const char* message;
} q27_gdn_prefill_sm121_status;

/*
 * Fixed batch-one Qwen3.8 GDN prefill. token_count is 1..2048; the exported
 * specialization keeps T dynamic, so the same object covers M512 and M2048.
 *
 * Q/K/V and output are compact BF16 [T,H,128]. alpha_f32 is the raw
 * per-token linear-space forget factor [T,48], i.e. exp(g_log), not either
 * SGLang's log-space g or the native path's chunk-local cumulative g.
 * beta_f32 is [T,48]. cu_seqlens_i64 is a device [2] equal to
 * {0,T}. initial_state_f32 and output_state_f32 are distinct compact
 * [1,48,128,128] buffers. tensormap_workspace must be 128-byte aligned and
 * have at least 128 bytes per physical SM; query the requirement below.
 *
 * The call allocates and synchronizes nothing. It changes no production
 * state unless a future owner explicitly adopts the FP32-state contract.
 */
typedef struct q27_gdn_prefill_sm121_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t token_count;
  uint32_t reserved;
  const void* q_bf16;
  uint64_t q_bytes;
  const void* k_bf16;
  uint64_t k_bytes;
  const void* v_bf16;
  uint64_t v_bytes;
  const float* alpha_f32;
  uint64_t alpha_bytes;
  const float* beta_f32;
  uint64_t beta_bytes;
  const float* initial_state_f32;
  uint64_t initial_state_bytes;
  float* output_state_f32;
  uint64_t output_state_bytes;
  void* output_bf16;
  uint64_t output_bytes;
  const int64_t* cu_seqlens_i64;
  uint64_t cu_seqlens_bytes;
  void* tensormap_workspace;
  uint64_t tensormap_workspace_bytes;
  void* cuda_stream;
} q27_gdn_prefill_sm121_args;

q27_gdn_prefill_sm121_status q27_gdn_prefill_sm121_workspace_bytes(
    uint64_t* output_bytes);
q27_gdn_prefill_sm121_status q27_gdn_prefill_sm121_forward(
    const q27_gdn_prefill_sm121_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_FLASHINFER_SM121_H_

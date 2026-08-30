#ifndef Q27_DFLASH2_ATTENTION_H_
#define Q27_DFLASH2_ATTENTION_H_

#include <stdint.h>

#include "q27_dflash2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_DFLASH2_ATTENTION_ABI_VERSION 1u

#define Q27_DFLASH2_ATTENTION_Q_BYTES (8ULL * 32ULL * 128ULL * 2ULL)
#define Q27_DFLASH2_ATTENTION_KV_BYTES (8ULL * 8ULL * 128ULL * 2ULL)
#define Q27_DFLASH2_ATTENTION_CONTEXT_BYTES \
  Q27_DFLASH2_ATTENTION_Q_BYTES
#define Q27_DFLASH2_ATTENTION_ROPE_FREQUENCY_BYTES (64ULL * 4ULL)
#define Q27_DFLASH2_ATTENTION_ROPE_CACHE_BYTES \
  (8ULL * 64ULL * 2ULL * 4ULL)
#define Q27_DFLASH2_ATTENTION_OUTPUT_BYTES (8ULL * 5120ULL * 2ULL)

/* Fixed outer attention-sublayer workspace, all offsets 16-byte aligned. */
#define Q27_DFLASH2_ATTENTION_SUBLAYER_ROPE_FREQUENCY_OFFSET 0ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_ROPE_CACHE_OFFSET 256ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_CONV_COEFFICIENT_OFFSET 4352ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_PREPARED_OFFSET 24832ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_Q_OFFSET 106752ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_K_OFFSET 172288ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_V_OFFSET 188672ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_CONTEXT_OFFSET 205056ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_RAW_OUTPUT_OFFSET 270592ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_FLASHINFER_OFFSET 352512ULL
#define Q27_DFLASH2_ATTENTION_SUBLAYER_WORKSPACE_BYTES 8769796ULL

/*
 * Populate the caller-owned 64-element FP32 inverse-frequency table for the
 * pinned NeoX-style full-dimension RoPE (theta=10,000,000, head_dim=128).
 * Run once on the same stream before the first forward or graph capture.
 */
typedef struct q27_dflash2_rope_init_args {
  uint32_t struct_size;
  uint32_t abi_version;
  float* inverse_frequencies_f32;
  void* cuda_stream;
} q27_dflash2_rope_init_args;

/*
 * Required AOT sliding-attention boundary. q/k/v are contiguous BF16
 * [8,32,128], [8,8,128], [8,8,128]. q and k have already passed per-head
 * RMSNorm and NeoX RoPE. The implementation must read the committed tagged
 * K/V ring plus the current ephemeral block and write BF16 [8,32,128]
 * context. It must not mutate or commit the ring: accepted target features are
 * materialized later through q27_dflash2_kv. It must implement causal GQA with scale
 * 1/sqrt(128), window_left=2047, absolute positions, and the fixed ring/tag
 * state contract. It may enqueue work only on cuda_stream and must not allocate
 * or synchronize. positions must be the contiguous absolute interval beginning
 * at state->committed_length and remain below Q27_DFLASH2_MAX_POSITION.
 */
typedef struct q27_dflash2_sliding_attention_call {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t layer_index;
  uint32_t token_count;
  const uint64_t* positions_u64;
  const void* q_bf16;
  const void* k_bf16;
  const void* v_bf16;
  void* context_bf16;
  q27_dflash2_state_view* state;
  void* workspace;
  uint64_t workspace_bytes;
  float scale;
  uint32_t window_left;
  void* cuda_stream;
} q27_dflash2_sliding_attention_call;

/*
 * Fixed one-block DFlash2 attention:
 *   1. BF16 q/k/v projections from input [8,5120]
 *   2. per-head Q/K RMSNorm and NeoX RoPE
 *   3. revision-pinned FlashInfer sliding attention (no callback/fallback)
 *   4. BF16 o projection [8,4096] -> [8,5120]
 *
 * Every scratch/output pointer is caller-owned and disjoint. rope_cache_f32 is
 * [8,64,2] (cos,sin). workspace must provide the fixed FlashInfer staging
 * contract in q27_dflash2_flashinfer.h. The cuBLAS handle must use host scalar
 * pointer mode and be warmed. The call allocates and synchronizes nothing.
 * CUDA-graph capture is accepted only after the selected cuBLAS and attention
 * tactics pass the Spark graph fixture; the ABI itself contains no framework
 * object and exposes no injectable attention implementation.
 */
typedef struct q27_dflash2_attention_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t layer_index;
  uint32_t reserved;
  const q27_dflash2_layer_weights* weights;
  const void* input_bf16;
  const uint64_t* positions_u64;
  const float* rope_inverse_frequencies_f32;
  float* rope_cache_f32;
  void* q_bf16;
  void* k_bf16;
  void* v_bf16;
  void* context_bf16;
  void* output_bf16;
  q27_dflash2_state_view* state;
  void* workspace;
  uint64_t workspace_bytes;
  float rms_epsilon;
  uint32_t reserved2;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_attention_args;

q27_dflash2_status q27_dflash2_initialize_rope(
    const q27_dflash2_rope_init_args* args);
q27_dflash2_status q27_dflash2_attention_forward(
    const q27_dflash2_attention_args* args);

/*
 * Fixed attention dependency for q27_dflash2_forward. It performs attention
 * convolution prepare, the direct pinned-FlashInfer attention above, and
 * convolution finish using only the caller workspace layout above. user_data
 * must be null; no alternate attention implementation can be injected.
 */
struct q27_dflash2_sublayer_call;
q27_dflash2_status q27_dflash2_attention_sublayer(
    const struct q27_dflash2_sublayer_call* call, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_ATTENTION_H_

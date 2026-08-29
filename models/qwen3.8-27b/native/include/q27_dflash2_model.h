#ifndef Q27_DFLASH2_MODEL_H_
#define Q27_DFLASH2_MODEL_H_

#include <stdint.h>

#include "q27_dflash2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_DFLASH2_MODEL_ABI_VERSION 1u

/*
 * Project post-target-layer features [tokens, 5*5120] through fc.weight and
 * hidden_norm.  scratch_bf16 is caller-owned [tokens,5120].  The cuBLAS handle
 * must already exist, use host scalar pointer mode, and be warmed before graph
 * capture; this call itself allocates nothing and is CUDA-graph safe.
 */
typedef struct q27_dflash2_context_projection_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_dflash2_weights* weights;
  const void* target_features_bf16;
  void* scratch_bf16;
  void* context_hidden_bf16;
  uint32_t token_count;
  float rms_epsilon;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_context_projection_args;

/* Project the seven predicted draft rows [tokens,5120] to [tokens,256] BF16. */
typedef struct q27_dflash2_selector_projection_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_dflash2_weights* weights;
  const void* hidden_bf16;
  void* projected_hidden_bf16;
  uint32_t token_count;
  uint32_t reserved;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_selector_projection_args;

/*
 * Build the DFlash2 lattice:
 *   score[b,e,p,c] = unary[b,e,c] +
 *       dot(pred_codebook[pred_id] * hidden[b,e], succ_codebook[candidate_id])
 *
 * candidate_ids is [batch,7,16] u32, anchor_tokens is [batch] u32,
 * unary_logits is [batch,7,16] FP32, projected_hidden is [batch,7,256] BF16,
 * and scores is [batch,7,16,16] FP32. For edge zero every predecessor row uses
 * the anchor; later edges use the previous edge's candidate id at predecessor
 * index p. invalid_id_count_u32 is a caller-owned device scalar. The launch
 * zeroes it, increments it for every score cell whose predecessor/successor is
 * outside [0,248320), and writes -infinity instead of indexing a codebook.
 * The caller must treat a nonzero value as a failed proposal before path walk.
 */
typedef struct q27_dflash2_selector_score_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_dflash2_weights* weights;
  const uint32_t* candidate_ids;
  const uint32_t* anchor_tokens;
  const float* unary_logits;
  const void* projected_hidden_bf16;
  float* scores;
  uint32_t* invalid_id_count_u32;
  uint32_t batch_size;
  uint32_t reserved;
  void* cuda_stream;
} q27_dflash2_selector_score_args;

/*
 * One fixed eight-token sublayer launch. Attention is linked directly to the
 * model-specific convolution + pinned-FlashInfer capsule; the remaining MLP
 * dependency owns mlp_conv, gate/up/SiLU/down, and mlp_conv finish. The model
 * coordinator owns residual/RMSNorm sequencing around these calls.
 */
typedef struct q27_dflash2_sublayer_call {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t layer_index;
  uint32_t batch_size;
  uint32_t token_count;
  uint32_t reserved;
  const q27_dflash2_layer_weights* weights;
  const uint64_t* positions_u64;
  const void* input_bf16;
  void* output_bf16;
  q27_dflash2_state_view* state;
  void* workspace;
  uint64_t workspace_bytes;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_sublayer_call;

typedef q27_dflash2_status (*q27_dflash2_mlp_hook)(
    const q27_dflash2_sublayer_call* call, void* user_data);

/*
 * Five-layer BF16 draft-forward coordinator. input_embeddings is
 * [batch,8,5120], borrowed from the target embedding. normalized, residual,
 * and sublayer_output are disjoint caller-owned [batch,8,5120] buffers.
 * final_hidden may alias normalized but no other input/scratch pointer.
 *
 * It performs every residual/RMSNorm transition itself and invokes the linked
 * fixed attention then fixed grouped-conv+dense MLP for layers 0..4. A null
 * mlp field selects that production model-specific MLP. A non-null field is a
 * development override retaining the same raw ABI. It never allocates or
 * synchronizes.
 */
typedef struct q27_dflash2_forward_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_dflash2_weights* weights;
  const void* input_embeddings_bf16;
  const uint64_t* positions_u64;
  void* normalized_bf16;
  void* residual_bf16;
  void* sublayer_output_bf16;
  void* final_hidden_bf16;
  q27_dflash2_state_view* state;
  void* workspace;
  uint64_t workspace_bytes;
  uint32_t batch_size;
  float rms_epsilon;
  q27_dflash2_mlp_hook mlp; /* Optional; null selects linked fixed MLP. */
  void* mlp_user_data;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_forward_args;

q27_dflash2_status q27_dflash2_project_context(
    const q27_dflash2_context_projection_args* args);
q27_dflash2_status q27_dflash2_project_selector_hidden(
    const q27_dflash2_selector_projection_args* args);
q27_dflash2_status q27_dflash2_score_selector(
    const q27_dflash2_selector_score_args* args);
q27_dflash2_status q27_dflash2_forward(
    const q27_dflash2_forward_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_MODEL_H_

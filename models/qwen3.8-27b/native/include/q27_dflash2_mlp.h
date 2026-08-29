#ifndef Q27_DFLASH2_MLP_H_
#define Q27_DFLASH2_MLP_H_

#include "q27_dflash2_model.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fixed BF16 dense MLP for the pinned DFlash2 draft.  This is batch=1, T=8,
 * K=5120, N=17408 only; it is not a generic linear/activation API.
 */
#define Q27_DFLASH2_MLP_ABI_VERSION 1u

#define Q27_DFLASH2_MLP_HIDDEN_BYTES                                      \
  (Q27_DFLASH2_BLOCK_SIZE * Q27_DFLASH2_HIDDEN_SIZE * 2ULL)
#define Q27_DFLASH2_MLP_INTERMEDIATE_BYTES                                \
  (Q27_DFLASH2_BLOCK_SIZE * Q27_DFLASH2_INTERMEDIATE_SIZE * 2ULL)
#define Q27_DFLASH2_MLP_DENSE_WORKSPACE_BYTES                             \
  (2ULL * Q27_DFLASH2_MLP_HIDDEN_BYTES +                                 \
   3ULL * Q27_DFLASH2_MLP_INTERMEDIATE_BYTES)
#define Q27_DFLASH2_MLP_MIN_CONV_WORKSPACE_BYTES                          \
  (Q27_DFLASH2_BLOCK_SIZE * Q27_DFLASH2_CONV_PROJECTION_SIZE * 2ULL)
#define Q27_DFLASH2_MLP_WORKSPACE_ALIGNMENT 256ULL

/*
 * Caller-owned dense buffers. All tensors are row-major BF16:
 * input/output [8,5120], gate/up/activated [8,17408]. The cuBLAS handle must
 * use host scalar pointer mode, already be warmed, and may not be shared with
 * another stream concurrently. No allocation or synchronization occurs.
 */
typedef struct q27_dflash2_mlp_dense_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const q27_dflash2_layer_weights* weights;
  const void* input_bf16;
  void* gate_bf16;
  void* up_bf16;
  void* activated_bf16;
  void* output_bf16;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_mlp_dense_args;

/*
 * SGLang's DFlash layer wraps the dense MLP in one dynamic grouped-conv pair:
 *
 *   prepared, finish_coefficients = mlp_conv.prepare(input)
 *   dense = mlp(prepared)
 *   output = mlp_conv.finish(dense, finish_coefficients)
 *
 * These callbacks keep that dependency explicit. conv_workspace is the same
 * stable caller-owned region for prepare and finish and may retain the finish
 * coefficients. At least 20,480 bytes are required for the exported
 * [8,2,2,320] BF16 coefficients. Implementations must be allocation-free and
 * graph-safe.
 */
typedef q27_dflash2_status (*q27_dflash2_mlp_conv_prepare_hook)(
    const q27_dflash2_sublayer_call* call, const void* input_bf16,
    void* prepared_bf16, void* conv_workspace, uint64_t conv_workspace_bytes,
    void* user_data);

typedef q27_dflash2_status (*q27_dflash2_mlp_conv_finish_hook)(
    const q27_dflash2_sublayer_call* call, const void* dense_output_bf16,
    void* output_bf16, void* conv_workspace, uint64_t conv_workspace_bytes,
    void* user_data);

/* Passed as user_data to q27_dflash2_mlp_forward_hook. */
typedef struct q27_dflash2_mlp_hook_config {
  uint32_t struct_size;
  uint32_t abi_version;
  q27_dflash2_mlp_conv_prepare_hook prepare_conv;
  q27_dflash2_mlp_conv_finish_hook finish_conv;
  void* conv_user_data;
  uint64_t conv_workspace_bytes;
} q27_dflash2_mlp_hook_config;

typedef struct q27_dflash2_mlp_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t prepared_input_offset;
  uint64_t gate_offset;
  uint64_t up_offset;
  uint64_t activated_offset;
  uint64_t dense_output_offset;
  uint64_t conv_workspace_offset;
  uint64_t dense_workspace_bytes;
  uint64_t min_conv_workspace_bytes;
  uint64_t workspace_alignment;
} q27_dflash2_mlp_layout;

q27_dflash2_status q27_dflash2_mlp_query_layout(
    q27_dflash2_mlp_layout* output);

/* Dense gate/up -> BF16 SiLU*up -> down only; no grouped convolution. */
q27_dflash2_status q27_dflash2_mlp_dense(
    const q27_dflash2_mlp_dense_args* args);

/*
 * Exact q27_dflash2_mlp_hook ABI for q27_dflash2_forward. Missing prepare or
 * finish callbacks return Q27_DFLASH2_UNIMPLEMENTED before any CUDA work.
 */
q27_dflash2_status q27_dflash2_mlp_forward_hook(
    const q27_dflash2_sublayer_call* call, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_MLP_H_

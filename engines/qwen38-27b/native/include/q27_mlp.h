#ifndef SPARKSERVE_Q27_MLP_H_
#define SPARKSERVE_Q27_MLP_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_MLP_ABI_VERSION 1u

typedef enum q27_mlp_status_code {
  Q27_MLP_OK = 0,
  Q27_MLP_INVALID_ARGUMENT = 1,
  Q27_MLP_KERNEL_ERROR = 2,
} q27_mlp_status_code;

typedef struct q27_mlp_status {
  int32_t code;
  const char* message;
} q27_mlp_status;

/*
 * Fixed decode scratch contract. Every offset is 256-byte aligned and is
 * relative to q27_mlp_decode_args::scratch. The gate/up quantized activation
 * is shared. The workspace is separate because the two pinned FlashInfer
 * GEMMs reuse it sequentially.
 */
typedef struct q27_mlp_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t scratch_bytes;
  uint64_t workspace_bytes;
  uint64_t packed_hidden_offset;
  uint64_t hidden_scales_offset;
  uint64_t gate_output_offset;
  uint64_t up_output_offset;
  uint64_t activated_offset;
  uint64_t packed_activated_offset;
  uint64_t activated_scales_offset;
} q27_mlp_layout;

typedef struct q27_mlp_decode_args {
  uint32_t struct_size;
  uint32_t abi_version;

  /* BF16 [5120]. May alias output_bf16 because down GEMM reads scratch. */
  const void* hidden_bf16;

  /* Real layer weights, raw ModelOpt packed E2M1 bytes. */
  const void* gate_weight_fp4_e2m1;
  const void* gate_weight_scales_e4m3_128x4;
  const float* gate_alpha;
  const void* up_weight_fp4_e2m1;
  const void* up_weight_scales_e4m3_128x4;
  const float* up_alpha;
  const void* down_weight_fp4_e2m1;
  const void* down_weight_scales_e4m3_128x4;
  const float* down_alpha;

  /* Device scalars: 1/input_scale. Gate and up share the first scalar. */
  const float* hidden_input_scale_inv;
  const float* activated_input_scale_inv;

  /* Caller-owned, fixed-address storage returned by q27_mlp_query. */
  void* scratch;
  uint64_t scratch_bytes;
  void* workspace;
  uint64_t workspace_bytes;

  /* BF16 [5120]. */
  void* output_bf16;
  void* cuda_stream;
} q27_mlp_decode_args;

q27_mlp_status q27_mlp_query(q27_mlp_layout* output);

/*
 * Fixed M=1 dense MLP:
 *   quantize(hidden) -> gate/up GEMMs -> SiLU(gate)*up
 *   -> quantize(activated) -> down GEMM.
 *
 * The call performs no allocation, synchronization, registry lookup, JIT, or
 * stream creation and is safe to place inside a CUDA graph capture.
 */
q27_mlp_status q27_mlp_decode(const q27_mlp_decode_args* args);

#ifdef __cplusplus
}
#endif

#endif  // SPARKSERVE_Q27_MLP_H_

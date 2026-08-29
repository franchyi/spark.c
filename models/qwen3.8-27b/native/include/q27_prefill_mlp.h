#ifndef Q27_PREFILL_MLP_H_
#define Q27_PREFILL_MLP_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_PREFILL_MLP_ABI_VERSION 1u

typedef enum q27_prefill_mlp_status_code {
  Q27_PREFILL_MLP_OK = 0,
  Q27_PREFILL_MLP_INVALID_ARGUMENT = 1,
  Q27_PREFILL_MLP_KERNEL_ERROR = 2,
  Q27_PREFILL_MLP_CUDA_ERROR = 3,
} q27_prefill_mlp_status_code;

typedef struct q27_prefill_mlp_status {
  int32_t code;
  const char* message;
} q27_prefill_mlp_status;

typedef struct q27_prefill_mlp_layout {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t tokens;
  uint32_t reserved;
  uint64_t scratch_bytes;
  uint64_t scratch_alignment;
  uint64_t workspace_bytes;
  uint64_t workspace_alignment;
  uint64_t gate_up_output_offset;
  uint64_t activated_offset;
  uint64_t packed_input_offset;
  uint64_t input_scales_offset;
} q27_prefill_mlp_layout;

/*
 * Fixed Qwen3.8-27B batched dense MLP:
 *
 *   BF16 [M,5120]
 *     -> merged ModelOpt NVFP4 gate/up [M,2,17408]
 *     -> BF16 round(SiLU(gate) * up) [M,17408]
 *     -> ModelOpt NVFP4 down [M,5120]
 *
 * M is exactly 128 or 512. Gate/up weights and 128x4 scale matrices are the
 * load-time contiguous merge already required by the decode capsule. All
 * pointers and byte capacities are caller-owned CUDA-visible memory. The hot
 * call allocates and synchronizes nothing and has no M=1 fallback.
 */
typedef struct q27_prefill_mlp_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t tokens;
  uint32_t reserved;

  const void* input_bf16;
  uint64_t input_bf16_bytes;
  const void* gate_up_weight_fp4_e2m1;
  uint64_t gate_up_weight_bytes;
  const void* gate_up_weight_scales_e4m3_128x4;
  uint64_t gate_up_weight_scale_bytes;
  const float* hidden_global_scale_inv;
  const float* gate_up_alpha;

  const void* down_weight_fp4_e2m1;
  uint64_t down_weight_bytes;
  const void* down_weight_scales_e4m3_128x4;
  uint64_t down_weight_scale_bytes;
  const float* activated_global_scale_inv;
  const float* down_alpha;

  void* output_bf16;
  uint64_t output_bf16_bytes;
  void* scratch;
  uint64_t scratch_bytes;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_prefill_mlp_args;

q27_prefill_mlp_status q27_prefill_mlp_query(
    uint32_t tokens, q27_prefill_mlp_layout* output);
q27_prefill_mlp_status q27_prefill_mlp_forward(
    const q27_prefill_mlp_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_PREFILL_MLP_H_

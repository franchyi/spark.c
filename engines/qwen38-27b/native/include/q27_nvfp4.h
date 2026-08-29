#ifndef SPARKSERVE_Q27_NVFP4_H_
#define SPARKSERVE_Q27_NVFP4_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_NVFP4_ABI_VERSION 1u

typedef enum q27_nvfp4_status_code {
  Q27_NVFP4_OK = 0,
  Q27_NVFP4_INVALID_ARGUMENT = 1,
  Q27_NVFP4_UNSUPPORTED_SHAPE = 2,
  Q27_NVFP4_CUDA_ERROR = 3,
  Q27_NVFP4_INTERNAL_ERROR = 4,
} q27_nvfp4_status_code;

typedef struct q27_nvfp4_status {
  int32_t code;
  const char* message;
} q27_nvfp4_status;

typedef enum q27_nvfp4_projection {
  Q27_NVFP4_GATE = 0,
  Q27_NVFP4_UP = 1,
  Q27_NVFP4_DOWN = 2,
} q27_nvfp4_projection;

typedef struct q27_nvfp4_shape {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t n;
  uint32_t k;
  uint64_t packed_input_bytes;
  uint64_t input_scale_bytes;
  uint64_t packed_weight_bytes;
  uint64_t weight_scale_bytes;
  uint64_t output_bytes;
  uint64_t workspace_bytes;
} q27_nvfp4_shape;

typedef struct q27_nvfp4_project_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t projection;
  uint32_t reserved;

  /* Device pointers. The decode capsule is deliberately fixed to M=1. */
  const void* input_bf16;
  const float* input_global_scale_inv;
  const void* weight_fp4_e2m1;
  const void* weight_scales_e4m3_128x4;
  const float* alpha;

  /* Caller-owned, fixed-address graph scratch/output. */
  void* packed_input_fp4_e2m1;
  void* input_scales_e4m3_128x4;
  void* output_bf16;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_nvfp4_project_args;

typedef struct q27_nvfp4_quantize_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t projection;
  uint32_t reserved;
  const void* input_bf16;
  const float* input_global_scale_inv;
  void* packed_input_fp4_e2m1;
  void* input_scales_e4m3_128x4;
  void* cuda_stream;
} q27_nvfp4_quantize_args;

typedef struct q27_nvfp4_gemm_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t projection;
  uint32_t reserved;
  const void* packed_input_fp4_e2m1;
  const void* input_scales_e4m3_128x4;
  const void* weight_fp4_e2m1;
  const void* weight_scales_e4m3_128x4;
  const float* alpha;
  void* output_bf16;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_nvfp4_gemm_args;

/*
 * Returns the only physical shapes accepted by the q27 dense MLP capsule:
 * gate/up [17408, 5120], down [5120, 17408]. Weight scales must already be
 * converted from the checkpoint's row-major E4M3 matrix to CUTLASS 128x4
 * order by the offline sidecar builder. input_global_scale_inv and alpha are
 * device FP32 scalars equal to 1/input_scale and
 * input_scale*weight_scale_2 respectively.
 */
q27_nvfp4_status q27_nvfp4_query(uint32_t projection,
                                 q27_nvfp4_shape* output);

/*
 * Quantize one BF16 row with a pinned FlashInfer CuTe SM121 AOT object, then
 * run the matching pinned FlashInfer/CUTLASS NVFP4 GEMM specialization.
 */
q27_nvfp4_status q27_nvfp4_project(
    const q27_nvfp4_project_args* args);

/*
 * Split form used by the fixed q27 graph. Quantize hidden state once with
 * Q27_NVFP4_GATE, then call GEMM for both GATE and UP with the same packed
 * activation. This removes one complete activation-quantize launch per layer.
 */
q27_nvfp4_status q27_nvfp4_quantize(
    const q27_nvfp4_quantize_args* args);
q27_nvfp4_status q27_nvfp4_gemm(const q27_nvfp4_gemm_args* args);

#ifdef __cplusplus
}
#endif

#endif  // SPARKSERVE_Q27_NVFP4_H_

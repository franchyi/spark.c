#ifndef Q27_PREFILL_FP8_H_
#define Q27_PREFILL_FP8_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed-shape batched FP8 projection ABI for the Qwen3.8-27B prefill path. */
#define Q27_PREFILL_FP8_ABI_VERSION 1u

typedef enum q27_prefill_fp8_status_code {
  Q27_PREFILL_FP8_OK = 0,
  Q27_PREFILL_FP8_INVALID_ARGUMENT = 1,
  Q27_PREFILL_FP8_UNSUPPORTED_SHAPE = 2,
  Q27_PREFILL_FP8_CUDA_ERROR = 3,
  Q27_PREFILL_FP8_CUBLAS_ERROR = 4,
  Q27_PREFILL_FP8_INTERNAL_ERROR = 5,
} q27_prefill_fp8_status_code;

typedef struct q27_prefill_fp8_status {
  int32_t code;
  const char* message;
} q27_prefill_fp8_status;

typedef struct q27_prefill_fp8_shape {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t reserved;
  uint64_t input_bf16_bytes;
  uint64_t quantized_input_bytes;
  uint64_t packed_weight_bytes;
  uint64_t output_bf16_bytes;
  uint64_t workspace_bytes;
  uint64_t workspace_alignment;
} q27_prefill_fp8_shape;

typedef struct q27_prefill_fp8_plan_config {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t fast_accum; /* 0 matches the accurate torch._scaled_mm default. */
  uint64_t workspace_bytes;
} q27_prefill_fp8_plan_config;

typedef struct q27_prefill_fp8_plan q27_prefill_fp8_plan;

typedef struct q27_prefill_fp8_project_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* input_bf16;               /* Row-major [M,K]. */
  uint64_t input_bf16_bytes;
  const float* input_scale;             /* Device scalar, static calibration. */
  const void* weight_fp8_e4m3;          /* Row-major [N,K]. */
  uint64_t packed_weight_bytes;
  const float* weight_scale;            /* Device scalar. */
  void* quantized_input_fp8_e4m3;       /* Row-major [M,K]. */
  uint64_t quantized_input_bytes;
  void* output_bf16;                    /* Row-major [M,N]. */
  uint64_t output_bf16_bytes;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_prefill_fp8_project_args;

/* Supported M is 8..8192; N,K must be one exact Q27 FP8 projection shape. */
q27_prefill_fp8_status q27_prefill_fp8_query(
    uint32_t m, uint32_t n, uint32_t k, q27_prefill_fp8_shape* output);

/* Plan creation owns the cuBLASLt descriptors/handle; it does no CUDA work. */
q27_prefill_fp8_status q27_prefill_fp8_plan_create(
    const q27_prefill_fp8_plan_config* config,
    q27_prefill_fp8_plan** output);
void q27_prefill_fp8_plan_destroy(q27_prefill_fp8_plan* plan);

/* Allocation-free hot call: static BF16 quantization followed by one FP8 GEMM. */
q27_prefill_fp8_status q27_prefill_fp8_project(
    q27_prefill_fp8_plan* plan,
    const q27_prefill_fp8_project_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_PREFILL_FP8_H_

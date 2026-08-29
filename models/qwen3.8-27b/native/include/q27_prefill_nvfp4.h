#ifndef Q27_PREFILL_NVFP4_H_
#define Q27_PREFILL_NVFP4_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_PREFILL_NVFP4_ABI_VERSION 1u

enum {
  Q27_PREFILL_NVFP4_M128 = 128,
  Q27_PREFILL_NVFP4_M512 = 512,
  Q27_PREFILL_NVFP4_SCALE_GROUP = 16,
  Q27_PREFILL_NVFP4_HIDDEN_SIZE = 5120,
  Q27_PREFILL_NVFP4_INTERMEDIATE_SIZE = 17408,
};

typedef enum q27_prefill_nvfp4_status_code {
  Q27_PREFILL_NVFP4_OK = 0,
  Q27_PREFILL_NVFP4_INVALID_ARGUMENT = 1,
  Q27_PREFILL_NVFP4_UNSUPPORTED_SHAPE = 2,
  Q27_PREFILL_NVFP4_CUDA_ERROR = 3,
  Q27_PREFILL_NVFP4_INTERNAL_ERROR = 4,
  Q27_PREFILL_NVFP4_UNIMPLEMENTED = 5,
} q27_prefill_nvfp4_status_code;

typedef struct q27_prefill_nvfp4_status {
  int32_t code;
  const char* message;
} q27_prefill_nvfp4_status;

typedef enum q27_prefill_nvfp4_projection {
  Q27_PREFILL_NVFP4_GATE = 0,
  Q27_PREFILL_NVFP4_UP = 1,
  Q27_PREFILL_NVFP4_DOWN = 2,
  Q27_PREFILL_NVFP4_GATE_UP = 3,
} q27_prefill_nvfp4_projection;

typedef struct q27_prefill_nvfp4_shape {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t padded_scale_rows;
  uint32_t scale_group;
  uint32_t reserved;
  uint64_t input_bf16_bytes;
  uint64_t packed_input_bytes;
  uint64_t input_scale_bytes;
  uint64_t packed_weight_bytes;
  uint64_t weight_scale_bytes;
  uint64_t output_bf16_bytes;
  uint64_t workspace_bytes;
  uint64_t workspace_alignment;
} q27_prefill_nvfp4_shape;

/* Tensor byte sizes are exact. Workspace bytes are caller capacity. */
typedef struct q27_prefill_nvfp4_quantize_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t m;
  uint32_t projection;
  const void* input_bf16;               /* [M,K], row-major. */
  uint64_t input_bf16_bytes;
  const float* input_global_scale_inv;  /* Device FP32 scalar. */
  void* packed_input_fp4_e2m1;          /* [M,K/2], row-major bytes. */
  uint64_t packed_input_bytes;
  void* input_scales_e4m3_128x4;
  uint64_t input_scale_bytes;
  void* cuda_stream;
} q27_prefill_nvfp4_quantize_args;

typedef struct q27_prefill_nvfp4_gemm_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t m;
  uint32_t projection;
  const void* packed_input_fp4_e2m1;
  uint64_t packed_input_bytes;
  const void* input_scales_e4m3_128x4;
  uint64_t input_scale_bytes;
  const void* weight_fp4_e2m1;
  uint64_t packed_weight_bytes;
  const void* weight_scales_e4m3_128x4;
  uint64_t weight_scale_bytes;
  const float* alpha;                   /* Device FP32 scale product. */
  void* output_bf16;                    /* [M,N], row-major. */
  uint64_t output_bf16_bytes;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_prefill_nvfp4_gemm_args;

typedef struct q27_prefill_nvfp4_project_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t m;
  uint32_t projection;
  const void* input_bf16;
  uint64_t input_bf16_bytes;
  const float* input_global_scale_inv;
  const void* weight_fp4_e2m1;
  uint64_t packed_weight_bytes;
  const void* weight_scales_e4m3_128x4;
  uint64_t weight_scale_bytes;
  const float* alpha;
  void* packed_input_fp4_e2m1;
  uint64_t packed_input_bytes;
  void* input_scales_e4m3_128x4;
  uint64_t input_scale_bytes;
  void* output_bf16;
  uint64_t output_bf16_bytes;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_prefill_nvfp4_project_args;

q27_prefill_nvfp4_status q27_prefill_nvfp4_query(
    uint32_t m, uint32_t projection, q27_prefill_nvfp4_shape* output);
q27_prefill_nvfp4_status q27_prefill_nvfp4_quantize(
    const q27_prefill_nvfp4_quantize_args* args);
q27_prefill_nvfp4_status q27_prefill_nvfp4_gemm(
    const q27_prefill_nvfp4_gemm_args* args);
q27_prefill_nvfp4_status q27_prefill_nvfp4_project(
    const q27_prefill_nvfp4_project_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_PREFILL_NVFP4_H_

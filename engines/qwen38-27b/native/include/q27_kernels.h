#ifndef SPARKSERVE_Q27_KERNELS_H_
#define SPARKSERVE_Q27_KERNELS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_KERNEL_ABI_VERSION 1u

typedef enum q27_kernel_status_code {
  Q27_KERNEL_OK = 0,
  Q27_KERNEL_INVALID_ARGUMENT = 1,
  Q27_KERNEL_UNSUPPORTED_SHAPE = 2,
  Q27_KERNEL_CUDA_ERROR = 3,
} q27_kernel_status_code;

typedef struct q27_kernel_status {
  int32_t code;
  const char* message;
} q27_kernel_status;

typedef struct q27_fp8_project_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t n;
  uint32_t k;
  const void* input_bf16;
  const void* weight_fp8_e4m3;
  const float* input_scale;
  const float* weight_scale;
  void* quantized_input_fp8_e4m3;
  void* output_bf16;
  void* cuda_stream;
} q27_fp8_project_args;

typedef struct q27_embedding_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t token;
  uint32_t vocabulary;
  uint32_t hidden_size;
  uint32_t reserved;
  const void* weight_bf16;
  void* output_bf16;
  void* cuda_stream;
} q27_embedding_args;

typedef struct q27_norm_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t hidden_size;
  uint32_t has_residual;
  float epsilon;
  uint32_t reserved;
  const void* input_bf16;
  const void* residual_bf16;
  const void* checkpoint_weight_bf16;
  void* output_bf16;
  void* residual_output_bf16;
  void* cuda_stream;
} q27_norm_args;

typedef struct q27_silu_mul_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t elements;
  uint32_t reserved;
  const void* gate_bf16;
  const void* up_bf16;
  void* output_bf16;
  void* cuda_stream;
} q27_silu_mul_args;

typedef struct q27_lm_head_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t vocabulary;
  uint32_t hidden_size;
  const void* hidden_bf16;
  const void* weight_bf16;
  float* logits_f32;
  void* cublas_handle;
  void* cuda_stream;
} q27_lm_head_args;

typedef struct q27_argmax_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t elements;
  uint32_t scratch_elements;
  const float* logits_f32;
  float* scratch_values_f32;
  int32_t* scratch_indices_i32;
  int32_t* output_token_i32;
  void* cuda_stream;
} q27_argmax_args;

/*
 * Decode-only M=1 projection. Supported shapes are exactly the q27 FP8
 * projection pairs; no dynamic GEMM dispatcher is retained in the process.
 */
q27_kernel_status q27_fp8_project(const q27_fp8_project_args* args);
q27_kernel_status q27_embedding(const q27_embedding_args* args);
q27_kernel_status q27_gemma_rmsnorm(const q27_norm_args* args);
q27_kernel_status q27_silu_mul(const q27_silu_mul_args* args);
q27_kernel_status q27_lm_head(const q27_lm_head_args* args);
q27_kernel_status q27_argmax(const q27_argmax_args* args);

#ifdef __cplusplus
}
#endif

#endif

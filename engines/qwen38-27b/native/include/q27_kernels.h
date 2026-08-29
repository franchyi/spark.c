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

/*
 * Decode-only M=1 projection. Supported shapes are exactly the q27 FP8
 * projection pairs; no dynamic GEMM dispatcher is retained in the process.
 */
q27_kernel_status q27_fp8_project(const q27_fp8_project_args* args);

#ifdef __cplusplus
}
#endif

#endif

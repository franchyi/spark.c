#ifndef SPARKSERVE_Q27_BF16_AB_H_
#define SPARKSERVE_Q27_BF16_AB_H_

#include "q27_kernels.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The two unquantized GDN gate projections in every recurrent q27 layer.
 *
 * Both checkpoint weights are row-major BF16 [48, 5120]. All storage and the
 * cuBLAS handle are caller-owned. The call only enqueues work on cuda_stream;
 * it does not allocate, synchronize, or retain framework state.
 */
typedef struct q27_bf16_ab_project_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t hidden_size;
  uint32_t value_heads;
  const void* input_bf16;
  const void* weight_a_bf16;
  const void* weight_b_bf16;
  void* output_a_bf16;
  void* output_b_bf16;
  void* cublas_handle;
  void* cuda_stream;
} q27_bf16_ab_project_args;

q27_kernel_status q27_bf16_ab_project(
    const q27_bf16_ab_project_args* args);

#ifdef __cplusplus
}
#endif

#endif

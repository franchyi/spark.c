#ifndef Q27_DFLASH2_CONV_H_
#define Q27_DFLASH2_CONV_H_

#include <stdint.h>

#include "q27_dflash2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_DFLASH2_CONV_ABI_VERSION 1u

#define Q27_DFLASH2_CONV_COEFFICIENT_BYTES \
  (8ULL * 1280ULL * 2ULL)

/*
 * Project one DFlash block [8,5120] through kernel_projection.weight
 * [1280,5120], retain all [side=2,tap=2,group=320] BF16 deltas in
 * coefficients_bf16, and apply the side-zero grouped convolution.
 *
 * base_kernel is the exact checkpoint layout [side=2,tap=2,channel=5120].
 * All buffers are device pointers and must be pairwise disjoint. The caller
 * owns a pre-created, host-scalar-mode cuBLAS handle warmed before graph capture.
 */
typedef struct q27_dflash2_conv_prepare_args {
  uint32_t struct_size;
  uint32_t abi_version;
  q27_dflash2_weight_view base_kernel;
  q27_dflash2_weight_view kernel_projection;
  const void* input_bf16;
  void* coefficients_bf16;
  void* output_bf16;
  void* cublas_handle;
  void* cuda_stream;
} q27_dflash2_conv_prepare_args;

/*
 * Apply side one to a sublayer output using the retained coefficients from the
 * matching prepare call. No projection or allocation occurs here.
 */
typedef struct q27_dflash2_conv_finish_args {
  uint32_t struct_size;
  uint32_t abi_version;
  q27_dflash2_weight_view base_kernel;
  const void* input_bf16;
  const void* coefficients_bf16;
  void* output_bf16;
  void* cuda_stream;
} q27_dflash2_conv_finish_args;

q27_dflash2_status q27_dflash2_conv_prepare(
    const q27_dflash2_conv_prepare_args* args);
q27_dflash2_status q27_dflash2_conv_finish(
    const q27_dflash2_conv_finish_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_DFLASH2_CONV_H_

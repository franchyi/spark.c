#ifndef SPARKSERVE_QWEN_PLE_BLOCK_API_H_
#define SPARKSERVE_QWEN_PLE_BLOCK_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_QWEN_PLE_BLOCK_ABI_VERSION 1u

// Fixed one-token Qwen3.8 Flash-Next PLE block. The caller gathers the 16
// selected FP8 rows into one BF16 [2560] embedding before this launch.
// `conv_state` is the persistent BF16 [10240, 9] depthwise-convolution state.
typedef struct SparkServeQwenPleBlockArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t tokens;
  uint32_t reserved;
  const void* hidden_states;
  const void* embedding;
  const void* key_weight;
  const void* value_weight;
  const void* norm_key_weight;
  const void* norm_query_weight;
  const void* norm_conv_weight;
  const void* conv_weight;
  void* conv_state;
  void* key_scratch;
  void* value_scratch;
  void* gated_scratch;
  void* normed_scratch;
  void* output;
  void* cublas_handle;
  void* cuda_stream;
} SparkServeQwenPleBlockArgs;

SparkServeStatus sparkserve_qwen_ple_block_launch(
    const SparkServeQwenPleBlockArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_QWEN_PLE_BLOCK_API_H_

#ifndef SPARKSERVE_QWEN_GDN_AUX_API_H_
#define SPARKSERVE_QWEN_GDN_AUX_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_QWEN_GDN_AUX_ABI_VERSION 1u

typedef struct SparkServeQwenBf16ToF32Args {
  uint32_t struct_size;
  uint32_t abi_version;
  const uint16_t* input_bf16;
  float* output_f32;
  uint64_t elements;
  void* cuda_stream;
} SparkServeQwenBf16ToF32Args;

SparkServeStatus sparkserve_qwen_bf16_to_f32_launch(
    const SparkServeQwenBf16ToF32Args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_QWEN_GDN_AUX_API_H_

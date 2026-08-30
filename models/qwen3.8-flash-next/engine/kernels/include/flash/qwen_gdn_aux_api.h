#ifndef FLASH_QWEN_GDN_AUX_API_H_
#define FLASH_QWEN_GDN_AUX_API_H_

#include <stdint.h>

#include "flash/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FLASH_QWEN_GDN_AUX_ABI_VERSION 1u

typedef struct FlashQwenBf16ToF32Args {
  uint32_t struct_size;
  uint32_t abi_version;
  const uint16_t* input_bf16;
  float* output_f32;
  uint64_t elements;
  void* cuda_stream;
} FlashQwenBf16ToF32Args;

FlashStatus flash_qwen_bf16_to_f32_launch(
    const FlashQwenBf16ToF32Args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FLASH_QWEN_GDN_AUX_API_H_

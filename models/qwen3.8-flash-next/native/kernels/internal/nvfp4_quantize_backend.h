#ifndef FLASH_INTERNAL_NVFP4_QUANTIZE_BACKEND_H_
#define FLASH_INTERNAL_NVFP4_QUANTIZE_BACKEND_H_

#include "flash/kernel_api.h"

FlashStatus flash_flashinfer_cute_segmented_nvfp4_quantize_launch(
    const FlashSegmentedNvfp4QuantizeArgs* args);

#endif  // FLASH_INTERNAL_NVFP4_QUANTIZE_BACKEND_H_

#ifndef FLASH_INTERNAL_NVFP4_SILU_BACKEND_H_
#define FLASH_INTERNAL_NVFP4_SILU_BACKEND_H_

#include "flash/kernel_api.h"

FlashStatus flash_flashinfer_cute_silu_nvfp4_launch(
    const FlashSiluNvfp4Args* args);

FlashStatus flash_flashinfer_cute_segmented_silu_nvfp4_launch(
    const FlashSegmentedSiluNvfp4Args* args);

#endif  // FLASH_INTERNAL_NVFP4_SILU_BACKEND_H_

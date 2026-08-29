#ifndef FLASH_INTERNAL_MHC_BACKEND_H_
#define FLASH_INTERNAL_MHC_BACKEND_H_

#include "flash/kernel_api.h"

FlashStatus flash_sglang_cublas_mhc_mix_cuda_launch(
    const FlashMhcArgs* args);

FlashStatus flash_sglang_cublas_mhc_combine_cuda_launch(
    const FlashMhcArgs* args);

#endif  // FLASH_INTERNAL_MHC_BACKEND_H_

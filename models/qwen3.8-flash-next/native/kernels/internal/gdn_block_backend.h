#pragma once

#include "flash/kernel_api.h"

FlashStatus flash_sglang_cublas_gdn_prepare_cuda_launch(
    const FlashGdnBlockArgs* args);

FlashStatus flash_sglang_cublas_gdn_finish_cuda_launch(
    const FlashGdnBlockArgs* args);

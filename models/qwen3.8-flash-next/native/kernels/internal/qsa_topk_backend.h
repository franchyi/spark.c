#pragma once

#include "flash/kernel_api.h"

FlashStatus flash_sglang_qsa_topk_cuda_launch(
    const FlashQsaTopkArgs* args);

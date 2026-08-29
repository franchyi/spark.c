#ifndef FLASH_INTERNAL_MOE_JOIN_BACKEND_H_
#define FLASH_INTERNAL_MOE_JOIN_BACKEND_H_

#include "flash/kernel_api.h"

FlashStatus flash_sglang_fused_moe_join_cuda_launch(
    const FlashMoeJoinArgs* args);

#endif  // FLASH_INTERNAL_MOE_JOIN_BACKEND_H_

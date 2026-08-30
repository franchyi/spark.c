#pragma once

#include "flash/kernel_api.h"

FlashStatus flash_sglang_cublas_moe_gate_cuda_launch(
    const FlashMoeGateArgs* args);

#pragma once

#include "flash/kernel_api.h"

FlashStatus flash_sglang_qsa_kv_pack_cuda_launch(
    const FlashQsaKvPackArgs* args);

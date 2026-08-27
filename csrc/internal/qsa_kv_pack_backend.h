#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_qsa_kv_pack_cuda_launch(
    const SparkServeQsaKvPackArgs* args);

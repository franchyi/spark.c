#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_qsa_topk_cuda_launch(
    const SparkServeQsaTopkArgs* args);

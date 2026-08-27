#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_qsa_expand_cuda_launch(
    const SparkServeQsaExpandArgs* args);

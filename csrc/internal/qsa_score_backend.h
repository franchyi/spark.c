#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_tilelang_qsa_score_cuda_launch(
    const SparkServeQsaScoreArgs* args);

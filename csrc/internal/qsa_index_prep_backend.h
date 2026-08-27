#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_qsa_index_prep_cuda_launch(
    const SparkServeQsaIndexPrepArgs* args);

#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_cublas_gdn_prepare_cuda_launch(
    const SparkServeGdnBlockArgs* args);

SparkServeStatus sparkserve_sglang_cublas_gdn_finish_cuda_launch(
    const SparkServeGdnBlockArgs* args);

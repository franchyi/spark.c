#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_cublas_shared_expert_cuda_launch(
    const SparkServeSharedExpertArgs* args);

#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_cublas_moe_gate_cuda_launch(
    const SparkServeMoeGateArgs* args);

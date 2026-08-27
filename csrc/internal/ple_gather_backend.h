#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_ple_gather_cuda_launch(
    const SparkServePleGatherArgs* args);

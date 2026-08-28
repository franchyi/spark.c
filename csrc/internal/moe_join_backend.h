#ifndef SPARKSERVE_INTERNAL_MOE_JOIN_BACKEND_H_
#define SPARKSERVE_INTERNAL_MOE_JOIN_BACKEND_H_

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_fused_moe_join_cuda_launch(
    const SparkServeMoeJoinArgs* args);

#endif  // SPARKSERVE_INTERNAL_MOE_JOIN_BACKEND_H_

#ifndef SPARKSERVE_INTERNAL_NVFP4_SILU_BACKEND_H_
#define SPARKSERVE_INTERNAL_NVFP4_SILU_BACKEND_H_

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_flashinfer_cute_silu_nvfp4_launch(
    const SparkServeSiluNvfp4Args* args);

#endif  // SPARKSERVE_INTERNAL_NVFP4_SILU_BACKEND_H_

#ifndef SPARKSERVE_INTERNAL_MHC_BACKEND_H_
#define SPARKSERVE_INTERNAL_MHC_BACKEND_H_

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_sglang_cublas_mhc_mix_cuda_launch(
    const SparkServeMhcArgs* args);

SparkServeStatus sparkserve_sglang_cublas_mhc_combine_cuda_launch(
    const SparkServeMhcArgs* args);

#endif  // SPARKSERVE_INTERNAL_MHC_BACKEND_H_

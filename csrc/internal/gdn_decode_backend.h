#ifndef SPARKSERVE_INTERNAL_GDN_DECODE_BACKEND_H_
#define SPARKSERVE_INTERNAL_GDN_DECODE_BACKEND_H_

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

SparkServeStatus sparkserve_gdn_decode_cuda_launch(
    const SparkServeGdnDecodeArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_INTERNAL_GDN_DECODE_BACKEND_H_

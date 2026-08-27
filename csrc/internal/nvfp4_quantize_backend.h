#ifndef SPARKSERVE_INTERNAL_NVFP4_QUANTIZE_BACKEND_H_
#define SPARKSERVE_INTERNAL_NVFP4_QUANTIZE_BACKEND_H_

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_flashinfer_cute_segmented_nvfp4_quantize_launch(
    const SparkServeSegmentedNvfp4QuantizeArgs* args);

#endif  // SPARKSERVE_INTERNAL_NVFP4_QUANTIZE_BACKEND_H_

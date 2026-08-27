#pragma once

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_flashinfer_xqa_decode_cuda_launch(
    const SparkServeQsaDecodeArgs* args);

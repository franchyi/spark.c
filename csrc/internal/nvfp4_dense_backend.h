#ifndef SPARKSERVE_NVFP4_DENSE_BACKEND_H_
#define SPARKSERVE_NVFP4_DENSE_BACKEND_H_

#include <stddef.h>

#include "sparkserve/kernel_api.h"

// Thin framework-free adapter over the pinned FlashInfer CUTLASS SM120/121
// template. The donor owns the arithmetic kernel; SparkServe owns validation,
// scheduling, storage, and the stable raw-pointer ABI.
size_t sparkserve_flashinfer_nvfp4_workspace_bytes(
    const SparkServeDenseNvfp4Plan* plan);

SparkServeStatus sparkserve_flashinfer_nvfp4_launch(
    const SparkServeDenseNvfp4Args* args);

#endif  // SPARKSERVE_NVFP4_DENSE_BACKEND_H_

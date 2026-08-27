#ifndef SPARKSERVE_INTERNAL_NVFP4_GROUPED_BACKEND_H_
#define SPARKSERVE_INTERNAL_NVFP4_GROUPED_BACKEND_H_

#include <stddef.h>

#include "sparkserve/kernel_api.h"

// FlashInfer uses two separate 32 MiB scratch arenas for this donor kernel.
size_t sparkserve_flashinfer_grouped_nvfp4_int_workspace_bytes(void);
size_t sparkserve_flashinfer_grouped_nvfp4_float_workspace_bytes(void);

SparkServeStatus sparkserve_flashinfer_grouped_nvfp4_launch(
    const SparkServeGroupedNvfp4Args* args);

#endif  // SPARKSERVE_INTERNAL_NVFP4_GROUPED_BACKEND_H_

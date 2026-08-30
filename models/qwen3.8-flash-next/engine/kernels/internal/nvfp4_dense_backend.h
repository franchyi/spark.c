#ifndef FLASH_NVFP4_DENSE_BACKEND_H_
#define FLASH_NVFP4_DENSE_BACKEND_H_

#include <stddef.h>

#include "flash/kernel_api.h"

// Thin framework-free adapter over the pinned FlashInfer CUTLASS SM120/121
// template. The donor owns the arithmetic kernel; Flash owns validation,
// scheduling, storage, and the stable raw-pointer ABI.
size_t flash_flashinfer_nvfp4_workspace_bytes(
    const FlashDenseNvfp4Plan* plan);

FlashStatus flash_flashinfer_nvfp4_launch(
    const FlashDenseNvfp4Args* args);

#endif  // FLASH_NVFP4_DENSE_BACKEND_H_

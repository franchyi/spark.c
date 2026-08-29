#ifndef FLASH_INTERNAL_NVFP4_GROUPED_BACKEND_H_
#define FLASH_INTERNAL_NVFP4_GROUPED_BACKEND_H_

#include <stddef.h>

#include "flash/kernel_api.h"

// FlashInfer uses two separate 32 MiB scratch arenas for this donor kernel.
size_t flash_flashinfer_grouped_nvfp4_int_workspace_bytes(void);
size_t flash_flashinfer_grouped_nvfp4_float_workspace_bytes(void);

FlashStatus flash_flashinfer_grouped_nvfp4_launch(
    const FlashGroupedNvfp4Args* args);

#endif  // FLASH_INTERNAL_NVFP4_GROUPED_BACKEND_H_

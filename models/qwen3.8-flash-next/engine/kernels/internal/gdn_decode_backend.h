#ifndef FLASH_INTERNAL_GDN_DECODE_BACKEND_H_
#define FLASH_INTERNAL_GDN_DECODE_BACKEND_H_

#include "flash/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

FlashStatus flash_gdn_decode_cuda_launch(
    const FlashGdnDecodeArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FLASH_INTERNAL_GDN_DECODE_BACKEND_H_

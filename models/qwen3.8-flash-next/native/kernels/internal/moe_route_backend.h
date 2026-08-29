#ifndef FLASH_INTERNAL_MOE_ROUTE_BACKEND_H_
#define FLASH_INTERNAL_MOE_ROUTE_BACKEND_H_

#include "flash/kernel_api.h"

FlashStatus flash_flashinfer_moe_route_dispatch(
    const FlashMoeRouteArgs* args);

FlashStatus flash_flashinfer_moe_route_finalize(
    const FlashMoeRouteArgs* args);

#endif  // FLASH_INTERNAL_MOE_ROUTE_BACKEND_H_

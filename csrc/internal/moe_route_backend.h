#ifndef SPARKSERVE_INTERNAL_MOE_ROUTE_BACKEND_H_
#define SPARKSERVE_INTERNAL_MOE_ROUTE_BACKEND_H_

#include "sparkserve/kernel_api.h"

SparkServeStatus sparkserve_flashinfer_moe_route_dispatch(
    const SparkServeMoeRouteArgs* args);

SparkServeStatus sparkserve_flashinfer_moe_route_finalize(
    const SparkServeMoeRouteArgs* args);

#endif  // SPARKSERVE_INTERNAL_MOE_ROUTE_BACKEND_H_

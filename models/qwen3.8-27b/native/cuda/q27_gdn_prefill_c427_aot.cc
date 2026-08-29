// SPDX-License-Identifier: Apache-2.0
// Python-free CUDA Driver adapter for pinned c427 Triton GDN cubins.

#include "q27_gdn_prefill_c427_aot.h"

#include <cuda.h>

#include <cstdint>
#include <new>
#include <string>

struct q27_c427_gdn_aot_kernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  uint32_t block_x = 0;
  uint32_t dynamic_shared_bytes = 0;
};

namespace {

thread_local std::string g_error;

q27_c427_gdn_aot_status Ok() { return {Q27_C427_GDN_AOT_OK, "ok"}; }

q27_c427_gdn_aot_status Invalid(const char* message) {
  return {Q27_C427_GDN_AOT_INVALID_ARGUMENT, message};
}

q27_c427_gdn_aot_status DriverError(CUresult result, const char* prefix) {
  const char* name = nullptr;
  const char* detail = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &detail);
  g_error.assign(prefix);
  if (name != nullptr) {
    g_error.append(name);
    g_error.append(": ");
  }
  g_error.append(detail != nullptr ? detail : "unknown CUDA Driver error");
  return {Q27_C427_GDN_AOT_CUDA_ERROR, g_error.c_str()};
}

q27_c427_gdn_aot_status RequireSm121() {
  CUdevice device = 0;
  CUresult result = cuCtxGetDevice(&device);
  if (result != CUDA_SUCCESS) return DriverError(result, "resolve CUDA device: ");
  int major = 0;
  int minor = 0;
  result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                device);
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetAttribute(&minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  device);
  }
  if (result != CUDA_SUCCESS) return DriverError(result, "query CUDA device: ");
  if (major != 12 || minor != 1) {
    return {Q27_C427_GDN_AOT_INCOMPATIBLE_ARTIFACT,
            "c427 GDN cubins require SM121"};
  }
  return Ok();
}

}  // namespace

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_aot_kernel_create(
    const q27_c427_gdn_aot_kernel_desc* desc,
    q27_c427_gdn_aot_kernel** output) {
  if (output == nullptr) return Invalid("null c427 GDN kernel output");
  *output = nullptr;
  if (desc == nullptr || desc->struct_size != sizeof(*desc) ||
      desc->abi_version != Q27_C427_GDN_AOT_ABI_VERSION ||
      desc->cubin == nullptr || desc->cubin_bytes == 0 ||
      desc->symbol == nullptr || desc->symbol[0] == '\0' ||
      desc->num_warps == 0 || desc->num_warps > 32 ||
      desc->cluster_x != 1 || desc->cluster_y != 1 || desc->cluster_z != 1) {
    return Invalid("invalid or unsupported c427 GDN cubin descriptor");
  }
  q27_c427_gdn_aot_status compatible = RequireSm121();
  if (compatible.code != Q27_C427_GDN_AOT_OK) return compatible;

  auto* kernel = new (std::nothrow) q27_c427_gdn_aot_kernel;
  if (kernel == nullptr) return Invalid("cannot allocate c427 GDN kernel handle");
  CUresult result = cuModuleLoadData(&kernel->module, desc->cubin);
  if (result == CUDA_SUCCESS) {
    result = cuModuleGetFunction(&kernel->function, kernel->module, desc->symbol);
  }
  if (result == CUDA_SUCCESS && desc->dynamic_shared_bytes > 49152) {
    result = cuFuncSetAttribute(
        kernel->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(desc->dynamic_shared_bytes));
  }
  if (result != CUDA_SUCCESS) {
    if (kernel->module != nullptr) cuModuleUnload(kernel->module);
    delete kernel;
    return DriverError(result, "load c427 GDN cubin: ");
  }
  kernel->block_x = desc->num_warps * 32;
  kernel->dynamic_shared_bytes = desc->dynamic_shared_bytes;
  *output = kernel;
  return Ok();
}

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_aot_kernel_launch(
    q27_c427_gdn_aot_kernel* kernel,
    const q27_c427_gdn_aot_launch* launch) {
  if (kernel == nullptr || launch == nullptr ||
      launch->struct_size != sizeof(*launch) ||
      launch->abi_version != Q27_C427_GDN_AOT_ABI_VERSION ||
      launch->grid_x == 0 || launch->grid_y == 0 || launch->grid_z == 0 ||
      launch->kernel_params == nullptr) {
    return Invalid("invalid c427 GDN launch arguments");
  }
  CUresult result = cuLaunchKernel(
      kernel->function, launch->grid_x, launch->grid_y, launch->grid_z,
      kernel->block_x, 1, 1, kernel->dynamic_shared_bytes,
      reinterpret_cast<CUstream>(launch->cuda_stream), launch->kernel_params,
      nullptr);
  return result == CUDA_SUCCESS ? Ok()
                                : DriverError(result, "launch c427 GDN cubin: ");
}

extern "C" void q27_c427_gdn_aot_kernel_destroy(
    q27_c427_gdn_aot_kernel* kernel) {
  if (kernel == nullptr) return;
  if (kernel->module != nullptr) cuModuleUnload(kernel->module);
  delete kernel;
}

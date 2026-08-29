#include "flash/qwen_gdn_aux_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

thread_local std::string g_error;

__global__ void Bf16ToF32Kernel(const uint16_t* input, float* output,
                               uint64_t elements) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  __nv_bfloat16_raw raw;
  raw.x = input[index];
  output[index] = __bfloat162float(__nv_bfloat16(raw));
}

FlashStatus Invalid(const char* message) {
  return {FLASH_STATUS_INVALID_ARGUMENT, message};
}

}  // namespace

extern "C" FlashStatus flash_qwen_bf16_to_f32_launch(
    const FlashQwenBf16ToF32Args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_GDN_AUX_ABI_VERSION ||
      args->input_bf16 == nullptr || args->output_f32 == nullptr ||
      args->elements == 0) {
    return Invalid("Qwen BF16-to-FP32 conversion arguments are invalid");
  }
  constexpr uint32_t kThreads = 256;
  const uint64_t blocks = (args->elements + kThreads - 1) / kThreads;
  if (blocks > static_cast<uint64_t>(UINT32_MAX)) {
    return Invalid("Qwen BF16-to-FP32 conversion is too large");
  }
  Bf16ToF32Kernel<<<static_cast<uint32_t>(blocks), kThreads, 0,
                    static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->input_bf16, args->output_f32, args->elements);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    g_error.assign("Qwen BF16-to-FP32 conversion failed: ");
    g_error.append(cudaGetErrorString(error));
    return {FLASH_STATUS_INTERNAL, g_error.c_str()};
  }
  return {FLASH_STATUS_OK, "ok"};
}

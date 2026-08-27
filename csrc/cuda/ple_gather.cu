#include "internal/ple_gather_backend.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

__global__ void PleGatherKernel(const uint8_t* coherent_base,
                                const SparkServePleRowFragment* fragments,
                                __nv_bfloat16* output,
                                uint64_t output_row_stride_bytes,
                                uint32_t row_bytes,
                                uint16_t scale_bf16_bits) {
  const uint32_t row = blockIdx.x;
  const uint32_t column = threadIdx.x;
  if (column >= row_bytes) return;

  const SparkServePleRowFragment fragment = fragments[row];
  const uint64_t source_offset =
      column < fragment.first_bytes
          ? fragment.first_offset_bytes + column
          : fragment.second_offset_bytes + column - fragment.first_bytes;
  const auto* source = reinterpret_cast<const __nv_fp8_e4m3*>(
      coherent_base + source_offset);
  const __nv_bfloat16 value = static_cast<__nv_bfloat16>(*source);
  __nv_bfloat16_raw scale_raw;
  scale_raw.x = scale_bf16_bits;
  const __nv_bfloat16 scale(scale_raw);
  auto* output_row = reinterpret_cast<__nv_bfloat16*>(
      reinterpret_cast<uint8_t*>(output) +
      static_cast<uint64_t>(row) * output_row_stride_bytes);
  output_row[column] = __hmul(value, scale);
}

}  // namespace

SparkServeStatus sparkserve_sglang_ple_gather_cuda_launch(
    const SparkServePleGatherArgs* args) {
  constexpr uint32_t kThreads = 256;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  PleGatherKernel<<<args->plan.rows, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(args->coherent_base), args->fragments,
      static_cast<__nv_bfloat16*>(args->output),
      args->output_row_stride_bytes, args->plan.row_bytes,
      args->scale_bf16_bits);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("PLE gather launch failed: ", error);
  return Ok();
}

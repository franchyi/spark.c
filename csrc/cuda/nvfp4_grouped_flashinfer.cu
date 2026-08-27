/*
 * Framework-free adapter for FlashInfer's Apache-2.0 grouped SM120 NVFP4
 * GEMM. SparkServe supplies routed rows and expert offsets; this file only
 * instantiates and launches the pinned upstream CUTLASS arithmetic kernel.
 */

#include "internal/nvfp4_grouped_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <string>

#include "flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh"

namespace flashinfer {
namespace group_gemm {

INSTANTIATE_GROUP_GEMM_NVFP4_GROUPWISE_SM120(
    128, 128, 256, false, cutlass::float_e2m1_t,
    cutlass::float_e2m1_t, cutlass::float_ue4m3_t,
    cutlass::float_ue4m3_t, cutlass::bfloat16_t, float_e2m1_t,
    float_e2m1_t, float_ue4m3_t, float_ue4m3_t, bfloat16_t)

}  // namespace group_gemm
}  // namespace flashinfer

namespace {

constexpr size_t kOfficialWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;
thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus Internal(const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

}  // namespace

size_t sparkserve_flashinfer_grouped_nvfp4_int_workspace_bytes(void) {
  return kOfficialWorkspaceBytes;
}

size_t sparkserve_flashinfer_grouped_nvfp4_float_workspace_bytes(void) {
  return kOfficialWorkspaceBytes;
}

SparkServeStatus sparkserve_flashinfer_grouped_nvfp4_launch(
    const SparkServeGroupedNvfp4Args* args) {
  if (args->int_workspace == nullptr ||
      args->int_workspace_bytes < kOfficialWorkspaceBytes) {
    return Invalid("FlashInfer grouped NVFP4 integer workspace is too small");
  }
  if (args->float_workspace == nullptr ||
      args->float_workspace_bytes < kOfficialWorkspaceBytes) {
    return Invalid("FlashInfer grouped NVFP4 float workspace is too small");
  }

  int device_id = 0;
  cudaError_t error = cudaGetDevice(&device_id);
  if (error != cudaSuccess) {
    return Internal("cannot resolve grouped NVFP4 CUDA device: ",
                    cudaGetErrorString(error));
  }

  try {
    error = flashinfer::group_gemm::
        CutlassNVFP4GroupwiseScaledGroupGEMMSM120<
            128, 128, 256, false, cutlass::float_e2m1_t,
            cutlass::float_e2m1_t, cutlass::float_ue4m3_t,
            cutlass::float_ue4m3_t, cutlass::bfloat16_t>(
            args->int_workspace, args->int_workspace_bytes,
            args->float_workspace, args->float_workspace_bytes,
            reinterpret_cast<cutlass::float_e2m1_t*>(
                const_cast<void*>(args->input.packed_data)),
            reinterpret_cast<cutlass::float_e2m1_t*>(
                const_cast<void*>(args->weights.packed_data)),
            reinterpret_cast<cutlass::float_ue4m3_t*>(
                const_cast<void*>(args->input.block_scales)),
            reinterpret_cast<cutlass::float_ue4m3_t*>(
                const_cast<void*>(args->weights.block_scales)),
            reinterpret_cast<cutlass::bfloat16_t*>(args->output),
            const_cast<float*>(args->alpha_device),
            const_cast<int*>(args->m_indptr), static_cast<int>(args->plan.n),
            static_cast<int>(args->plan.k),
            static_cast<int>(args->plan.num_groups),
            static_cast<cudaStream_t>(args->cuda_stream), device_id);
    if (error != cudaSuccess) {
      return Internal("FlashInfer grouped NVFP4 launch failed: ",
                      cudaGetErrorString(error));
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return Internal("FlashInfer grouped NVFP4 kernel failed: ",
                      cudaGetErrorString(error));
    }
    return Ok();
  } catch (const std::exception& exception) {
    return Internal("FlashInfer grouped NVFP4 rejected launch: ",
                    exception.what());
  } catch (...) {
    return {SPARKSERVE_STATUS_INTERNAL,
            "FlashInfer grouped NVFP4 raised an unknown launch error"};
  }
}

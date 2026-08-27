/*
 * SparkServe raw-pointer adapter for FlashInfer's Apache-2.0 CUTLASS FP4 GEMM.
 * Arithmetic implementation:
 *   flashinfer-ai/flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62
 *   include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h
 *
 * This file deliberately contains no Torch, TVM-FFI, Python, or SGLang API.
 */

#include "internal/nvfp4_dense_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <limits>
#include <string>

#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

// FlashInfer normally expands this upstream macro from a Jinja build step and
// links the generated tactic objects into its extension. SparkServe instantiates
// only the tactic it has measured and exposes it through the raw C ABI.
namespace flashinfer {
namespace gemm {
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256,
                                     1, 1, 1, _1SM, false)
}  // namespace gemm
}  // namespace flashinfer

namespace {

using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM120;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::MainloopScheduleType;

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

CutlassGemmConfig DecodeConfig() {
  // This is FlashInfer's default SM120/121 tactic. Autotuned tactics remain a
  // scheduler concern and can be selected behind the same ABI later.
  return CutlassGemmConfig(
      CutlassTileConfigSM120::CtaShape128x128x128B,
      MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
      flashinfer::gemm::ClusterShape::ClusterShape_1x1x1,
      /*swap_ab=*/false,
      /*use_stream_k=*/false);
}

size_t Run(void* output, const void* input, const void* weight,
           const void* input_scales, const void* weight_scales,
           const float* alpha, int m, int n, int k, char* workspace,
           size_t workspace_bytes, cudaStream_t stream) {
  return flashinfer::gemm::genericFp4GemmKernelLauncher<
      __nv_bfloat16, cute::Int<128>, cute::Int<128>, cute::Int<256>,
      cute::Int<1>, cute::Int<1>, cute::Int<1>, flashinfer::gemm::_1SM,
      false>(output, input, weight, input_scales, weight_scales, alpha, m,
             n, k, 1, DecodeConfig(), workspace, workspace_bytes, stream,
             nullptr);
}

bool FitsInt(uint64_t value) {
  return value <= static_cast<uint64_t>(std::numeric_limits<int>::max());
}

}  // namespace

size_t sparkserve_flashinfer_nvfp4_workspace_bytes(
    const SparkServeDenseNvfp4Plan* plan) {
  if (plan == nullptr || !FitsInt(plan->m) || !FitsInt(plan->padded_n) ||
      !FitsInt(plan->padded_k)) {
    return std::numeric_limits<size_t>::max();
  }
  try {
    return Run(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
               static_cast<int>(plan->m),
               static_cast<int>(plan->padded_n),
               static_cast<int>(plan->padded_k), nullptr, 0, nullptr);
  } catch (...) {
    return std::numeric_limits<size_t>::max();
  }
}

SparkServeStatus sparkserve_flashinfer_nvfp4_launch(
    const SparkServeDenseNvfp4Args* args) {
  if (args == nullptr || args->alpha_device == nullptr) {
    return Invalid("FlashInfer NVFP4 requires a GPU-addressable alpha scalar");
  }
  if (!FitsInt(args->plan.m) || !FitsInt(args->plan.padded_n) ||
      !FitsInt(args->plan.padded_k)) {
    return Invalid("FlashInfer NVFP4 dimensions exceed the C++ runner range");
  }
  const size_t required =
      sparkserve_flashinfer_nvfp4_workspace_bytes(&args->plan);
  if (required == std::numeric_limits<size_t>::max()) {
    return {SPARKSERVE_STATUS_UNSUPPORTED,
            "FlashInfer NVFP4 rejected this shape"};
  }
  if (args->workspace_bytes < required ||
      (required != 0 && args->workspace == nullptr)) {
    return Invalid("FlashInfer NVFP4 workspace is smaller than required");
  }

  try {
    Run(args->output, args->input.packed_data, args->weight.packed_data,
        args->input.block_scales, args->weight.block_scales,
        args->alpha_device, static_cast<int>(args->plan.m),
        static_cast<int>(args->plan.padded_n),
        static_cast<int>(args->plan.padded_k),
        static_cast<char*>(args->workspace), args->workspace_bytes,
        static_cast<cudaStream_t>(args->cuda_stream));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return Internal("FlashInfer NVFP4 launch failed: ",
                      cudaGetErrorString(error));
    }
    return Ok();
  } catch (const std::exception& error) {
    return Internal("FlashInfer NVFP4 rejected launch: ", error.what());
  } catch (...) {
    return {SPARKSERVE_STATUS_INTERNAL,
            "FlashInfer NVFP4 raised an unknown launch error"};
  }
}

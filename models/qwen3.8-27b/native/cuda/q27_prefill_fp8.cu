// SPDX-License-Identifier: Apache-2.0
// Model-specific Qwen3.8-27B batched FP8 prefill projection for GB10.

#include "q27_prefill_fp8.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

struct q27_prefill_fp8_plan {
  q27_prefill_fp8_shape shape{};
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t input_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  size_t algorithm_workspace_bytes = 0;
  bool algorithm_ready = false;
};

namespace {

constexpr uint32_t kMinM = 8;
constexpr uint32_t kMaxM = 8192;
constexpr uint64_t kWorkspaceBytes = 64ULL * 1024 * 1024;
constexpr uint64_t kWorkspaceAlignment = 256;
thread_local std::string g_error;

q27_prefill_fp8_status Ok() { return {Q27_PREFILL_FP8_OK, "ok"}; }

q27_prefill_fp8_status Invalid(const char* message) {
  return {Q27_PREFILL_FP8_INVALID_ARGUMENT, message};
}

q27_prefill_fp8_status Unsupported(const char* message) {
  return {Q27_PREFILL_FP8_UNSUPPORTED_SHAPE, message};
}

q27_prefill_fp8_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_PREFILL_FP8_CUDA_ERROR, g_error.c_str()};
}

q27_prefill_fp8_status CublasError(const char* operation,
                                   cublasStatus_t status) {
  g_error.assign(operation);
  g_error.append(": ");
  const char* name = cublasGetStatusString(status);
  g_error.append(name == nullptr ? "unknown cuBLASLt error" : name);
  return {Q27_PREFILL_FP8_CUBLAS_ERROR, g_error.c_str()};
}

bool IsAligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool SupportedProjection(uint32_t n, uint32_t k) {
  return (n == 16384 && k == 5120) || (n == 10240 && k == 5120) ||
         (n == 6144 && k == 5120) || (n == 5120 && k == 6144) ||
         (n == 12288 && k == 5120) || (n == 1024 && k == 5120);
}

q27_prefill_fp8_shape ExpectedShape(uint32_t m, uint32_t n, uint32_t k) {
  q27_prefill_fp8_shape shape{};
  shape.struct_size = sizeof(shape);
  shape.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  shape.m = m;
  shape.n = n;
  shape.k = k;
  shape.input_bf16_bytes = static_cast<uint64_t>(m) * k * 2;
  shape.quantized_input_bytes = static_cast<uint64_t>(m) * k;
  shape.packed_weight_bytes = static_cast<uint64_t>(n) * k;
  shape.output_bf16_bytes = static_cast<uint64_t>(m) * n * 2;
  shape.workspace_bytes = kWorkspaceBytes;
  shape.workspace_alignment = kWorkspaceAlignment;
  return shape;
}

__global__ void QuantizeBf16ToFp8(const __nv_bfloat16* input,
                                  __nv_fp8_e4m3* output,
                                  const float* input_scale,
                                  uint64_t elements) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float inverse_scale = 1.0F / *input_scale;
  float value = __bfloat162float(input[index]) * inverse_scale;
  value = fminf(448.0F, fmaxf(-448.0F, value));
  output[index] = __nv_fp8_e4m3(value);
}

void Destroy(q27_prefill_fp8_plan* plan) {
  if (plan == nullptr) return;
  if (plan->preference != nullptr)
    cublasLtMatmulPreferenceDestroy(plan->preference);
  if (plan->output_layout != nullptr)
    cublasLtMatrixLayoutDestroy(plan->output_layout);
  if (plan->input_layout != nullptr)
    cublasLtMatrixLayoutDestroy(plan->input_layout);
  if (plan->weight_layout != nullptr)
    cublasLtMatrixLayoutDestroy(plan->weight_layout);
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->handle != nullptr) cublasLtDestroy(plan->handle);
  delete plan;
}

q27_prefill_fp8_status SetScalePointers(
    q27_prefill_fp8_plan* plan,
    const q27_prefill_fp8_project_args& args) {
  const void* weight_scale = args.weight_scale;
  cublasStatus_t status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale,
      sizeof(weight_scale));
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("set Q27 FP8 weight scale pointer", status);
  const void* input_scale = args.input_scale;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &input_scale,
      sizeof(input_scale));
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("set Q27 FP8 input scale pointer", status);
}

q27_prefill_fp8_status SelectAlgorithm(q27_prefill_fp8_plan* plan) {
  cublasLtMatmulHeuristicResult_t result{};
  int returned = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      plan->handle, plan->operation, plan->weight_layout, plan->input_layout,
      plan->output_layout, plan->output_layout, plan->preference, 1, &result,
      &returned);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("select Q27 batched FP8 algorithm", status);
  if (returned != 1 || result.state != CUBLAS_STATUS_SUCCESS)
    return {Q27_PREFILL_FP8_INTERNAL_ERROR,
            "no cuBLASLt algorithm supports this Q27 FP8 shape"};
  plan->algorithm = result.algo;
  plan->algorithm_workspace_bytes = result.workspaceSize;
  plan->algorithm_ready = true;
  return Ok();
}

}  // namespace

extern "C" q27_prefill_fp8_status q27_prefill_fp8_query(
    uint32_t m, uint32_t n, uint32_t k, q27_prefill_fp8_shape* output) {
  if (output == nullptr || output->struct_size < sizeof(*output) ||
      output->abi_version != Q27_PREFILL_FP8_ABI_VERSION)
    return Invalid("Q27 batched FP8 shape output ABI mismatch");
  if (m < kMinM || m > kMaxM)
    return Unsupported("Q27 batched FP8 M must be in [8,8192]");
  if (!SupportedProjection(n, k))
    return Unsupported("shape is not a Qwen3.8-27B FP8 projection");
  *output = ExpectedShape(m, n, k);
  return Ok();
}

extern "C" q27_prefill_fp8_status q27_prefill_fp8_plan_create(
    const q27_prefill_fp8_plan_config* config,
    q27_prefill_fp8_plan** output) {
  if (output == nullptr) return Invalid("Q27 batched FP8 plan output is null");
  *output = nullptr;
  if (config == nullptr || config->struct_size < sizeof(*config) ||
      config->abi_version != Q27_PREFILL_FP8_ABI_VERSION ||
      config->fast_accum > 1)
    return Invalid("Q27 batched FP8 plan config ABI mismatch");
  q27_prefill_fp8_shape shape{sizeof(shape),
                              Q27_PREFILL_FP8_ABI_VERSION};
  q27_prefill_fp8_status query =
      q27_prefill_fp8_query(config->m, config->n, config->k, &shape);
  if (query.code != Q27_PREFILL_FP8_OK) return query;
  if (config->workspace_bytes < shape.workspace_bytes)
    return Invalid("Q27 batched FP8 plan workspace cap is too small");

  auto* plan = new (std::nothrow) q27_prefill_fp8_plan;
  if (plan == nullptr)
    return {Q27_PREFILL_FP8_INTERNAL_ERROR,
            "cannot allocate Q27 batched FP8 plan"};
  plan->shape = shape;

  cublasStatus_t status = cublasLtCreate(&plan->handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    Destroy(plan);
    return CublasError("create Q27 cuBLASLt handle", status);
  }
  status = cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F,
                                    CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    Destroy(plan);
    return CublasError("create Q27 FP8 operation", status);
  }

  const cublasOperation_t transpose = CUBLAS_OP_T;
  const cublasOperation_t no_transpose = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
      sizeof(transpose));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose,
        sizeof(no_transpose));
  const int8_t fast_accum = static_cast<int8_t>(config->fast_accum);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum,
        sizeof(fast_accum));
  if (status != CUBLAS_STATUS_SUCCESS) {
    Destroy(plan);
    return CublasError("configure Q27 FP8 operation", status);
  }

  /* Column-major views avoid an intermediate transpose:
   * weight [N,K] row-major == [K,N] column-major, then op(A)=T;
   * input [M,K] row-major == [K,M] column-major;
   * output [M,N] row-major == [N,M] column-major. */
  status = cublasLtMatrixLayoutCreate(&plan->weight_layout, CUDA_R_8F_E4M3,
                                      shape.k, shape.n, shape.k);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&plan->input_layout, CUDA_R_8F_E4M3,
                                        shape.k, shape.m, shape.k);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&plan->output_layout, CUDA_R_16BF,
                                        shape.n, shape.m, shape.n);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulPreferenceCreate(&plan->preference);
  if (status == CUBLAS_STATUS_SUCCESS) {
    const uint64_t workspace = shape.workspace_bytes;
    status = cublasLtMatmulPreferenceSetAttribute(
        plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
        sizeof(workspace));
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    Destroy(plan);
    return CublasError("configure Q27 FP8 matrix layouts", status);
  }

  *output = plan;
  return Ok();
}

extern "C" void q27_prefill_fp8_plan_destroy(q27_prefill_fp8_plan* plan) {
  Destroy(plan);
}

extern "C" q27_prefill_fp8_status q27_prefill_fp8_project(
    q27_prefill_fp8_plan* plan,
    const q27_prefill_fp8_project_args* args) {
  if (plan == nullptr || args == nullptr ||
      args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_FP8_ABI_VERSION)
    return Invalid("Q27 batched FP8 project ABI mismatch");
  const q27_prefill_fp8_shape& shape = plan->shape;
  if (!IsAligned(args->input_bf16, 16) ||
      args->input_bf16_bytes != shape.input_bf16_bytes ||
      !IsAligned(args->input_scale, alignof(float)) ||
      !IsAligned(args->weight_fp8_e4m3, 16) ||
      args->packed_weight_bytes != shape.packed_weight_bytes ||
      !IsAligned(args->weight_scale, alignof(float)) ||
      !IsAligned(args->quantized_input_fp8_e4m3, 16) ||
      args->quantized_input_bytes != shape.quantized_input_bytes ||
      !IsAligned(args->output_bf16, 16) ||
      args->output_bf16_bytes != shape.output_bf16_bytes ||
      !IsAligned(args->workspace, shape.workspace_alignment) ||
      args->workspace_bytes < shape.workspace_bytes)
    return Invalid("Q27 batched FP8 pointer, alignment, or byte size mismatch");

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const uint64_t elements = static_cast<uint64_t>(shape.m) * shape.k;
  constexpr uint32_t threads = 256;
  QuantizeBf16ToFp8<<<static_cast<uint32_t>((elements + threads - 1) / threads),
                       threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->input_bf16),
      static_cast<__nv_fp8_e4m3*>(args->quantized_input_fp8_e4m3),
      args->input_scale, elements);
  cudaError_t cuda_status = cudaPeekAtLastError();
  if (cuda_status != cudaSuccess)
    return CudaError("quantize Q27 batched FP8 input", cuda_status);

  q27_prefill_fp8_status scale_status = SetScalePointers(plan, *args);
  if (scale_status.code != Q27_PREFILL_FP8_OK) return scale_status;
  if (!plan->algorithm_ready) {
    q27_prefill_fp8_status algorithm_status = SelectAlgorithm(plan);
    if (algorithm_status.code != Q27_PREFILL_FP8_OK) return algorithm_status;
  }
  if (args->workspace_bytes < plan->algorithm_workspace_bytes)
    return Invalid("Q27 batched FP8 selected algorithm workspace is too small");

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasStatus_t status = cublasLtMatmul(
      plan->handle, plan->operation, &alpha, args->weight_fp8_e4m3,
      plan->weight_layout, args->quantized_input_fp8_e4m3,
      plan->input_layout, &beta, args->output_bf16, plan->output_layout,
      args->output_bf16, plan->output_layout, &plan->algorithm,
      args->workspace, args->workspace_bytes, stream);
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("launch Q27 batched FP8 GEMM", status);
}

// Qwen3.8 Flash-Next QSA framing adapted from SGLang at commit
// d91c3682b0b429e4c70df63cd57f819588ce29b0. cuBLAS owns checkpoint
// projections. The fused Q/K Gemma RMSNorm, partial NeoX RoPE, gate
// deinterleave, and sigmoid-multiply follow SGLang's Apache-2.0 Triton
// programs without retaining a Python/Torch/Triton runtime dependency.

#include "sparkserve/qwen_qsa_block_api.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kQueryWidth = kQueryHeads * kHeadDim;
constexpr int kProjectedQueryWidth = 2 * kQueryWidth;
constexpr int kKvWidth = kKvHeads * kHeadDim;
constexpr int kIndexWidth = 640;
constexpr int kThreads = 256;
constexpr float kEps = 1.0e-6F;

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }
SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

const char* CublasErrorName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "not supported";
    default:
      return "unknown error";
  }
}

SparkServeStatus CublasError(const char* prefix, cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append(CublasErrorName(status));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

cublasStatus_t RowMajorBf16Linear(cublasHandle_t handle, int tokens,
                                  int output_columns, int input_columns,
                                  const void* input, const void* weight,
                                  void* output) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns, tokens, input_columns,
      &alpha, weight, CUDA_R_16BF, input_columns, input, CUDA_R_16BF,
      input_columns, &beta, output, CUDA_R_16BF, output_columns,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__device__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

// One block owns one [token, head]. It deliberately rounds normalized values
// to BF16 before RoPE, matching the SGLang Triton donor.
__global__ __launch_bounds__(kThreads) void SglangQkNormRopeGate(
    const __nv_bfloat16* projected_q, const __nv_bfloat16* projected_k,
    const __nv_bfloat16* q_weight, const __nv_bfloat16* k_weight,
    const float* cos_sin, uint64_t cos_sin_stride,
    const int64_t* positions, int rotary_dim, __nv_bfloat16* query,
    __nv_bfloat16* key, __nv_bfloat16* gate) {
  const int token = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const bool is_key = head >= kQueryHeads;
  const int local_head = is_key ? head - kQueryHeads : head;
  const __nv_bfloat16* input =
      is_key ? projected_k + (static_cast<int64_t>(token) * kKvHeads + local_head) * kHeadDim
             : projected_q +
                   (static_cast<int64_t>(token) * kQueryHeads + local_head) *
                       (2 * kHeadDim);
  const __nv_bfloat16* weight = is_key ? k_weight : q_weight;
  __nv_bfloat16* output =
      is_key ? key + (static_cast<int64_t>(token) * kKvHeads + local_head) * kHeadDim
             : query +
                   (static_cast<int64_t>(token) * kQueryHeads + local_head) * kHeadDim;

  const int dimension = static_cast<int>(threadIdx.x);
  const float value = __bfloat162float(input[dimension]);
  float sum = value * value;
  sum = WarpReduceSum(sum);
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  const int warp = dimension >> 5;
  const int lane = dimension & 31;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < 8 ? warp_sums[lane] : 0.0F;
    total = WarpReduceSum(total);
    if (lane == 0) inverse_rms = rsqrtf(total / kHeadDim + kEps);
  }
  __syncthreads();

  const float scale = 1.0F + __bfloat162float(weight[dimension]);
  const __nv_bfloat16 normalized =
      __float2bfloat16_rn(value * inverse_rms * scale);
  const int half = rotary_dim / 2;
  if (dimension < rotary_dim) {
    const int pair = dimension < half ? dimension + half : dimension - half;
    const float paired_value = __bfloat162float(input[pair]);
    const float paired_scale = 1.0F + __bfloat162float(weight[pair]);
    const float paired = __bfloat162float(
        __float2bfloat16_rn(paired_value * inverse_rms * paired_scale));
    const int rotary_column = dimension < half ? dimension : pair;
    const float* cache = cos_sin + positions[token] * cos_sin_stride;
    const float cosine = cache[rotary_column];
    const float sine = cache[half + rotary_column];
    const float current = __bfloat162float(normalized);
    const float rotated = dimension < half
                              ? current * cosine - paired * sine
                              : current * cosine + paired * sine;
    output[dimension] = __float2bfloat16_rn(rotated);
  } else {
    output[dimension] = normalized;
  }

  if (!is_key) {
    gate[(static_cast<int64_t>(token) * kQueryHeads + local_head) * kHeadDim +
         dimension] = input[kHeadDim + dimension];
  }
}

__global__ void SglangSigmoidMultiply(const __nv_bfloat16* attention,
                                      const __nv_bfloat16* gate,
                                      __nv_bfloat16* output, int elements) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float a = __bfloat162float(attention[index]);
  const float g = __bfloat162float(gate[index]);
  output[index] = __float2bfloat16_rn(a * (1.0F / (1.0F + __expf(-g))));
}

SparkServeStatus Bind(cublasHandle_t handle, cudaStream_t stream) {
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS QSA stream bind failed: ", status);
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS QSA math-mode selection failed: ", status);
  return Ok();
}

}  // namespace

extern "C" SparkServeStatus sparkserve_qwen_qsa_project_launch(
    const SparkServeQwenQsaProjectArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_QWEN_QSA_BLOCK_ABI_VERSION ||
      args->tokens == 0 || args->tokens > static_cast<uint32_t>(INT32_MAX) ||
      (args->rotary_dim != 64 && args->rotary_dim != 128) ||
      args->cos_sin_stride < args->rotary_dim) {
    return Invalid("Qwen QSA projection geometry is invalid");
  }
  const void* required[] = {
      args->hidden_states, args->q_weight, args->k_weight, args->v_weight,
      args->index_qk_weight, args->q_norm_weight, args->k_norm_weight,
      args->cos_sin_cache, args->positions, args->projected_q,
      args->projected_k, args->query, args->key, args->value, args->gate,
      args->index_qk, args->cublas_handle};
  for (const void* pointer : required) {
    if (pointer == nullptr) return Invalid("Qwen QSA projection pointer is null");
  }
  const int tokens = static_cast<int>(args->tokens);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  SparkServeStatus bound = Bind(handle, stream);
  if (bound.code != SPARKSERVE_STATUS_OK) return bound;

  struct Projection {
    int width;
    const void* weight;
    void* output;
    const char* error;
  };
  const Projection projections[] = {
      {kProjectedQueryWidth, args->q_weight, args->projected_q,
       "cuBLAS QSA Q/gate projection failed: "},
      {kKvWidth, args->k_weight, args->projected_k,
       "cuBLAS QSA K projection failed: "},
      {kKvWidth, args->v_weight, args->value,
       "cuBLAS QSA V projection failed: "},
      {kIndexWidth, args->index_qk_weight, args->index_qk,
       "cuBLAS QSA index projection failed: "},
  };
  for (const Projection& projection : projections) {
    const cublasStatus_t status = RowMajorBf16Linear(
        handle, tokens, projection.width, kHidden, args->hidden_states,
        projection.weight, projection.output);
    if (status != CUBLAS_STATUS_SUCCESS)
      return CublasError(projection.error, status);
  }

  const dim3 grid(args->tokens, kQueryHeads + kKvHeads);
  SglangQkNormRopeGate<<<grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->projected_q),
      static_cast<const __nv_bfloat16*>(args->projected_k),
      static_cast<const __nv_bfloat16*>(args->q_norm_weight),
      static_cast<const __nv_bfloat16*>(args->k_norm_weight),
      args->cos_sin_cache, args->cos_sin_stride, args->positions,
      static_cast<int>(args->rotary_dim),
      static_cast<__nv_bfloat16*>(args->query),
      static_cast<__nv_bfloat16*>(args->key),
      static_cast<__nv_bfloat16*>(args->gate));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("SGLang QSA Q/K norm-rope-gate failed: ", error);
}

extern "C" SparkServeStatus sparkserve_qwen_qsa_finish_launch(
    const SparkServeQwenQsaFinishArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_QWEN_QSA_BLOCK_ABI_VERSION ||
      args->tokens == 0 || args->tokens > static_cast<uint32_t>(INT32_MAX) ||
      args->attention_output == nullptr || args->gate == nullptr ||
      args->out_weight == nullptr || args->gated_output == nullptr ||
      args->output == nullptr || args->cublas_handle == nullptr) {
    return Invalid("Qwen QSA finish arguments are invalid");
  }
  const int tokens = static_cast<int>(args->tokens);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  SparkServeStatus bound = Bind(handle, stream);
  if (bound.code != SPARKSERVE_STATUS_OK) return bound;

  const int elements = tokens * kQueryWidth;
  SglangSigmoidMultiply<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                           stream>>>(
      static_cast<const __nv_bfloat16*>(args->attention_output),
      static_cast<const __nv_bfloat16*>(args->gate),
      static_cast<__nv_bfloat16*>(args->gated_output), elements);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return CudaError("SGLang QSA sigmoid gate failed: ", error);
  const cublasStatus_t status = RowMajorBf16Linear(
      handle, tokens, kHidden, kQueryWidth, args->gated_output,
      args->out_weight, args->output);
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("cuBLAS QSA output projection failed: ", status);
}

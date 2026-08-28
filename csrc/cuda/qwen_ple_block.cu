// Qwen3.8 Flash-Next PLE decode block, adapted from SGLang qwen4_exp.py and
// grouped_gemma_rmsnorm. This keeps only the fixed one-token GB10 serving path.

#include "sparkserve/qwen_ple_block_api.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kHyper = kHidden * kStreams;
constexpr int kConvState = 9;
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

cublasStatus_t RowMajorBf16Linear(cublasHandle_t handle, int output_columns,
                                  int input_columns, const void* input,
                                  const void* weight, void* output) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns, 1, input_columns,
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

__device__ float BlockReduceSum(float value, float* warp_sums) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  value = WarpReduceSum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  float total = (warp == 0 && lane < kThreads / 32) ? warp_sums[lane] : 0.0F;
  if (warp == 0) total = WarpReduceSum(total);
  if (warp == 0 && lane == 0) warp_sums[0] = total;
  __syncthreads();
  return warp_sums[0];
}

// One block owns one hyper stream. It preserves the donor's BF16 boundary
// after each grouped Gemma RMSNorm and after the sigmoid gate.
__global__ __launch_bounds__(kThreads) void SglangPleGateNorm(
    const __nv_bfloat16* hidden, __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* norm_key,
    const __nv_bfloat16* norm_query, const __nv_bfloat16* norm_conv,
    __nv_bfloat16* gated, __nv_bfloat16* normed) {
  const int stream = static_cast<int>(blockIdx.x);
  const int base = stream * kHidden;
  __shared__ float reductions[kThreads / 32];
  __shared__ float inverse_key;
  __shared__ float inverse_query;
  __shared__ float gate_value;
  __shared__ float inverse_conv;

  float key_square = 0.0F;
  float query_square = 0.0F;
  for (int column = static_cast<int>(threadIdx.x); column < kHidden;
       column += kThreads) {
    const float k = __bfloat162float(key[base + column]);
    const float q = __bfloat162float(hidden[base + column]);
    key_square += k * k;
    query_square += q * q;
  }
  const float key_sum = BlockReduceSum(key_square, reductions);
  const float query_sum = BlockReduceSum(query_square, reductions);
  if (threadIdx.x == 0) {
    inverse_key = rsqrtf(key_sum / kHidden + kEps);
    inverse_query = rsqrtf(query_sum / kHidden + kEps);
  }
  __syncthreads();

  float dot = 0.0F;
  for (int column = static_cast<int>(threadIdx.x); column < kHidden;
       column += kThreads) {
    const int index = base + column;
    const __nv_bfloat16 normalized_key = __float2bfloat16_rn(
        __bfloat162float(key[index]) * inverse_key *
        (1.0F + __bfloat162float(norm_key[index])));
    const __nv_bfloat16 normalized_query = __float2bfloat16_rn(
        __bfloat162float(hidden[index]) * inverse_query *
        (1.0F + __bfloat162float(norm_query[index])));
    key[index] = normalized_key;
    dot += __bfloat162float(normalized_key) *
           __bfloat162float(normalized_query);
  }
  dot = BlockReduceSum(dot, reductions);
  if (threadIdx.x == 0) {
    float signed_root = dot / sqrtf(static_cast<float>(kHidden));
    signed_root = copysignf(sqrtf(fmaxf(fabsf(signed_root), 1.0e-6F)),
                            signed_root);
    gate_value = __bfloat162float(
        __float2bfloat16_rn(1.0F / (1.0F + __expf(-signed_root))));
  }
  __syncthreads();

  float gated_square = 0.0F;
  for (int column = static_cast<int>(threadIdx.x); column < kHidden;
       column += kThreads) {
    const int index = base + column;
    const __nv_bfloat16 result = __float2bfloat16_rn(
        gate_value * __bfloat162float(value[column]));
    gated[index] = result;
    const float converted = __bfloat162float(result);
    gated_square += converted * converted;
  }
  const float gated_sum = BlockReduceSum(gated_square, reductions);
  if (threadIdx.x == 0)
    inverse_conv = rsqrtf(gated_sum / kHidden + kEps);
  __syncthreads();
  for (int column = static_cast<int>(threadIdx.x); column < kHidden;
       column += kThreads) {
    const int index = base + column;
    normed[index] = __float2bfloat16_rn(
        __bfloat162float(gated[index]) * inverse_conv *
        (1.0F + __bfloat162float(norm_conv[index])));
  }
}

__global__ void SglangPleShortConv(
    const __nv_bfloat16* normed, const __nv_bfloat16* gated,
    const __nv_bfloat16* conv_weight, __nv_bfloat16* state,
    __nv_bfloat16* output) {
  const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= kHyper) return;
  __nv_bfloat16* channel_state = state + channel * kConvState;
  const __nv_bfloat16 current = normed[channel];
  float sum = __bfloat162float(channel_state[0]) *
                  __bfloat162float(conv_weight[channel * 4]) +
              __bfloat162float(channel_state[3]) *
                  __bfloat162float(conv_weight[channel * 4 + 1]) +
              __bfloat162float(channel_state[6]) *
                  __bfloat162float(conv_weight[channel * 4 + 2]) +
              __bfloat162float(current) *
                  __bfloat162float(conv_weight[channel * 4 + 3]);
  const float rounded_conv =
      __bfloat162float(__float2bfloat16_rn(sum));
  const __nv_bfloat16 activated = __float2bfloat16_rn(
      rounded_conv / (1.0F + __expf(-rounded_conv)));
  output[channel] = __float2bfloat16_rn(
      __bfloat162float(gated[channel]) + __bfloat162float(activated));
#pragma unroll
  for (int position = 0; position < kConvState - 1; ++position)
    channel_state[position] = channel_state[position + 1];
  channel_state[kConvState - 1] = current;
}

}  // namespace

extern "C" SparkServeStatus sparkserve_qwen_ple_block_launch(
    const SparkServeQwenPleBlockArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_QWEN_PLE_BLOCK_ABI_VERSION ||
      args->tokens != 1) {
    return Invalid("Qwen PLE block supports exactly one decode token");
  }
  const void* required[] = {
      args->hidden_states, args->embedding,       args->key_weight,
      args->value_weight,  args->norm_key_weight, args->norm_query_weight,
      args->norm_conv_weight, args->conv_weight, args->conv_state,
      args->key_scratch, args->value_scratch, args->gated_scratch,
      args->normed_scratch, args->output, args->cublas_handle};
  for (const void* pointer : required) {
    if (pointer == nullptr) return Invalid("Qwen PLE block pointer is null");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS PLE stream bind failed: ", status);
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS PLE math-mode selection failed: ", status);
  status = RowMajorBf16Linear(handle, kHyper, kHidden, args->embedding,
                              args->key_weight, args->key_scratch);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS PLE key projection failed: ", status);
  status = RowMajorBf16Linear(handle, kHidden, kHidden, args->embedding,
                              args->value_weight, args->value_scratch);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS PLE value projection failed: ", status);

  SglangPleGateNorm<<<kStreams, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->hidden_states),
      static_cast<__nv_bfloat16*>(args->key_scratch),
      static_cast<const __nv_bfloat16*>(args->value_scratch),
      static_cast<const __nv_bfloat16*>(args->norm_key_weight),
      static_cast<const __nv_bfloat16*>(args->norm_query_weight),
      static_cast<const __nv_bfloat16*>(args->norm_conv_weight),
      static_cast<__nv_bfloat16*>(args->gated_scratch),
      static_cast<__nv_bfloat16*>(args->normed_scratch));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return CudaError("SGLang PLE gate/norm failed: ", error);
  SglangPleShortConv<<<(kHyper + kThreads - 1) / kThreads, kThreads, 0,
                        stream>>>(
      static_cast<const __nv_bfloat16*>(args->normed_scratch),
      static_cast<const __nv_bfloat16*>(args->gated_scratch),
      static_cast<const __nv_bfloat16*>(args->conv_weight),
      static_cast<__nv_bfloat16*>(args->conv_state),
      static_cast<__nv_bfloat16*>(args->output));
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("SGLang PLE short convolution failed: ", error);
}

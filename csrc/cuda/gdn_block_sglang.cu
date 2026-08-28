// Qwen GDN framing adapted from SGLang at commit
// d91c3682b0b429e4c70df63cd57f819588ce29b0. The decode convolution is the
// BF16,width=4 specialization of SGLang's Dao-AILab/causal-conv1d donor; the
// epilogue follows SGLang's FLA layernorm_gated Triton program. cuBLAS owns the
// checkpoint projections. The FlashInfer recurrent update is intentionally a
// separate ABI call so Rust can schedule and checkpoint both state pools.

#include "internal/gdn_block_backend.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kQkHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadDim = 128;
constexpr int kQkWidth = kQkHeads * kHeadDim;
constexpr int kValueWidth = kValueHeads * kHeadDim;
constexpr int kConvWidth = 2 * kQkWidth + kValueWidth;
constexpr int kConvKernel = 4;
constexpr int kConvState = kConvKernel - 1;
constexpr int kConvThreads = 64;
constexpr int kNormThreads = 32;
thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

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

// Specialization of SGLang's causal_conv1d_update_kernel for BF16, W=4,
// seqlen=1, no bias, SiLU enabled, and an indexed non-circular state pool.
__global__ __launch_bounds__(kConvThreads) void SglangCausalConvUpdate(
    const __nv_bfloat16* input, __nv_bfloat16* state,
    const __nv_bfloat16* weight, const int32_t* state_indices,
    __nv_bfloat16* output) {
  const int token = static_cast<int>(blockIdx.x);
  const int channel =
      static_cast<int>(blockIdx.y) * kConvThreads + threadIdx.x;
  if (channel >= kConvWidth) return;
  const int slot = state_indices[token];
  if (slot < 0) return;

  __nv_bfloat16* channel_state =
      state + (static_cast<int64_t>(slot) * kConvWidth + channel) * kConvState;
  const __nv_bfloat16 x = input[token * kConvWidth + channel];
  float x_values[kConvKernel] = {
      __bfloat162float(channel_state[0]),
      __bfloat162float(channel_state[1]),
      __bfloat162float(channel_state[2]),
      __bfloat162float(x),
  };
  channel_state[0] = channel_state[1];
  channel_state[1] = channel_state[2];
  channel_state[2] = x;

  const __nv_bfloat16* channel_weight = weight + channel * kConvKernel;
  float result = 0.0F;
#pragma unroll
  for (int index = 0; index < kConvKernel; ++index) {
    // Triton keeps the two operands in BF16 and emits mul.bf16 before the
    // FP32 accumulator add; preserving that product rounding is required for
    // byte parity with SGLang.
    const __nv_bfloat16 product = __float2bfloat16_rn(
        __bfloat162float(channel_weight[index]) * x_values[index]);
    result += __bfloat162float(product);
  }
  // Triton's tl.exp lowers to the CUDA fast exponential intrinsic.
  result = result / (1.0F + __expf(-result));
  output[token * kConvWidth + channel] = __float2bfloat16_rn(result);
}

// Short-prefill specialization for one logical sequence. The decode donor's
// token-parallel grid is unsafe when several tokens name the same recurrent
// slot: all CTAs race on the width-three state. One thread owns a channel and
// advances every token in program order, preserving the donor's BF16 product
// rounding and final SiLU exactly while projections remain one batched GEMM.
__global__ __launch_bounds__(kConvThreads) void SglangCausalConvPrefill(
    const __nv_bfloat16* input, __nv_bfloat16* state,
    const __nv_bfloat16* weight, const int32_t* state_indices,
    __nv_bfloat16* output, int tokens) {
  const int channel =
      static_cast<int>(blockIdx.x) * kConvThreads + threadIdx.x;
  if (channel >= kConvWidth) return;
  const int slot = state_indices[0];
  if (slot < 0) return;

  __nv_bfloat16* channel_state =
      state + (static_cast<int64_t>(slot) * kConvWidth + channel) * kConvState;
  __nv_bfloat16 s0 = channel_state[0];
  __nv_bfloat16 s1 = channel_state[1];
  __nv_bfloat16 s2 = channel_state[2];
  const __nv_bfloat16* channel_weight = weight + channel * kConvKernel;
  for (int token = 0; token < tokens; ++token) {
    const __nv_bfloat16 x = input[token * kConvWidth + channel];
    const __nv_bfloat16 values[kConvKernel] = {s0, s1, s2, x};
    float result = 0.0F;
#pragma unroll
    for (int index = 0; index < kConvKernel; ++index) {
      const __nv_bfloat16 product = __float2bfloat16_rn(
          __bfloat162float(channel_weight[index]) *
          __bfloat162float(values[index]));
      result += __bfloat162float(product);
    }
    result = result / (1.0F + __expf(-result));
    // The projection is token-major [T,Q|K|V], while FlashInfer's AOT ABI
    // consumes three individually contiguous [T,H,D] tensors. Pack the
    // recurrent input into Q-major, K-major, V-major planes as the state-safe
    // convolution advances. T=1 has the identical byte layout.
    int64_t output_index = 0;
    if (channel < kQkWidth) {
      output_index = static_cast<int64_t>(token) * kQkWidth + channel;
    } else if (channel < 2 * kQkWidth) {
      output_index = static_cast<int64_t>(tokens) * kQkWidth +
                     static_cast<int64_t>(token) * kQkWidth +
                     channel - kQkWidth;
    } else {
      output_index = static_cast<int64_t>(2) * tokens * kQkWidth +
                     static_cast<int64_t>(token) * kValueWidth +
                     channel - 2 * kQkWidth;
    }
    output[output_index] = __float2bfloat16_rn(result);
    s0 = s1;
    s1 = s2;
    s2 = x;
  }
  channel_state[0] = s0;
  channel_state[1] = s1;
  channel_state[2] = s2;
}

__device__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset /= 2) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

// SGLang's T=1 configuration uses one Triton program and one warp per
// [value-head,128] row. Match that mapping and keep all math in FP32 until the
// final BF16 store.
__global__ __launch_bounds__(kNormThreads) void SglangFlaGatedRmsNorm(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output, float eps) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const int begin = row * kHeadDim;
  float sum = 0.0F;
  float values[4];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const float value =
        __bfloat162float(input[begin + lane + item * kNormThreads]);
    values[item] = value;
    sum += value * value;
  }
  sum = WarpReduceSum(sum);
  __shared__ float inverse_rms;
  if (lane == 0) inverse_rms = rsqrtf(sum / kHeadDim + eps);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = lane + item * kNormThreads;
    const float z = __bfloat162float(gate[begin + column]);
    const float normalized = values[item] * inverse_rms *
                             __bfloat162float(weight[column]);
    output[begin + column] =
        __float2bfloat16_rn(normalized * (1.0F / (1.0F + expf(-z))));
  }
}

SparkServeStatus Bind(cublasHandle_t handle, cudaStream_t stream) {
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS GDN stream bind failed: ", status);
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS GDN math-mode selection failed: ", status);
  return Ok();
}

}  // namespace

SparkServeStatus sparkserve_sglang_cublas_gdn_prepare_cuda_launch(
    const SparkServeGdnBlockArgs* args) {
  const int tokens = static_cast<int>(args->plan.num_tokens);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  SparkServeStatus bound = Bind(handle, stream);
  if (bound.code != SPARKSERVE_STATUS_OK) return bound;

  struct Projection {
    int width;
    const void* weight;
    void* output;
    const char* error;
  };
  const Projection projections[] = {
      {kConvWidth, args->in_proj_qkv_weight, args->projected_qkv,
       "cuBLAS GDN QKV projection failed: "},
      {kValueWidth, args->in_proj_z_weight, args->projected_z,
       "cuBLAS GDN Z projection failed: "},
      {kValueHeads, args->in_proj_b_weight, args->projected_b,
       "cuBLAS GDN B projection failed: "},
      {kValueHeads, args->in_proj_a_weight, args->projected_a,
       "cuBLAS GDN A projection failed: "},
  };
  for (const Projection& projection : projections) {
    const cublasStatus_t status = RowMajorBf16Linear(
        handle, tokens, projection.width, kHidden, args->hidden_states,
        projection.weight, projection.output);
    if (status != CUBLAS_STATUS_SUCCESS)
      return CublasError(projection.error, status);
  }

  if (tokens == 1) {
    const dim3 grid(1, (kConvWidth + kConvThreads - 1) / kConvThreads);
    SglangCausalConvUpdate<<<grid, kConvThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args->projected_qkv),
        static_cast<__nv_bfloat16*>(args->conv_state_pool),
        static_cast<const __nv_bfloat16*>(args->conv_weight),
        args->state_indices,
        static_cast<__nv_bfloat16*>(args->convolved_qkv));
  } else {
    SglangCausalConvPrefill<<<(kConvWidth + kConvThreads - 1) / kConvThreads,
                               kConvThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args->projected_qkv),
        static_cast<__nv_bfloat16*>(args->conv_state_pool),
        static_cast<const __nv_bfloat16*>(args->conv_weight),
        args->state_indices,
        static_cast<__nv_bfloat16*>(args->convolved_qkv), tokens);
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return CudaError("SGLang causal convolution failed: ", error);
  return Ok();
}

SparkServeStatus sparkserve_sglang_cublas_gdn_finish_cuda_launch(
    const SparkServeGdnBlockArgs* args) {
  const int tokens = static_cast<int>(args->plan.num_tokens);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  SparkServeStatus bound = Bind(handle, stream);
  if (bound.code != SPARKSERVE_STATUS_OK) return bound;

  SglangFlaGatedRmsNorm<<<tokens * kValueHeads, kNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->gdn_core_output),
      static_cast<const __nv_bfloat16*>(args->projected_z),
      static_cast<const __nv_bfloat16*>(args->gated_norm_weight),
      static_cast<__nv_bfloat16*>(args->gated_norm_output),
      args->plan.rms_norm_eps);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return CudaError("SGLang FLA gated RMSNorm failed: ", error);

  const cublasStatus_t status = RowMajorBf16Linear(
      handle, tokens, kHidden, kValueWidth, args->gated_norm_output,
      args->out_proj_weight, args->attention_output);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("cuBLAS GDN output projection failed: ", status);
  return Ok();
}

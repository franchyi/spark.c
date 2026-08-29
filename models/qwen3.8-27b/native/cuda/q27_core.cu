#include "q27_kernels.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <string>

namespace {

constexpr int kThreads = 256;
thread_local std::string g_error;

q27_kernel_status Ok() { return {Q27_KERNEL_OK, "ok"}; }

q27_kernel_status Invalid(const char* message) {
  return {Q27_KERNEL_INVALID_ARGUMENT, message};
}

q27_kernel_status CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_KERNEL_CUDA_ERROR, g_error.c_str()};
}

q27_kernel_status CublasError(const char* prefix, cublasStatus_t error) {
  g_error.assign(prefix);
  g_error.append(std::to_string(static_cast<int>(error)));
  return {Q27_KERNEL_CUDA_ERROR, g_error.c_str()};
}

__global__ void GatherEmbedding(const __nv_bfloat16* weight,
                                __nv_bfloat16* output, uint32_t token,
                                uint32_t hidden_size) {
  for (uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
       column < hidden_size; column += blockDim.x * gridDim.x) {
    output[column] = weight[static_cast<uint64_t>(token) * hidden_size + column];
  }
}

__device__ float BlockSum(float value) {
  __shared__ float warps[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  if (lane == 0) warps[warp] = value;
  __syncthreads();
  value = threadIdx.x < (blockDim.x + 31) / 32 ? warps[lane] : 0.0F;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
  }
  return value;
}

__global__ void GemmaRmsNorm(const __nv_bfloat16* input,
                             const __nv_bfloat16* residual,
                             const __nv_bfloat16* weight,
                             __nv_bfloat16* output,
                             __nv_bfloat16* residual_output,
                             uint32_t hidden_size, bool has_residual,
                             float epsilon) {
  float square_sum = 0.0F;
  for (uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    float value = __bfloat162float(input[column]);
    if (has_residual) value += __bfloat162float(residual[column]);
    const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
    residual_output[column] = rounded;
    const float normalized_input = __bfloat162float(rounded);
    square_sum += normalized_input * normalized_input;
  }
  square_sum = BlockSum(square_sum);
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();
  for (uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const float value = __bfloat162float(residual_output[column]);
    const float gamma = 1.0F + __bfloat162float(weight[column]);
    output[column] = __float2bfloat16_rn(value * inverse_rms * gamma);
  }
}

__global__ void SiluMultiply(const __nv_bfloat16* gate,
                             const __nv_bfloat16* up,
                             __nv_bfloat16* output, uint32_t elements) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    const float gate_value = __bfloat162float(gate[index]);
    const float activated = gate_value / (1.0F + expf(-gate_value));
    output[index] =
        __float2bfloat16_rn(activated * __bfloat162float(up[index]));
  }
}

struct Maximum {
  float value;
  int index;
};

__device__ Maximum Better(Maximum left, Maximum right) {
  if (right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

__global__ void ArgmaxBlocks(const float* input, uint32_t elements,
                             float* values, int32_t* indices) {
  Maximum maximum = {-FLT_MAX, 0};
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    maximum = Better(maximum, {input[index], static_cast<int>(index)});
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    Maximum other = {
        __shfl_down_sync(0xFFFFFFFFU, maximum.value, offset),
        __shfl_down_sync(0xFFFFFFFFU, maximum.index, offset),
    };
    maximum = Better(maximum, other);
  }
  __shared__ Maximum warp_maxima[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_maxima[warp] = maximum;
  __syncthreads();
  if (warp == 0) {
    maximum = lane < 8 ? warp_maxima[lane] : Maximum{-FLT_MAX, 0};
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      Maximum other = {
          __shfl_down_sync(0xFFFFFFFFU, maximum.value, offset),
          __shfl_down_sync(0xFFFFFFFFU, maximum.index, offset),
      };
      maximum = Better(maximum, other);
    }
    if (lane == 0) {
      values[blockIdx.x] = maximum.value;
      indices[blockIdx.x] = maximum.index;
    }
  }
}

__global__ void ArgmaxFinal(const float* values, const int32_t* indices,
                            uint32_t elements, int32_t* output) {
  Maximum maximum = {-FLT_MAX, 0};
  for (uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
    maximum = Better(maximum, {values[index], indices[index]});
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    Maximum other = {
        __shfl_down_sync(0xFFFFFFFFU, maximum.value, offset),
        __shfl_down_sync(0xFFFFFFFFU, maximum.index, offset),
    };
    maximum = Better(maximum, other);
  }
  __shared__ Maximum warp_maxima[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_maxima[warp] = maximum;
  __syncthreads();
  if (warp == 0) {
    maximum = lane < 8 ? warp_maxima[lane] : Maximum{-FLT_MAX, 0};
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      Maximum other = {
          __shfl_down_sync(0xFFFFFFFFU, maximum.value, offset),
          __shfl_down_sync(0xFFFFFFFFU, maximum.index, offset),
      };
      maximum = Better(maximum, other);
    }
    if (lane == 0) *output = maximum.index;
  }
}

}  // namespace

extern "C" q27_kernel_status q27_embedding(const q27_embedding_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_KERNEL_ABI_VERSION ||
      args->token >= args->vocabulary || args->hidden_size != 5120 ||
      args->weight_bf16 == nullptr || args->output_bf16 == nullptr) {
    return Invalid("invalid q27 embedding arguments");
  }
  GatherEmbedding<<<20, kThreads, 0,
                    static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->weight_bf16),
      static_cast<__nv_bfloat16*>(args->output_bf16), args->token,
      args->hidden_size);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("q27 embedding: ", error);
}

extern "C" q27_kernel_status q27_gemma_rmsnorm(const q27_norm_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_KERNEL_ABI_VERSION ||
      args->hidden_size != 5120 || args->input_bf16 == nullptr ||
      args->checkpoint_weight_bf16 == nullptr || args->output_bf16 == nullptr ||
      args->residual_output_bf16 == nullptr ||
      (args->has_residual != 0 && args->residual_bf16 == nullptr)) {
    return Invalid("invalid q27 Gemma RMSNorm arguments");
  }
  GemmaRmsNorm<<<1, kThreads, 0,
                 static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input_bf16),
      static_cast<const __nv_bfloat16*>(args->residual_bf16),
      static_cast<const __nv_bfloat16*>(args->checkpoint_weight_bf16),
      static_cast<__nv_bfloat16*>(args->output_bf16),
      static_cast<__nv_bfloat16*>(args->residual_output_bf16),
      args->hidden_size, args->has_residual != 0, args->epsilon);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("q27 RMSNorm: ", error);
}

extern "C" q27_kernel_status q27_silu_mul(const q27_silu_mul_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_KERNEL_ABI_VERSION || args->elements != 17408 ||
      args->gate_bf16 == nullptr || args->up_bf16 == nullptr ||
      args->output_bf16 == nullptr) {
    return Invalid("invalid q27 SiLU multiply arguments");
  }
  SiluMultiply<<<68, kThreads, 0,
                 static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->gate_bf16),
      static_cast<const __nv_bfloat16*>(args->up_bf16),
      static_cast<__nv_bfloat16*>(args->output_bf16), args->elements);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("q27 SiLU multiply: ", error);
}

extern "C" q27_kernel_status q27_lm_head(const q27_lm_head_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_KERNEL_ABI_VERSION ||
      args->vocabulary != 248320 || args->hidden_size != 5120 ||
      args->hidden_bf16 == nullptr || args->weight_bf16 == nullptr ||
      args->logits_f32 == nullptr || args->cublas_handle == nullptr) {
    return Invalid("invalid q27 LM-head arguments");
  }
  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cublasStatus_t status =
      cublasSetStream(handle, static_cast<cudaStream_t>(args->cuda_stream));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("q27 LM-head stream: ", status);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, args->vocabulary, 1,
      args->hidden_size, &alpha, args->weight_bf16, CUDA_R_16BF,
      args->hidden_size, args->hidden_bf16, CUDA_R_16BF, args->hidden_size,
      &beta, args->logits_f32, CUDA_R_32F, args->vocabulary,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("q27 LM-head projection: ", status);
}

extern "C" q27_kernel_status q27_argmax(const q27_argmax_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_KERNEL_ABI_VERSION || args->elements == 0 ||
      args->logits_f32 == nullptr || args->scratch_values_f32 == nullptr ||
      args->scratch_indices_i32 == nullptr || args->output_token_i32 == nullptr) {
    return Invalid("invalid q27 argmax arguments");
  }
  const uint32_t blocks = (args->elements + kThreads - 1) / kThreads;
  if (args->scratch_elements < blocks || blocks > kThreads * 4) {
    return Invalid("q27 argmax scratch is too small");
  }
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  ArgmaxBlocks<<<blocks, kThreads, 0, stream>>>(
      args->logits_f32, args->elements, args->scratch_values_f32,
      args->scratch_indices_i32);
  ArgmaxFinal<<<1, kThreads, 0, stream>>>(
      args->scratch_values_f32, args->scratch_indices_i32, blocks,
      args->output_token_i32);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("q27 argmax: ", error);
}

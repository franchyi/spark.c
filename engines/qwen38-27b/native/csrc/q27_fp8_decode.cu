/*
 * Lightweight Qwen3.8-27B single-token FP8 projection for GB10.
 *
 * The streaming GEMV arithmetic is adapted from SGLang's Apache-2.0
 * sm120_fp8_gemv.cuh at sha256 b30efef9bc1a000b46e0710357f4531a...
 * Framework TensorView/JIT wrappers are removed. SparkServe instantiates only
 * the five shapes present in the q27 checkpoint and calls raw CUDA pointers.
 */

#include "q27_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kFp8VectorBytes = 16;
thread_local std::string g_error;

q27_kernel_status Ok() { return {Q27_KERNEL_OK, "ok"}; }

q27_kernel_status Invalid(const char* message) {
  return {Q27_KERNEL_INVALID_ARGUMENT, message};
}

q27_kernel_status CudaError(cudaError_t error) {
  g_error.assign("q27 FP8 projection: ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_KERNEL_CUDA_ERROR, g_error.c_str()};
}

__global__ void QuantizeBf16ToFp8(const __nv_bfloat16* input,
                                  uint8_t* output, const float* scale,
                                  uint32_t elements) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float inverse_scale = 1.0F / *scale;
  float value = __bfloat162float(input[index]) * inverse_scale;
  value = fminf(448.0F, fmaxf(-448.0F, value));
  output[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float Dot16Fp8(float4 weight, float4 input) {
  const auto* weight_pairs =
      reinterpret_cast<const __nv_fp8x2_e4m3*>(&weight);
  const auto* input_pairs =
      reinterpret_cast<const __nv_fp8x2_e4m3*>(&input);
  float accumulator = 0.0F;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    const float2 weight_values = static_cast<float2>(weight_pairs[index]);
    const float2 input_values = static_cast<float2>(input_pairs[index]);
    accumulator = fmaf(weight_values.x, input_values.x, accumulator);
    accumulator = fmaf(weight_values.y, input_values.y, accumulator);
  }
  return accumulator;
}

template <uint32_t N, uint32_t K, uint32_t RowsPerWarp,
          uint32_t NumWarps>
__global__ void __launch_bounds__(NumWarps * 32) Fp8Gemv(
    __nv_bfloat16* output, const uint8_t* input, const uint8_t* weight,
    const float* input_scale, const float* weight_scale) {
  __shared__ uint8_t staged_input[K];
  const uint32_t thread = threadIdx.x;
  for (uint32_t offset = thread * kFp8VectorBytes; offset < K;
       offset += NumWarps * 32 * kFp8VectorBytes) {
    *reinterpret_cast<float4*>(staged_input + offset) =
        *reinterpret_cast<const float4*>(input + offset);
  }
  __syncthreads();

  const uint32_t warp = thread / 32;
  const uint32_t lane = thread % 32;
  const uint32_t first_row =
      (blockIdx.x * NumWarps + warp) * RowsPerWarp;
  if (first_row >= N) return;

  float accumulator[RowsPerWarp];
#pragma unroll
  for (uint32_t row = 0; row < RowsPerWarp; ++row) accumulator[row] = 0.0F;

  constexpr uint32_t kStep = 32 * kFp8VectorBytes;
  for (uint32_t column = lane * kFp8VectorBytes; column < K;
       column += kStep) {
    const float4 input_values =
        *reinterpret_cast<const float4*>(staged_input + column);
#pragma unroll
    for (uint32_t row = 0; row < RowsPerWarp; ++row) {
      if (first_row + row < N) {
        const uint8_t* weight_row =
            weight + static_cast<size_t>(first_row + row) * K + column;
        const float4 weight_values =
            __ldcs(reinterpret_cast<const float4*>(weight_row));
        accumulator[row] += Dot16Fp8(weight_values, input_values);
      }
    }
  }

#pragma unroll
  for (uint32_t row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      accumulator[row] +=
          __shfl_down_sync(0xFFFFFFFFU, accumulator[row], offset);
    }
  }
  if (lane == 0) {
    const float alpha = *input_scale * *weight_scale;
#pragma unroll
    for (uint32_t row = 0; row < RowsPerWarp; ++row) {
      if (first_row + row < N) {
        output[first_row + row] =
            __float2bfloat16_rn(accumulator[row] * alpha);
      }
    }
  }
}

template <uint32_t N, uint32_t K>
void Launch(const q27_fp8_project_args& args, cudaStream_t stream) {
  constexpr uint32_t kWarps = 8;
  constexpr uint32_t kRows = N >= 8192 ? 2 : 1;
  constexpr uint32_t kRowsPerBlock = kRows * kWarps;
  constexpr uint32_t kBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
  QuantizeBf16ToFp8<<<(K + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args.input_bf16),
      static_cast<uint8_t*>(args.quantized_input_fp8_e4m3), args.input_scale,
      K);
  Fp8Gemv<N, K, kRows, kWarps><<<kBlocks, kWarps * 32, 0, stream>>>(
      static_cast<__nv_bfloat16*>(args.output_bf16),
      static_cast<const uint8_t*>(args.quantized_input_fp8_e4m3),
      static_cast<const uint8_t*>(args.weight_fp8_e4m3), args.input_scale,
      args.weight_scale);
}

bool Supported(uint32_t n, uint32_t k) {
  return (n == 10240 && k == 5120) || (n == 6144 && k == 5120) ||
         (n == 5120 && k == 6144) || (n == 12288 && k == 5120) ||
         (n == 1024 && k == 5120);
}

}  // namespace

extern "C" q27_kernel_status q27_fp8_project(
    const q27_fp8_project_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_KERNEL_ABI_VERSION ||
      args->input_bf16 == nullptr || args->weight_fp8_e4m3 == nullptr ||
      args->input_scale == nullptr || args->weight_scale == nullptr ||
      args->quantized_input_fp8_e4m3 == nullptr ||
      args->output_bf16 == nullptr) {
    return Invalid("invalid q27 FP8 projection arguments");
  }
  if (!Supported(args->n, args->k)) {
    return {Q27_KERNEL_UNSUPPORTED_SHAPE,
            "shape is not a Qwen3.8-27B FP8 projection"};
  }
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  if (args->n == 10240 && args->k == 5120) {
    Launch<10240, 5120>(*args, stream);
  } else if (args->n == 6144 && args->k == 5120) {
    Launch<6144, 5120>(*args, stream);
  } else if (args->n == 5120 && args->k == 6144) {
    Launch<5120, 6144>(*args, stream);
  } else if (args->n == 12288 && args->k == 5120) {
    Launch<12288, 5120>(*args, stream);
  } else {
    Launch<1024, 5120>(*args, stream);
  }
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError(error);
}

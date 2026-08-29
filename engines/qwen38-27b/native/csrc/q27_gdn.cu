/*
 * Qwen3.8-27B decode-only GDN framing.
 *
 * The causal convolution and gated RMSNorm preserve the arithmetic of
 * SGLang d91c3682b0b429e4c70df63cd57f819588ce29b0. The recurrent update is
 * the separately pinned FlashInfer 906181e3f4cf4bcc81835fb480db4011bbd80b62
 * SM121 AOT object. This file retains no framework dispatcher or allocator.
 */

#include "q27_gdn.h"
#include "q27_gdn_flashinfer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kConvThreads = 64;
constexpr int kNormThreads = 32;
thread_local std::string g_error;

q27_gdn_status Ok() { return {Q27_GDN_OK, "ok"}; }

q27_gdn_status Invalid(const char* message) {
  return {Q27_GDN_INVALID_ARGUMENT, message};
}

q27_gdn_status CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_CUDA_ERROR, g_error.c_str()};
}

__global__ void ConvertParameters(const __nv_bfloat16* a_log,
                                  const __nv_bfloat16* dt_bias,
                                  float* a_log_f32, float* dt_bias_f32) {
  const int index = static_cast<int>(threadIdx.x);
  if (index >= Q27_GDN_VALUE_HEADS) return;
  a_log_f32[index] = __bfloat162float(a_log[index]);
  dt_bias_f32[index] = __bfloat162float(dt_bias[index]);
}

// BF16,width=4,T=1 specialization of SGLang causal_conv1d_update_kernel.
// The donor multiplies in BF16 and accumulates the rounded products in FP32.
__global__ __launch_bounds__(kConvThreads) void CausalConvUpdate(
    const __nv_bfloat16* input, __nv_bfloat16* state,
    const __nv_bfloat16* weight, const int32_t* state_indices,
    uint32_t state_slots, __nv_bfloat16* output) {
  const int channel = static_cast<int>(blockIdx.x) * kConvThreads + threadIdx.x;
  if (channel >= Q27_GDN_CONV_WIDTH) return;
  const int slot = state_indices[0];
  if (slot < 0 || static_cast<uint32_t>(slot) >= state_slots) {
    output[channel] = __float2bfloat16_rn(0.0F);
    return;
  }

  __nv_bfloat16* history =
      state + (static_cast<int64_t>(slot) * Q27_GDN_CONV_WIDTH + channel) *
                  Q27_GDN_CONV_HISTORY;
  const __nv_bfloat16 x = input[channel];
  const __nv_bfloat16 values[Q27_GDN_CONV_KERNEL] = {
      history[0], history[1], history[2], x};
  history[0] = history[1];
  history[1] = history[2];
  history[2] = x;

  const __nv_bfloat16* channel_weight =
      weight + channel * Q27_GDN_CONV_KERNEL;
  float result = 0.0F;
#pragma unroll
  for (int index = 0; index < Q27_GDN_CONV_KERNEL; ++index) {
    const __nv_bfloat16 product = __float2bfloat16_rn(
        __bfloat162float(channel_weight[index]) *
        __bfloat162float(values[index]));
    result += __bfloat162float(product);
  }
  result = result / (1.0F + __expf(-result));
  output[channel] = __float2bfloat16_rn(result);
}

__device__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset /= 2)
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  return value;
}

// One warp owns each [value-head,128] row, matching SGLang FLA's T=1 path.
__global__ __launch_bounds__(kNormThreads) void GatedRmsNorm(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const int begin = row * Q27_GDN_HEAD_DIM;
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
  if (lane == 0)
    inverse_rms = rsqrtf(sum / Q27_GDN_HEAD_DIM + 1.0e-6F);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = lane + item * kNormThreads;
    const float z = __bfloat162float(gate[begin + column]);
    const float normalized = values[item] * inverse_rms *
                             __bfloat162float(weight[column]);
    output[begin + column] = __float2bfloat16_rn(
        normalized * (1.0F / (1.0F + expf(-z))));
  }
}

bool ValidDecodePointers(const q27_gdn_decode_args& args) {
  return args.projected_qkv_bf16 != nullptr &&
         args.projected_z_bf16 != nullptr &&
         args.projected_a_bf16 != nullptr &&
         args.projected_b_bf16 != nullptr && args.conv_weight_bf16 != nullptr &&
         args.norm_weight_bf16 != nullptr && args.a_log_f32 != nullptr &&
         args.dt_bias_f32 != nullptr && args.convolution_state_bf16 != nullptr &&
         args.recurrent_state_bf16 != nullptr &&
         args.state_indices_i32 != nullptr && args.convolved_qkv_bf16 != nullptr &&
         args.recurrent_output_bf16 != nullptr &&
         args.normalized_output_bf16 != nullptr;
}

}  // namespace

extern "C" q27_gdn_status q27_gdn_query_layout(q27_gdn_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_ABI_VERSION)
    return Invalid("invalid q27 GDN layout query");
  output->convolution_state_bytes_per_slot =
      static_cast<uint64_t>(Q27_GDN_CONV_WIDTH) * Q27_GDN_CONV_HISTORY * 2;
  output->recurrent_state_bytes_per_slot =
      static_cast<uint64_t>(Q27_GDN_VALUE_HEADS) * Q27_GDN_HEAD_DIM *
      Q27_GDN_HEAD_DIM * 2;
  output->projected_qkv_bytes =
      static_cast<uint64_t>(Q27_GDN_CONV_WIDTH) * 2;
  output->recurrent_output_bytes =
      static_cast<uint64_t>(Q27_GDN_VALUE_WIDTH) * 2;
  return Ok();
}

extern "C" q27_gdn_status q27_gdn_convert_parameters(
    const q27_gdn_convert_parameters_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_ABI_VERSION ||
      args->a_log_bf16 == nullptr || args->dt_bias_bf16 == nullptr ||
      args->a_log_f32 == nullptr || args->dt_bias_f32 == nullptr)
    return Invalid("invalid q27 GDN parameter conversion arguments");
  ConvertParameters<<<1, 64, 0,
                      static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->a_log_bf16),
      static_cast<const __nv_bfloat16*>(args->dt_bias_bf16), args->a_log_f32,
      args->dt_bias_f32);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("q27 GDN parameter conversion: ", error);
}

extern "C" q27_gdn_status q27_gdn_decode(
    const q27_gdn_decode_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_ABI_VERSION || args->state_slots == 0 ||
      !ValidDecodePointers(*args))
    return Invalid("invalid q27 GDN decode arguments");

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  CausalConvUpdate<<<
      (Q27_GDN_CONV_WIDTH + kConvThreads - 1) / kConvThreads, kConvThreads, 0,
      stream>>>(
      static_cast<const __nv_bfloat16*>(args->projected_qkv_bf16),
      static_cast<__nv_bfloat16*>(args->convolution_state_bf16),
      static_cast<const __nv_bfloat16*>(args->conv_weight_bf16),
      args->state_indices_i32, args->state_slots,
      static_cast<__nv_bfloat16*>(args->convolved_qkv_bf16));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return CudaError("q27 GDN causal convolution: ", error);

  q27_gdn_status recurrent = q27_gdn_flashinfer_decode(args);
  if (recurrent.code != Q27_GDN_OK) return recurrent;

  GatedRmsNorm<<<Q27_GDN_VALUE_HEADS, kNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->recurrent_output_bf16),
      static_cast<const __nv_bfloat16*>(args->projected_z_bf16),
      static_cast<const __nv_bfloat16*>(args->norm_weight_bf16),
      static_cast<__nv_bfloat16*>(args->normalized_output_bf16));
  error = cudaGetLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("q27 GDN gated RMSNorm: ", error);
}

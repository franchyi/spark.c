#include "flash/qwen_decode_glue_api.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kHyper = kHidden * kStreams;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kThreads = 256;
constexpr int kHc = 4;
constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kIntermediate = 640;
constexpr int kGateUp = 2 * kIntermediate;
constexpr int kWarpsPerBlock = kThreads / 32;

thread_local std::string g_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }
FlashStatus Invalid(const char* message) {
  return {FLASH_STATUS_INVALID_ARGUMENT, message};
}

FlashStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

FlashStatus CublasError(const char* prefix, cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append(std::to_string(static_cast<int>(status)));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

cublasStatus_t RowMajorBf16Linear(cublasHandle_t handle, int tokens,
                                  int output_columns, int input_columns,
                                  const void* input, const void* weight,
                                  void* output) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns, tokens,
      input_columns, &alpha, weight, CUDA_R_16BF, input_columns, input,
      CUDA_R_16BF, input_columns, &beta, output, CUDA_R_16BF, output_columns,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__global__ void RepeatEmbedding(const __nv_bfloat16* input,
                                __nv_bfloat16* output) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < kHyper) output[index] = input[index % kHidden];
}

__global__ void AddHyper(const __nv_bfloat16* delta,
                         __nv_bfloat16* hidden) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < kHyper) {
    hidden[index] = __float2bfloat16_rn(__bfloat162float(hidden[index]) +
                                        __bfloat162float(delta[index]));
  }
}

// At sequence length one, every query head has one legal key. Softmax is one,
// so QSA attention is exactly the corresponding grouped KV value.
__global__ void ExpandSingleValue(const __nv_bfloat16* value,
                                  __nv_bfloat16* output) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kQueryHeads * kHeadDim) return;
  const int query_head = index / kHeadDim;
  const int column = index % kHeadDim;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  output[index] = value[kv_head * kHeadDim + column];
}

struct MaxPair {
  float value;
  uint32_t index;
};

__device__ MaxPair Better(MaxPair left, MaxPair right) {
  if (right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

// The vocabulary is only about one MiB of FP32 data.  A persistent block with
// coalesced striding is faster than migrating that region to the CPU and also
// makes the decode tail capturable by a CUDA graph.
__global__ __launch_bounds__(kThreads) void GreedyArgmax(
    const float* values, uint32_t elements, uint32_t row_stride,
    uint32_t* output_index) {
  const uint32_t row = static_cast<uint32_t>(blockIdx.x);
  values += static_cast<uint64_t>(row) * row_stride;
  MaxPair best = {-CUDART_INF_F, 0U};
  for (uint32_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    const float value = values[index];
    if (value > best.value ||
        (value == best.value && index < best.index)) {
      best = {value, index};
    }
  }

  for (int offset = 16; offset != 0; offset >>= 1) {
    MaxPair other = {__shfl_down_sync(0xFFFFFFFFU, best.value, offset),
                     __shfl_down_sync(0xFFFFFFFFU, best.index, offset)};
    best = Better(best, other);
  }
  __shared__ MaxPair warp_best[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_best[warp] = best;
  __syncthreads();
  if (warp == 0) {
    best = lane < 8 ? warp_best[lane] : MaxPair{-CUDART_INF_F, UINT32_MAX};
    for (int offset = 16; offset != 0; offset >>= 1) {
      MaxPair other = {__shfl_down_sync(0xFFFFFFFFU, best.value, offset),
                       __shfl_down_sync(0xFFFFFFFFU, best.index, offset)};
      best = Better(best, other);
    }
    if (lane == 0) output_index[row] = best.index;
  }
}

__device__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

// Gemma RMSNorm uses (1 + weight), independently for every HC branch.
__global__ __launch_bounds__(kThreads) void MtpRmsNorm(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  const int begin = row * kHidden;
  float sum = 0.0F;
  for (int column = static_cast<int>(threadIdx.x); column < kHidden;
       column += kThreads) {
    const float value = __bfloat162float(input[begin + column]);
    sum += value * value;
  }
  sum = WarpSum(sum);
  __shared__ float warp_sums[kWarpsPerBlock];
  __shared__ float inverse_rms;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
    total = WarpSum(total);
    if (lane == 0) inverse_rms = rsqrtf(total / kHidden + 1.0e-6F);
  }
  __syncthreads();
  for (int column = static_cast<int>(threadIdx.x); column < kHidden;
       column += kThreads) {
    const int index = begin + column;
    const float value = __bfloat162float(input[index]);
    const float scale = 1.0F + __bfloat162float(weight[index]);
    output[index] = __float2bfloat16_rn(value * inverse_rms * scale);
  }
}

__global__ void AddMtpEmbedding(const __nv_bfloat16* embedding,
                                __nv_bfloat16* hidden) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kHc * kHidden) return;
  hidden[index] = __float2bfloat16_rn(
      __bfloat162float(hidden[index]) +
      __bfloat162float(embedding[index % kHidden]));
}

__global__ __launch_bounds__(kThreads) void MtpGateUpGemv(
    const __nv_bfloat16* input, const int32_t* expert_ids,
    const __nv_bfloat16* weights, __nv_bfloat16* output) {
  const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int item = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (item >= kTopK * kGateUp) return;
  const int rank = item / kGateUp;
  const int column = item % kGateUp;
  const int expert = expert_ids[rank];
  const __nv_bfloat16* row =
      weights + (static_cast<int64_t>(expert) * kGateUp + column) * kHidden;
  float sum = 0.0F;
  for (int input_column = lane; input_column < kHidden; input_column += 32) {
    sum += __bfloat162float(input[input_column]) *
           __bfloat162float(row[input_column]);
  }
  sum = WarpSum(sum);
  if (lane == 0) output[item] = __float2bfloat16_rn(sum);
}

__global__ void MtpSiluMultiply(const __nv_bfloat16* gate_up,
                                __nv_bfloat16* activated) {
  const int item = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= kTopK * kIntermediate) return;
  const int rank = item / kIntermediate;
  const int column = item % kIntermediate;
  const float gate =
      __bfloat162float(gate_up[rank * kGateUp + column]);
  const float up = __bfloat162float(
      gate_up[rank * kGateUp + kIntermediate + column]);
  activated[item] =
      __float2bfloat16_rn((gate / (1.0F + __expf(-gate))) * up);
}

__global__ __launch_bounds__(kThreads) void MtpDownGemv(
    const __nv_bfloat16* activated, const int32_t* expert_ids,
    const __nv_bfloat16* weights, __nv_bfloat16* output) {
  const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int item = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (item >= kTopK * kHidden) return;
  const int rank = item / kHidden;
  const int column = item % kHidden;
  const int expert = expert_ids[rank];
  const __nv_bfloat16* row =
      weights + (static_cast<int64_t>(expert) * kHidden + column) * kIntermediate;
  const __nv_bfloat16* input = activated + rank * kIntermediate;
  float sum = 0.0F;
  for (int input_column = lane; input_column < kIntermediate;
       input_column += 32) {
    sum += __bfloat162float(input[input_column]) *
           __bfloat162float(row[input_column]);
  }
  sum = WarpSum(sum);
  if (lane == 0) output[item] = __float2bfloat16_rn(sum);
}

__global__ void MtpWeightedSum(const __nv_bfloat16* expert_output,
                               const float* expert_weights,
                               __nv_bfloat16* output) {
  const int column = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= kHidden) return;
  float sum = 0.0F;
#pragma unroll
  for (int rank = 0; rank < kTopK; ++rank) {
    sum += expert_weights[rank] *
           __bfloat162float(expert_output[rank * kHidden + column]);
  }
  output[column] = __float2bfloat16_rn(sum);
}

FlashStatus ValidateGlue(const FlashQwenDecodeGlueArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_DECODE_GLUE_ABI_VERSION ||
      args->input == nullptr || args->output == nullptr) {
    return Invalid("Qwen decode glue arguments are invalid");
  }
  return Ok();
}

}  // namespace

extern "C" FlashStatus flash_qwen_repeat_embedding_launch(
    const FlashQwenDecodeGlueArgs* args) {
  FlashStatus valid = ValidateGlue(args);
  if (valid.code != FLASH_STATUS_OK) return valid;
  RepeatEmbedding<<<(kHyper + kThreads - 1) / kThreads, kThreads, 0,
                     static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input),
      static_cast<__nv_bfloat16*>(args->output));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen embedding repeat failed: ", error);
}

extern "C" FlashStatus flash_qwen_add_hyper_launch(
    const FlashQwenDecodeGlueArgs* args) {
  FlashStatus valid = ValidateGlue(args);
  if (valid.code != FLASH_STATUS_OK) return valid;
  AddHyper<<<(kHyper + kThreads - 1) / kThreads, kThreads, 0,
             static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input),
      static_cast<__nv_bfloat16*>(args->output));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("Qwen PLE add failed: ", error);
}

extern "C" FlashStatus flash_qwen_qsa_single_value_launch(
    const FlashQwenDecodeGlueArgs* args) {
  FlashStatus valid = ValidateGlue(args);
  if (valid.code != FLASH_STATUS_OK) return valid;
  constexpr int kElements = kQueryHeads * kHeadDim;
  ExpandSingleValue<<<(kElements + kThreads - 1) / kThreads, kThreads, 0,
                      static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input),
      static_cast<__nv_bfloat16*>(args->output));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen single-token QSA failed: ", error);
}

extern "C" FlashStatus flash_qwen_lm_head_launch(
    const FlashQwenLmHeadArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_DECODE_GLUE_ABI_VERSION ||
      args->tokens == 0 || args->vocabulary == 0 || args->hidden_size == 0 ||
      args->hidden_states == nullptr || args->weight == nullptr ||
      args->logits == nullptr || args->cublas_handle == nullptr) {
    return Invalid("Qwen LM-head arguments are invalid");
  }
  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cublasStatus_t status =
      cublasSetStream(handle, static_cast<cudaStream_t>(args->cuda_stream));
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen LM-head stream bind failed: ", status);
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen LM-head math-mode failed: ", status);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(args->vocabulary),
      static_cast<int>(args->tokens),
      static_cast<int>(args->hidden_size), &alpha, args->weight, CUDA_R_16BF,
      static_cast<int>(args->hidden_size), args->hidden_states, CUDA_R_16BF,
      static_cast<int>(args->hidden_size), &beta, args->logits, CUDA_R_32F,
      static_cast<int>(args->vocabulary), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("Qwen LM-head projection failed: ", status);
}

extern "C" FlashStatus flash_qwen_argmax_launch(
    const FlashQwenArgmaxArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_DECODE_GLUE_ABI_VERSION ||
      args->rows == 0 || args->elements == 0 ||
      args->row_stride < args->elements || args->values == nullptr ||
      args->output_index == nullptr) {
    return Invalid("Qwen argmax arguments are invalid");
  }
  GreedyArgmax<<<args->rows, kThreads, 0,
                 static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->values, args->elements, args->row_stride, args->output_index);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("Qwen argmax failed: ", error);
}

extern "C" FlashStatus flash_qwen_mtp_input_launch(
    const FlashQwenMtpInputArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_DECODE_GLUE_ABI_VERSION ||
      args->embedding == nullptr || args->target_hidden == nullptr ||
      args->embedding_norm_weight == nullptr ||
      args->hidden_norm_weight == nullptr ||
      args->embedding_fc_weight == nullptr || args->hidden_fc_weight == nullptr ||
      args->embedding_norm_scratch == nullptr ||
      args->embedding_projected_scratch == nullptr ||
      args->hidden_norm_scratch == nullptr || args->output == nullptr ||
      args->cublas_handle == nullptr) {
    return Invalid("Qwen MTP input arguments are invalid");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen MTP stream bind failed: ", status);
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen MTP math-mode failed: ", status);

  MtpRmsNorm<<<1, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->embedding),
      static_cast<const __nv_bfloat16*>(args->embedding_norm_weight),
      static_cast<__nv_bfloat16*>(args->embedding_norm_scratch), 1);
  MtpRmsNorm<<<kHc, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->target_hidden),
      static_cast<const __nv_bfloat16*>(args->hidden_norm_weight),
      static_cast<__nv_bfloat16*>(args->hidden_norm_scratch), kHc);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return CudaError("Qwen MTP RMSNorm failed: ", error);

  status = RowMajorBf16Linear(
      handle, 1, kHidden, kHidden, args->embedding_norm_scratch,
      args->embedding_fc_weight, args->embedding_projected_scratch);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen MTP embedding projection failed: ", status);
  status = RowMajorBf16Linear(
      handle, kHc, kHidden, kHidden, args->hidden_norm_scratch,
      args->hidden_fc_weight, args->output);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("Qwen MTP hidden projection failed: ", status);
  AddMtpEmbedding<<<(kHc * kHidden + kThreads - 1) / kThreads, kThreads,
                     0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->embedding_projected_scratch),
      static_cast<__nv_bfloat16*>(args->output));
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen MTP input fusion failed: ", error);
}

extern "C" FlashStatus flash_qwen_mtp_experts_launch(
    const FlashQwenMtpExpertsArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_DECODE_GLUE_ABI_VERSION ||
      args->hidden_states == nullptr || args->expert_ids == nullptr ||
      args->expert_weights == nullptr || args->gate_up_weight == nullptr ||
      args->down_weight == nullptr || args->gate_up_scratch == nullptr ||
      args->activated_scratch == nullptr ||
      args->expert_output_scratch == nullptr || args->output == nullptr) {
    return Invalid("Qwen MTP expert arguments are invalid");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const int gate_up_items = kTopK * kGateUp;
  MtpGateUpGemv<<<(gate_up_items + kWarpsPerBlock - 1) / kWarpsPerBlock,
                    kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->hidden_states), args->expert_ids,
      static_cast<const __nv_bfloat16*>(args->gate_up_weight),
      static_cast<__nv_bfloat16*>(args->gate_up_scratch));
  MtpSiluMultiply<<<(kTopK * kIntermediate + kThreads - 1) / kThreads,
                      kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->gate_up_scratch),
      static_cast<__nv_bfloat16*>(args->activated_scratch));
  const int down_items = kTopK * kHidden;
  MtpDownGemv<<<(down_items + kWarpsPerBlock - 1) / kWarpsPerBlock,
                 kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->activated_scratch),
      args->expert_ids, static_cast<const __nv_bfloat16*>(args->down_weight),
      static_cast<__nv_bfloat16*>(args->expert_output_scratch));
  MtpWeightedSum<<<(kHidden + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->expert_output_scratch),
      args->expert_weights, static_cast<__nv_bfloat16*>(args->output));
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen MTP BF16 experts failed: ", error);
}

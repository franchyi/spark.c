// Narrow CUDA extraction of SGLang's GLM5Next KPool compression kernel at
// commit 9a26e7490f8db83a7fde29ae38f3bbff50ba035c. The per-dimension softmax,
// BF16 boundaries, normalized Hadamard-128 transform, and FP8 E4M3 scale
// convention are preserved without Torch, Triton, pools, or allocation.

#include "sparkserve/glm_dsa_api.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr uint32_t kPoolSize = 4;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kPageSize = 64;
constexpr uint64_t kMinKeyPageBytes = kPageSize * kHeadDim;
constexpr uint64_t kMinScalePageBytes = kPageSize * sizeof(float);
constexpr float kHadamardScale = 0.08838834764831845f;
constexpr uint32_t kIndexHeads = 32;
constexpr float kFp8Max = 448.0f;
thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

__device__ __forceinline__ float LoadBf16(const uint16_t* pointer) {
  __nv_bfloat16_raw raw;
  raw.x = *pointer;
  return __bfloat162float(__nv_bfloat16(raw));
}

__device__ __forceinline__ float RoundBf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint16_t ToBf16(float value) {
  const __nv_bfloat16_raw raw = __float2bfloat16_rn(value);
  return raw.x;
}

__device__ __forceinline__ uint8_t ToFp8(float value) {
  const __nv_fp8_e4m3 converted(value);
  return *reinterpret_cast<const uint8_t*>(&converted);
}

__device__ __forceinline__ float HadamardQuantize(
    float weighted, float* shared, uint32_t dimension, bool round_scale,
    float* scale_output) {
  shared[dimension] = RoundBf16(weighted);
  __syncthreads();
#pragma unroll
  for (uint32_t stride = 1; stride < kHeadDim; stride *= 2) {
    const float mine = shared[dimension];
    const float other = shared[dimension ^ stride];
    __syncthreads();
    shared[dimension] =
        (dimension & stride) == 0 ? mine + other : other - mine;
    __syncthreads();
  }
  const float transformed = RoundBf16(shared[dimension] * kHadamardScale);
  shared[dimension] = fabsf(transformed);
  __syncthreads();
#pragma unroll
  for (uint32_t stride = kHeadDim / 2; stride > 0; stride /= 2) {
    if (dimension < stride) {
      shared[dimension] =
          fmaxf(shared[dimension], shared[dimension + stride]);
    }
    __syncthreads();
  }
  float scale = fmaxf(shared[0], 1.0e-4f) / kFp8Max;
  if (round_scale) scale = exp2f(ceilf(log2f(scale)));
  *scale_output = scale;
  return fminf(fmaxf(transformed / scale, -kFp8Max), kFp8Max);
}

__global__ __launch_bounds__(kHeadDim) void GlmKPoolCompressKernel(
    const uint16_t* slot_key, const uint16_t* slot_score, const float* ape,
    const int64_t* locations, uint8_t* key_cache, float* scale_cache,
    uint64_t key_page_stride_bytes, uint64_t scale_page_stride_bytes,
    bool round_scale) {
  const uint32_t row = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  const uint64_t row_base = static_cast<uint64_t>(row) * kPoolSize * kHeadDim;

  float maximum = -__int_as_float(0x7f800000);
#pragma unroll
  for (uint32_t slot = 0; slot < kPoolSize; ++slot) {
    const uint64_t offset = row_base + slot * kHeadDim + dimension;
    maximum = fmaxf(maximum,
                    LoadBf16(slot_score + offset) +
                        ape[slot * kHeadDim + dimension]);
  }
  float accumulator = 0.0f;
  float denominator = 0.0f;
#pragma unroll
  for (uint32_t slot = 0; slot < kPoolSize; ++slot) {
    const uint64_t offset = row_base + slot * kHeadDim + dimension;
    const float probability = expf(LoadBf16(slot_score + offset) +
                                   ape[slot * kHeadDim + dimension] - maximum);
    denominator += probability;
    accumulator += LoadBf16(slot_key + offset) * probability;
  }

  __shared__ float shared[kHeadDim];
  float scale = 0.0f;
  const float quantized = HadamardQuantize(
      accumulator / denominator, shared, dimension, round_scale, &scale);

  const int64_t location = locations[row];
  if (location < 0) return;
  const uint64_t page = static_cast<uint64_t>(location) / kPageSize;
  const uint64_t page_offset = static_cast<uint64_t>(location) % kPageSize;
  const uint64_t key_offset =
      page * key_page_stride_bytes + page_offset * kHeadDim + dimension;
  key_cache[key_offset] = ToFp8(quantized);
  if (dimension == 0) {
    auto* scale_page = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(scale_cache) + page * scale_page_stride_bytes);
    scale_page[page_offset] = scale;
  }
}

__global__ __launch_bounds__(kHeadDim) void GlmKPoolDecodeKernel(
    uint16_t* tail_key, uint16_t* tail_score, const uint16_t* key,
    const uint16_t* score, const float* ape, const int32_t* block_tables,
    const int32_t* request_indices, const int64_t* positions,
    const int32_t* sequence_lengths, const int64_t* output_cache_locations,
    uint8_t* key_cache, float* scale_cache, uint32_t request_capacity,
    uint32_t tail_size, uint64_t block_table_stride,
    uint64_t key_page_stride_bytes, uint64_t scale_page_stride_bytes,
    bool round_scale) {
  const uint32_t row = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  const int32_t request_raw = request_indices[row];
  const int64_t position = positions[row];
  const int32_t sequence_length = sequence_lengths[row];
  const int64_t output_cache_location = output_cache_locations[row];
  const bool valid = request_raw >= 0 &&
                     static_cast<uint32_t>(request_raw) < request_capacity &&
                     output_cache_location != 0 && position >= 0 &&
                     position < sequence_length;
  if (!valid) return;

  const uint32_t request = static_cast<uint32_t>(request_raw);
  const uint64_t safe_position = static_cast<uint64_t>(position);
  const uint32_t slot = safe_position % kPoolSize;
  const uint32_t physical_slot = safe_position % tail_size;
  const uint64_t current_offset = static_cast<uint64_t>(row) * kHeadDim + dimension;
  const float current_key = LoadBf16(key + current_offset);
  const float current_score = LoadBf16(score + current_offset);

  if (slot == kPoolSize - 1) {
    const uint64_t pool_start = safe_position - slot;
    float maximum = -__int_as_float(0x7f800000);
#pragma unroll
    for (uint32_t pool_slot = 0; pool_slot < kPoolSize; ++pool_slot) {
      const uint32_t physical = (pool_start + pool_slot) % tail_size;
      const uint64_t tail_offset =
          (static_cast<uint64_t>(request) * tail_size + physical) * kHeadDim +
          dimension;
      const float buffered = LoadBf16(tail_score + tail_offset);
      const float value = pool_slot == slot ? current_score : buffered;
      maximum = fmaxf(maximum, value + ape[pool_slot * kHeadDim + dimension]);
    }
    float accumulator = 0.0f;
    float denominator = 0.0f;
#pragma unroll
    for (uint32_t pool_slot = 0; pool_slot < kPoolSize; ++pool_slot) {
      const uint32_t physical = (pool_start + pool_slot) % tail_size;
      const uint64_t tail_offset =
          (static_cast<uint64_t>(request) * tail_size + physical) * kHeadDim +
          dimension;
      const float buffered_score = LoadBf16(tail_score + tail_offset);
      const float selected_score =
          pool_slot == slot ? current_score : buffered_score;
      const float probability =
          expf(selected_score + ape[pool_slot * kHeadDim + dimension] - maximum);
      const float buffered_key = LoadBf16(tail_key + tail_offset);
      const float selected_key = pool_slot == slot ? current_key : buffered_key;
      denominator += probability;
      accumulator += selected_key * probability;
    }

    __shared__ float shared[kHeadDim];
    float scale = 0.0f;
    const float quantized = HadamardQuantize(
        accumulator / denominator, shared, dimension, round_scale, &scale);
    const uint64_t pool_id = safe_position / kPoolSize;
    const uint64_t pool_page_group = pool_id / kPageSize;
    const uint64_t token_page_row = pool_page_group * kPoolSize;
    if (token_page_row < block_table_stride) {
      const int32_t packed_page =
          block_tables[static_cast<uint64_t>(row) * block_table_stride +
                       token_page_row];
      if (packed_page >= 0) {
        const uint64_t page = static_cast<uint32_t>(packed_page);
        const uint64_t page_offset = pool_id % kPageSize;
        key_cache[page * key_page_stride_bytes + page_offset * kHeadDim +
                  dimension] = ToFp8(quantized);
        if (dimension == 0) {
          auto* scale_page = reinterpret_cast<float*>(
              reinterpret_cast<uint8_t*>(scale_cache) +
              page * scale_page_stride_bytes);
          scale_page[page_offset] = scale;
        }
      }
    }
  }

  const uint64_t tail_offset =
      (static_cast<uint64_t>(request) * tail_size + physical_slot) * kHeadDim +
      dimension;
  tail_key[tail_offset] = key[current_offset];
  tail_score[tail_offset] = score[current_offset];
}

// SGLang's GLM KPool query path first stores the quantized linear output as
// BF16, applies Tri Dao's 128-wide FP32 Hadamard butterfly, stores BF16 again,
// then runs SGLang's 16-lane per-group FP8 quantizer with a UE8M0-rounded
// (power-of-two, still represented as FP32) scale.
__global__ __launch_bounds__(kHeadDim) void GlmIndexerQueryPrepKernel(
    const float* query, const float* head_gate, uint8_t* query_fp8,
    float* query_scale, float* logit_weights, uint32_t heads) {
  const uint32_t row = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  const uint64_t base = static_cast<uint64_t>(row) * kHeadDim;
  __shared__ float shared[kHeadDim];
  __shared__ float row_scale;
  shared[dimension] = RoundBf16(query[base + dimension]);
  __syncthreads();
#pragma unroll
  for (uint32_t stride = 1; stride < kHeadDim; stride *= 2) {
    const float mine = shared[dimension];
    const float other = shared[dimension ^ stride];
    __syncthreads();
    shared[dimension] =
        (dimension & stride) == 0 ? mine + other : other - mine;
    __syncthreads();
  }
  shared[dimension] = RoundBf16(shared[dimension] * kHadamardScale);
  __syncthreads();

  if (dimension < 16) {
    float local_absmax = 1.0e-10f;
#pragma unroll
    for (uint32_t item = 0; item < 8; ++item) {
      local_absmax =
          fmaxf(local_absmax, fabsf(shared[dimension * 8 + item]));
    }
    constexpr unsigned kHalfWarpMask = 0x0000ffffu;
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(kHalfWarpMask, local_absmax, 8));
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(kHalfWarpMask, local_absmax, 4));
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(kHalfWarpMask, local_absmax, 2));
    local_absmax = fmaxf(
        local_absmax,
        __shfl_xor_sync(kHalfWarpMask, local_absmax, 1));
    if (dimension == 0) {
      const float raw_scale = local_absmax / kFp8Max;
      row_scale = exp2f(ceilf(log2f(fmaxf(raw_scale, 1.0e-10f))));
      query_scale[row] = row_scale;
    }
  }
  __syncthreads();
  const float scale = row_scale;
  if (dimension < 16) {
#pragma unroll
    for (uint32_t item = 0; item < 8; ++item) {
      const uint32_t index = dimension * 8 + item;
      const float quantized =
          fminf(fmaxf(shared[index] / scale, -kFp8Max), kFp8Max);
      query_fp8[base + index] = ToFp8(quantized);
    }
  }
  if (dimension == 0) {
    float weight = head_gate[row] * rsqrtf(static_cast<float>(heads));
    weight *= scale;
    weight *= kHadamardScale;
    logit_weights[row] = weight;
  }
}

__global__ __launch_bounds__(kHeadDim) void GlmIndexerKeyNormKernel(
    const float* key, const float* weight, const float* bias,
    uint16_t* key_bf16, float epsilon) {
  const uint32_t token = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  const uint64_t base = static_cast<uint64_t>(token) * kHeadDim;
  const float value = RoundBf16(key[base + dimension]);
  __shared__ float reduction[kHeadDim];
  reduction[dimension] = value;
  __syncthreads();
#pragma unroll
  for (uint32_t stride = kHeadDim / 2; stride > 0; stride /= 2) {
    if (dimension < stride) reduction[dimension] += reduction[dimension + stride];
    __syncthreads();
  }
  const float mean = reduction[0] / static_cast<float>(kHeadDim);
  const float centered = value - mean;
  reduction[dimension] = centered * centered;
  __syncthreads();
#pragma unroll
  for (uint32_t stride = kHeadDim / 2; stride > 0; stride /= 2) {
    if (dimension < stride) reduction[dimension] += reduction[dimension + stride];
    __syncthreads();
  }
  const float inverse = rsqrtf(
      reduction[0] / static_cast<float>(kHeadDim) + epsilon);
  key_bf16[base + dimension] =
      ToBf16(centered * inverse * weight[dimension] + bias[dimension]);
}

}  // namespace

extern "C" SparkServeStatus sparkserve_glm_kpool_compress_validate(
    const SparkServeGlmKPoolCompressArgs* args) {
  if (args == nullptr) return Invalid("GLM KPool compress args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_DSA_ABI_VERSION) {
    return Invalid("GLM KPool compress ABI mismatch");
  }
  if (args->rows == 0 || args->rows > 1U << 20 ||
      args->pool_size != kPoolSize || args->head_dim != kHeadDim ||
      args->page_size != kPageSize) {
    return Invalid("GLM KPool compress geometry must be rows x 4 x 128, page 64");
  }
  if (args->round_scale > 1 || args->reserved != 0) {
    return Invalid("GLM KPool compress flags are invalid");
  }
  if (args->slot_key_bf16 == nullptr || args->slot_score_bf16 == nullptr ||
      args->ape == nullptr || args->locations == nullptr ||
      args->key_cache_fp8 == nullptr || args->scale_cache == nullptr) {
    return Invalid("all GLM KPool compress tensor pointers are required");
  }
  if (args->key_page_stride_bytes < kMinKeyPageBytes ||
      args->scale_page_stride_bytes < kMinScalePageBytes ||
      args->scale_page_stride_bytes % alignof(float) != 0) {
    return Invalid("GLM KPool cache page strides are invalid");
  }
  if (static_cast<uint64_t>(args->rows) >
      std::numeric_limits<size_t>::max() / (kPoolSize * kHeadDim * sizeof(uint16_t))) {
    return Invalid("GLM KPool input geometry overflows size_t");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kpool_compress_launch(
    const SparkServeGlmKPoolCompressArgs* args) {
  const SparkServeStatus validation = sparkserve_glm_kpool_compress_validate(args);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  GlmKPoolCompressKernel<<<args->rows, kHeadDim, 0, stream>>>(
      args->slot_key_bf16, args->slot_score_bf16, args->ape, args->locations,
      args->key_cache_fp8, args->scale_cache, args->key_page_stride_bytes,
      args->scale_page_stride_bytes, args->round_scale != 0);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM KPool compress launch failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kpool_decode_validate(
    const SparkServeGlmKPoolDecodeArgs* args) {
  if (args == nullptr) return Invalid("GLM KPool decode args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_DSA_ABI_VERSION) {
    return Invalid("GLM KPool decode ABI mismatch");
  }
  if (args->rows == 0 || args->rows > 1U << 20 ||
      args->request_capacity == 0 || args->tail_size < kPoolSize ||
      args->head_dim != kHeadDim || args->pool_size != kPoolSize ||
      args->page_size != kPageSize || args->slots_per_page != kPageSize) {
    return Invalid("GLM KPool decode geometry is invalid");
  }
  if (args->round_scale > 1 || args->block_table_stride == 0) {
    return Invalid("GLM KPool decode flags or block-table stride are invalid");
  }
  if (args->tail_key_bf16 == nullptr || args->tail_score_bf16 == nullptr ||
      args->key_bf16 == nullptr || args->score_bf16 == nullptr ||
      args->ape == nullptr || args->block_tables == nullptr ||
      args->request_indices == nullptr || args->positions == nullptr ||
      args->sequence_lengths == nullptr ||
      args->output_cache_locations == nullptr ||
      args->key_cache_fp8 == nullptr || args->scale_cache == nullptr) {
    return Invalid("all GLM KPool decode tensor pointers are required");
  }
  if (args->key_page_stride_bytes < kMinKeyPageBytes ||
      args->scale_page_stride_bytes < kMinScalePageBytes ||
      args->scale_page_stride_bytes % alignof(float) != 0) {
    return Invalid("GLM KPool decode cache page strides are invalid");
  }
  const uint64_t tail_vectors =
      static_cast<uint64_t>(args->request_capacity) * args->tail_size;
  if (tail_vectors >
      std::numeric_limits<size_t>::max() / (kHeadDim * sizeof(uint16_t)) ||
      static_cast<uint64_t>(args->rows) >
          std::numeric_limits<size_t>::max() / (kHeadDim * sizeof(uint16_t))) {
    return Invalid("GLM KPool decode geometry overflows size_t");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kpool_decode_launch(
    const SparkServeGlmKPoolDecodeArgs* args) {
  const SparkServeStatus validation = sparkserve_glm_kpool_decode_validate(args);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  GlmKPoolDecodeKernel<<<args->rows, kHeadDim, 0, stream>>>(
      args->tail_key_bf16, args->tail_score_bf16, args->key_bf16,
      args->score_bf16, args->ape, args->block_tables,
      args->request_indices, args->positions, args->sequence_lengths,
      args->output_cache_locations, args->key_cache_fp8, args->scale_cache,
      args->request_capacity, args->tail_size, args->block_table_stride,
      args->key_page_stride_bytes, args->scale_page_stride_bytes,
      args->round_scale != 0);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM KPool decode launch failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_indexer_prep_validate(
    const SparkServeGlmIndexerPrepArgs* args) {
  if (args == nullptr) return Invalid("GLM indexer prep args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_DSA_ABI_VERSION) {
    return Invalid("GLM indexer prep ABI mismatch");
  }
  if (args->tokens == 0 || args->tokens > 1U << 20 ||
      args->heads != kIndexHeads || args->head_dim != kHeadDim) {
    return Invalid("GLM indexer prep geometry must be tokens x 32 x 128");
  }
  if (!std::isfinite(args->layer_norm_epsilon) ||
      args->layer_norm_epsilon <= 0.0f || args->round_scale != 1 ||
      args->reserved != 0) {
    return Invalid("GLM indexer prep scalar contract is invalid");
  }
  if (args->query_fp32 == nullptr || args->key_fp32 == nullptr ||
      args->key_norm_weight == nullptr || args->key_norm_bias == nullptr ||
      args->head_gate_fp32 == nullptr || args->query_fp8 == nullptr ||
      args->query_scale == nullptr || args->key_bf16 == nullptr ||
      args->logit_weights == nullptr) {
    return Invalid("all GLM indexer prep tensor pointers are required");
  }
  const uint64_t rows = static_cast<uint64_t>(args->tokens) * args->heads;
  if (rows > std::numeric_limits<uint32_t>::max() ||
      rows > std::numeric_limits<size_t>::max() / kHeadDim) {
    return Invalid("GLM indexer prep geometry overflows the launch contract");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_indexer_prep_launch(
    const SparkServeGlmIndexerPrepArgs* args) {
  const SparkServeStatus status = sparkserve_glm_indexer_prep_validate(args);
  if (status.code != SPARKSERVE_STATUS_OK) return status;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const uint32_t rows = args->tokens * args->heads;
  GlmIndexerKeyNormKernel<<<args->tokens, kHeadDim, 0, stream>>>(
      args->key_fp32, args->key_norm_weight, args->key_norm_bias,
      args->key_bf16, args->layer_norm_epsilon);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM indexer key LayerNorm failed: ", error);
  }
  GlmIndexerQueryPrepKernel<<<rows, kHeadDim, 0, stream>>>(
      args->query_fp32, args->head_gate_fp32, args->query_fp8,
      args->query_scale, args->logit_weights, args->heads);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM indexer query FP8 preparation failed: ", error);
  }
  return Ok();
}

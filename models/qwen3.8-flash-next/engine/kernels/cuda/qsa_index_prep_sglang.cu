// Adapted from SGLang's Apache-2.0 qsa_indexer.cuh at commit
// 7c66045d71f067c1c5da2b85baad3c47d9a19cb7. The locked Qwen3.8
// Flash-Next specialization is BF16, head_dim=128, four query heads, one key
// head, compression ratio four, and partial/full NeoX RoPE. TVM-FFI,
// sgl-kernel containers, framework allocation, and PDL dispatch are removed.

#include "internal/qsa_index_prep_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 4;
constexpr int kCompressRatio = 4;
constexpr int kWarpThreads = 32;
constexpr int kPerLane = kHeadDim / kWarpThreads;

thread_local std::string g_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }

FlashStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

__device__ float Bf16Float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __nv_bfloat16 Bf16(float value) {
  return __float2bfloat16_rn(value);
}

__device__ float EagerRound(float value) { return Bf16Float(Bf16(value)); }

__device__ float WarpReduceSum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffU, value, mask);
  }
  return value;
}

__device__ void GemmaNormRow(const __nv_bfloat16* input,
                             const __nv_bfloat16* weight, float eps,
                             __nv_bfloat16* shared_row) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpThreads - 1);
  float values[kPerLane];
#pragma unroll
  for (int item = 0; item < kPerLane; ++item) {
    values[item] = Bf16Float(input[lane * kPerLane + item]);
  }

  float sum_squares = 0.0f;
  if (lane < kHeadDim / 8) {
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = Bf16Float(input[lane * 8 + item]);
      sum_squares += value * value;
    }
  }
  sum_squares = WarpReduceSum(sum_squares);
  const float norm = rsqrtf(sum_squares / static_cast<float>(kHeadDim) + eps);
#pragma unroll
  for (int item = 0; item < kPerLane; ++item) {
    const int dimension = lane * kPerLane + item;
    const float scale = 1.0f + Bf16Float(weight[dimension]);
    shared_row[dimension] = Bf16(values[item] * norm * scale);
  }
  __syncwarp();
}

__device__ void ApplyNeoxMrope(const __nv_bfloat16* shared_row,
                               __nv_bfloat16* output,
                               const float* cos_sin_cache,
                               const int32_t* axis_map,
                               const int64_t positions[3], int rotary_dim) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpThreads - 1);
  const int half = rotary_dim / 2;
#pragma unroll
  for (int item = 0; item < kPerLane; ++item) {
    const int dimension = lane * kPerLane + item;
    if (dimension < half) {
      const int pair = dimension + half;
      const float* cache_row =
          cos_sin_cache + positions[axis_map[dimension]] * rotary_dim;
      const float cosine = EagerRound(cache_row[dimension]);
      const float sine = EagerRound(cache_row[half + dimension]);
      const float current = Bf16Float(shared_row[dimension]);
      const float paired = Bf16Float(shared_row[pair]);
      output[dimension] =
          Bf16(EagerRound(current * cosine) - EagerRound(paired * sine));
    } else if (dimension < rotary_dim) {
      const int pair = dimension - half;
      const float* cache_row =
          cos_sin_cache + positions[axis_map[pair]] * rotary_dim;
      const float cosine = EagerRound(cache_row[pair]);
      const float sine = EagerRound(cache_row[half + pair]);
      const float current = Bf16Float(shared_row[dimension]);
      const float paired = Bf16Float(shared_row[pair]);
      output[dimension] =
          Bf16(EagerRound(current * cosine) + EagerRound(paired * sine));
    } else {
      output[dimension] = shared_row[dimension];
    }
  }
}

struct QPrepParams {
  const __nv_bfloat16* qk;
  __nv_bfloat16* q_output;
  const __nv_bfloat16* weight;
  const float* cos_sin_cache;
  const int32_t* axis_map;
  const int64_t* positions;
  const int64_t* cache_locs;
  __nv_bfloat16* key_state;
  int64_t* rope_positions;
  uint64_t positions_stride;
  uint32_t num_position_axes;
  uint32_t q_heads_padded;
  int rotary_dim;
  float eps;
};

__global__ __launch_bounds__(128) void QsaQPrepKernel(QPrepParams params) {
  const uint32_t token = blockIdx.x;
  const int warp = static_cast<int>(threadIdx.x) / kWarpThreads;
  const int lane = static_cast<int>(threadIdx.x) & (kWarpThreads - 1);
  __shared__ __nv_bfloat16 shared_rows[kQueryHeads][kHeadDim];

  const int64_t qk_row =
      static_cast<int64_t>(token) * (kQueryHeads + 1) * kHeadDim;
  const int64_t location = params.cache_locs[token];
  int64_t positions[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const uint32_t source_axis =
        axis < static_cast<int>(params.num_position_axes) ? axis : 0;
    positions[axis] = params.positions[
        static_cast<uint64_t>(source_axis) * params.positions_stride + token];
  }

  for (uint32_t head = warp; head < params.q_heads_padded; head += 4) {
    __nv_bfloat16* output_row =
        params.q_output +
        (static_cast<int64_t>(token) * params.q_heads_padded + head) * kHeadDim;
    if (head < kQueryHeads) {
      const __nv_bfloat16* input_row =
          params.qk + qk_row + static_cast<int64_t>(head) * kHeadDim;
      GemmaNormRow(input_row, params.weight, params.eps, shared_rows[warp]);
      ApplyNeoxMrope(shared_rows[warp], output_row, params.cos_sin_cache,
                     params.axis_map, positions, params.rotary_dim);
    } else {
#pragma unroll
      for (int item = 0; item < kPerLane; ++item) {
        output_row[lane * kPerLane + item] = Bf16(0.0f);
      }
    }
  }

  if (warp == 0) {
#pragma unroll
    for (int item = 0; item < kPerLane; ++item) {
      const int dimension = lane * kPerLane + item;
      params.key_state[location * kHeadDim + dimension] =
          params.qk[qk_row + kQueryHeads * kHeadDim + dimension];
    }
  }
  if (warp == 1 && lane < 3) {
    params.rope_positions[location * 3 + lane] = positions[lane];
  }
}

struct KCompressParams {
  const __nv_bfloat16* key_state;
  const int32_t* group_locs;
  const int64_t* rope_positions;
  const float* cos_sin_cache;
  const int32_t* axis_map;
  const __nv_bfloat16* weight;
  const int32_t* write_locs;
  __nv_bfloat16* compressed_keys;
  uint32_t groups;
  int rotary_dim;
  float eps;
};

__global__ __launch_bounds__(128) void QsaKCompressKernel(
    KCompressParams params) {
  const int warp = static_cast<int>(threadIdx.x) / kWarpThreads;
  const int lane = static_cast<int>(threadIdx.x) & (kWarpThreads - 1);
  const uint32_t group = blockIdx.x * 4 + warp;
  if (group >= params.groups) return;
  __shared__ __nv_bfloat16 shared_rows[4][kHeadDim];
  const int32_t* locations = params.group_locs + group * kCompressRatio;

  float means[kPerLane] = {};
  for (int member = 0; member < kCompressRatio; ++member) {
#pragma unroll
    for (int item = 0; item < kPerLane; ++item) {
      const int dimension = lane * kPerLane + item;
      means[item] += Bf16Float(
          params.key_state[static_cast<int64_t>(locations[member]) * kHeadDim +
                           dimension]);
    }
  }
#pragma unroll
  for (int item = 0; item < kPerLane; ++item) {
    const int dimension = lane * kPerLane + item;
    shared_rows[warp][dimension] = Bf16(means[item] / kCompressRatio);
    means[item] = Bf16Float(shared_rows[warp][dimension]);
  }
  __syncwarp();

  float sum_squares = 0.0f;
  if (lane < kHeadDim / 8) {
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = Bf16Float(shared_rows[warp][lane * 8 + item]);
      sum_squares += value * value;
    }
  }
  sum_squares = WarpReduceSum(sum_squares);
  const float norm =
      rsqrtf(sum_squares / static_cast<float>(kHeadDim) + params.eps);
#pragma unroll
  for (int item = 0; item < kPerLane; ++item) {
    const int dimension = lane * kPerLane + item;
    const float scale = 1.0f + Bf16Float(params.weight[dimension]);
    shared_rows[warp][dimension] = Bf16(means[item] * norm * scale);
  }
  __syncwarp();

  int64_t positions[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    positions[axis] = params.rope_positions[
        static_cast<int64_t>(locations[0]) * 3 + axis];
  }
  __nv_bfloat16* output =
      params.compressed_keys +
      static_cast<int64_t>(params.write_locs[group]) * kHeadDim;
  ApplyNeoxMrope(shared_rows[warp], output, params.cos_sin_cache,
                 params.axis_map, positions, params.rotary_dim);
}

}  // namespace

FlashStatus flash_sglang_qsa_index_prep_cuda_launch(
    const FlashQsaIndexPrepArgs* args) {
  const QPrepParams q_params = {
      static_cast<const __nv_bfloat16*>(args->qk),
      static_cast<__nv_bfloat16*>(args->q_output),
      static_cast<const __nv_bfloat16*>(args->q_norm_weight),
      args->cos_sin_cache,
      args->axis_map,
      args->positions,
      args->cache_locs,
      static_cast<__nv_bfloat16*>(args->key_state),
      args->rope_positions,
      args->positions_stride,
      args->plan.num_position_axes,
      args->plan.q_heads_padded,
      static_cast<int>(args->plan.rotary_dim),
      args->eps,
  };
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  QsaQPrepKernel<<<args->plan.tokens, 128, 0, stream>>>(q_params);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("QSA Q prep launch failed: ", error);

  if (args->plan.groups != 0) {
    const KCompressParams k_params = {
        static_cast<const __nv_bfloat16*>(args->key_state),
        args->group_locs,
        args->rope_positions,
        args->cos_sin_cache,
        args->axis_map,
        static_cast<const __nv_bfloat16*>(args->k_norm_weight),
        args->write_locs,
        static_cast<__nv_bfloat16*>(args->compressed_keys),
        args->plan.groups,
        static_cast<int>(args->plan.rotary_dim),
        args->eps,
    };
    const uint32_t blocks = (args->plan.groups + 3) / 4;
    QsaKCompressKernel<<<blocks, 128, 0, stream>>>(k_params);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaError("QSA K compression launch failed: ", error);
    }
  }
  return Ok();
}

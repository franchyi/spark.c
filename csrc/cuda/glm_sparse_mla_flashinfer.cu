// Narrow, framework-free adapter around FlashInfer's BSD-3-Clause
// sparse_mla_sm120 GLM_NSA decode kernel at commit
// 906181e3f4cf4bcc81835fb480db4011bbd80b62. The tensor-core kernels and
// split merge below are direct template instantiations of the pinned donor;
// SparkServe adds only no-RoPE packing, KPool segmentation, validation, and a
// two-segment LSE merge. No Torch, Python, TVM-FFI, or serving-time JIT enters
// this translation unit.

#include "sparkserve/glm_sparse_mla_api.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <flashinfer/attention/sparse_mla_sm120/arch/mma_sm120.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/scale_convert.cuh>

// GB10/SM121 supports ordinary FP8 MMA but not the SM120 data-center
// `kind::mxf8f6f4.block_scale` instruction used by the donor's QK leaf. The
// exact equivalent is a zero-C ordinary MMA followed by the two UE8M0 scales
// in FP32 and an add to C. Include the donor primitives first so its own
// function body remains untouched, then redirect only calls in the sparse MLA
// templates to this GB10 adapter.
__device__ __forceinline__ MmaFp8Result SparkServeMmaFp8BlockScaledSm121(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, float c0, float c1, float c2, float c3, uint8_t scale_a,
    uint8_t scale_b) {
  const MmaFp8Result product =
      mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, 0.0F, 0.0F, 0.0F, 0.0F);
  const float scale = ue8m0_to_fp32(scale_a) * ue8m0_to_fp32(scale_b);
  return {c0 + product.d0 * scale, c1 + product.d1 * scale,
          c2 + product.d2 * scale, c3 + product.d3 * scale};
}

#define mma_fp8_block_scaled_m16n8k32 SparkServeMmaFp8BlockScaledSm121
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh>
#undef mma_fp8_block_scaled_m16n8k32

namespace {

namespace donor = flashinfer::sparse_mla_sm120;

constexpr uint32_t kMaxBatch = 32;
constexpr uint32_t kHeads = SPARKSERVE_GLM_SPARSE_MLA_HEADS;
constexpr uint32_t kLatentDim = SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM;
constexpr uint32_t kPaddedQueryDim = SPARKSERVE_GLM_SPARSE_MLA_PADDED_Q_DIM;
constexpr uint32_t kPageSize = SPARKSERVE_GLM_SPARSE_MLA_PAGE_SIZE;
constexpr uint32_t kTokenBytes = SPARKSERVE_GLM_SPARSE_MLA_TOKEN_BYTES;
constexpr uint32_t kHistoryTopk = SPARKSERVE_GLM_SPARSE_MLA_HISTORY_TOPK;
constexpr uint32_t kTailTopk = SPARKSERVE_GLM_SPARSE_MLA_TAIL_TOPK;
constexpr uint32_t kHistorySplits = SPARKSERVE_GLM_SPARSE_MLA_HISTORY_SPLITS;
constexpr uint32_t kTailSplits = SPARKSERVE_GLM_SPARSE_MLA_TAIL_SPLITS;
constexpr uint32_t kSelectionWidth = SPARKSERVE_GLM_SPARSE_MLA_SELECTION_WIDTH;
constexpr uint32_t kNumSms = SPARKSERVE_GLM_SPARSE_MLA_GB10_SMS;
constexpr uint32_t kQuantGroup = 128;
constexpr uint32_t kScales = kLatentDim / kQuantGroup;
constexpr uint64_t kMinPageBytes = static_cast<uint64_t>(kPageSize) * kTokenBytes;
constexpr float kFp8Max = 448.0F;
constexpr float kKvScaleEpsilon = 1.0e-8F;

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus Unsupported(const char* message) {
  return {SPARKSERVE_STATUS_UNSUPPORTED, message};
}

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

bool IsAligned(const void* pointer, uintptr_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

__device__ __forceinline__ float WarpMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__global__ __launch_bounds__(kQuantGroup) void PackKvKernel(
    const __nv_bfloat16* input, const int32_t* locations, uint8_t* cache,
    uint32_t tokens, uint32_t num_pages, uint64_t page_stride_bytes) {
  const uint32_t token = blockIdx.x;
  const uint32_t group = blockIdx.y;
  const uint32_t lane = threadIdx.x;
  if (token >= tokens || group >= kScales) return;

  const int32_t location = locations[token];
  const uint32_t total_slots = num_pages * kPageSize;
  if (location < 0 || static_cast<uint32_t>(location) >= total_slots) return;

  const uint32_t page = static_cast<uint32_t>(location) / kPageSize;
  const uint32_t slot = static_cast<uint32_t>(location) % kPageSize;
  uint8_t* row = cache + static_cast<uint64_t>(page) * page_stride_bytes +
                 static_cast<uint64_t>(slot) * kTokenBytes;

  const uint32_t column = group * kQuantGroup + lane;
  const float value = __bfloat162float(input[static_cast<uint64_t>(token) * kLatentDim + column]);
  float maximum = WarpMax(fabsf(value));
  __shared__ float warp_maxima[4];
  __shared__ float scale;
  if ((lane & 31U) == 0) warp_maxima[lane >> 5U] = maximum;
  __syncthreads();
  if (lane < 32) {
    maximum = lane < 4 ? warp_maxima[lane] : 0.0F;
    maximum = WarpMax(maximum);
    if (lane == 0) scale = fmaxf(maximum, kKvScaleEpsilon) / kFp8Max;
  }
  __syncthreads();

  const float quantized = fmaxf(-kFp8Max, fminf(kFp8Max, value / scale));
  const __nv_fp8_e4m3 fp8_value(quantized);
  row[column] = fp8_value.__x;
  if (lane == 0) {
    reinterpret_cast<float*>(row + kLatentDim)[group] = scale;
  }
  if (group == 0) {
    // GLM-5.3 has no RoPE. FlashInfer's fixed V32 ABI still gathers these
    // 64 BF16 lanes, so materialize them once as exact zeros.
    row[kLatentDim + kScales * sizeof(float) + lane] = 0;
  }
}

__global__ __launch_bounds__(256) void PadQueryKernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t rows) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  for (uint32_t column = threadIdx.x; column < kPaddedQueryDim;
       column += blockDim.x) {
    output[static_cast<uint64_t>(row) * kPaddedQueryDim + column] =
        column < kLatentDim
            ? input[static_cast<uint64_t>(row) * kLatentDim + column]
            : __float2bfloat16(0.0F);
  }
}

__global__ __launch_bounds__(256) void SegmentSelectionKernel(
    const int32_t* selected, const int64_t* query_positions,
    const int32_t* sequence_lengths, int32_t* history, int32_t* tail,
    int32_t* history_lengths, int32_t* tail_lengths, uint32_t batch_size,
    uint32_t selected_stride) {
  const uint32_t row = blockIdx.x;
  if (row >= batch_size) return;
  const int64_t raw_sequence_length = static_cast<int64_t>(sequence_lengths[row]);
  const int64_t sequence_length = raw_sequence_length > 0 ? raw_sequence_length : 0;
  const int64_t raw_visible = query_positions[row] + 1;
  const int64_t nonnegative_visible = raw_visible > 0 ? raw_visible : 0;
  const int64_t visible =
      nonnegative_visible < sequence_length ? nonnegative_visible : sequence_length;
  const int32_t history_length = static_cast<int32_t>(
      min((visible / 4) * 4, static_cast<int64_t>(kHistoryTopk)));
  const int32_t tail_length = static_cast<int32_t>(visible % 4);
  if (threadIdx.x == 0) {
    history_lengths[row] = history_length;
    tail_lengths[row] = tail_length;
  }
  const int32_t* source = selected + static_cast<uint64_t>(row) * selected_stride;
  int32_t* history_row = history + static_cast<uint64_t>(row) * kHistoryTopk;
  int32_t* tail_row = tail + static_cast<uint64_t>(row) * kTailTopk;
  for (uint32_t index = threadIdx.x; index < kHistoryTopk; index += blockDim.x) {
    history_row[index] = index < static_cast<uint32_t>(history_length) ? source[index] : -1;
  }
  for (uint32_t index = threadIdx.x; index < kTailTopk; index += blockDim.x) {
    tail_row[index] = index < static_cast<uint32_t>(tail_length)
                          ? source[static_cast<uint32_t>(history_length) + index]
                          : -1;
  }
}

template <int Topk, int Splits>
cudaError_t LaunchDonor(const __nv_bfloat16* query, const uint8_t* cache,
                        const int32_t* indices, __nv_bfloat16* mid_out,
                        float* mid_lse, const int32_t* topk_lengths,
                        __nv_bfloat16* output, float* output_lse,
                        uint32_t batch_size, float softmax_scale,
                        uint64_t page_stride_bytes, cudaStream_t stream) {
  using KV = KVCacheTraits<ModelType::GLM_NSA>;
  static_assert(KV::D_QK == static_cast<int>(kPaddedQueryDim));
  static_assert(KV::D_V == static_cast<int>(kLatentDim));
  static_assert(KV::KV_GMEM_STRIDE == static_cast<int>(kTokenBytes));
  constexpr int kHeadBlocks = kHeads / HPB;
  constexpr int kValueChunks = KV::D_NOPE / KV::QUANT_TILE;
  constexpr int kDynamicSmem =
      HPB * KV::D_ROPE * static_cast<int>(sizeof(bf16)) +
      HPB * KV::Q_NOPE_STRIDE +
      HPB * KV::NUM_SCALES * static_cast<int>(sizeof(float)) +
      donor::DSV3_2_KV_BUF_COUNT * donor::DSV3_2_BI * KV::KV_SMEM_STRIDE +
      donor::DSV3_2_KV_BUF_COUNT * donor::DSV3_2_BI * KV::D_ROPE *
          static_cast<int>(sizeof(bf16)) +
      16 + 4 * static_cast<int>(sizeof(uint64_t)) +
      2 * donor::DSV3_2_N_WARPS * HPB * static_cast<int>(sizeof(float)) +
      kValueChunks * HPB * static_cast<int>(sizeof(float)) +
      2 * HPB * (donor::DSV3_2_BI + 16);

  auto kernel = donor::sparse_mla_decode_dsv3_2_kernel<
      ModelType::GLM_NSA, kHeads, Topk, kPageSize>;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmem);
  if (error != cudaSuccess) return error;

  // One 64-candidate chunk per split is a fixed GB10 schedule: 128 CTAs for
  // batch-one history and eight CTAs for its tail. It is CUDA-graph stable.
  kernel<<<dim3(batch_size, kHeadBlocks, Splits),
           dim3(donor::DSV3_2_BLOCK_THREADS), kDynamicSmem, stream>>>(
      reinterpret_cast<const bf16*>(query), cache, indices,
      reinterpret_cast<bf16*>(mid_out), mid_lse, topk_lengths,
      static_cast<int>(batch_size), Splits, 1, softmax_scale,
      static_cast<size_t>(page_stride_bytes));
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  constexpr int kMergeThreads = 64;
  constexpr int kDimsPerThread = kLatentDim / kMergeThreads;
  auto merge = donor::sparse_mla_decode_dsv4_merge_kernel<
      kHeads, kLatentDim, kMergeThreads, kDimsPerThread>;
  merge<<<dim3(batch_size, kHeads), dim3(kMergeThreads),
          Splits * sizeof(float), stream>>>(
      reinterpret_cast<const bf16*>(mid_out), mid_lse,
      reinterpret_cast<bf16*>(output), output_lse, nullptr,
      static_cast<int>(batch_size), Splits);
  return cudaGetLastError();
}

__global__ __launch_bounds__(256) void MergeSegmentsKernel(
    __nv_bfloat16* history_output, float* history_lse,
    const __nv_bfloat16* tail_output, const float* tail_lse,
    uint32_t batch_size) {
  const uint32_t token = blockIdx.x;
  const uint32_t head = blockIdx.y;
  if (token >= batch_size || head >= kHeads) return;
  const uint64_t row = static_cast<uint64_t>(token) * kHeads + head;
  const float left_lse = history_lse[row];
  const float right_lse = tail_lse[row];
  const bool left_valid = left_lse > -1.0e29F;
  const bool right_valid = right_lse > -1.0e29F;
  float maximum = 0.0F;
  float left_weight = 0.0F;
  float right_weight = 0.0F;
  float inverse = 0.0F;
  if (left_valid || right_valid) {
    maximum = left_valid && right_valid ? fmaxf(left_lse, right_lse)
                                        : (left_valid ? left_lse : right_lse);
    left_weight = left_valid ? exp2f(left_lse - maximum) : 0.0F;
    right_weight = right_valid ? exp2f(right_lse - maximum) : 0.0F;
    inverse = 1.0F / (left_weight + right_weight);
  }
  __nv_bfloat16* destination = history_output + row * kLatentDim;
  const __nv_bfloat16* right = tail_output + row * kLatentDim;
  for (uint32_t column = threadIdx.x; column < kLatentDim;
       column += blockDim.x) {
    const float left_value = left_valid ? __bfloat162float(destination[column]) : 0.0F;
    const float right_value = right_valid ? __bfloat162float(right[column]) : 0.0F;
    destination[column] = __float2bfloat16(
        (left_value * left_weight + right_value * right_weight) * inverse);
  }
  if (threadIdx.x == 0) {
    history_lse[row] = (left_valid || right_valid)
                           ? maximum + log2f(left_weight + right_weight)
                           : -1.0e30F;
  }
}

SparkServeStatus ValidateGb10() {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return CudaError("GLM sparse MLA device query failed: ", error);
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return CudaError("GLM sparse MLA device properties failed: ", error);
  }
  if (properties.major != 12 || properties.minor != 1 ||
      properties.multiProcessorCount != static_cast<int>(kNumSms)) {
    return Unsupported("GLM sparse MLA donor is specialized to 48-SM GB10/SM121");
  }
  return Ok();
}

}  // namespace

extern "C" SparkServeStatus sparkserve_glm_sparse_mla_pack_kv_validate(
    const SparkServeGlmSparseMlaPackKvArgs* args) {
  if (args == nullptr) return Invalid("GLM sparse MLA KV-pack args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_SPARSE_MLA_ABI_VERSION) {
    return Invalid("GLM sparse MLA KV-pack ABI mismatch");
  }
  if (args->tokens == 0 || args->page_size != kPageSize ||
      args->latent_dim != kLatentDim || args->quant_group != kQuantGroup ||
      args->num_pages == 0 || args->reserved != 0) {
    return Invalid("GLM sparse MLA KV-pack geometry must be 512-wide, group 128, page 64");
  }
  if (args->num_pages > std::numeric_limits<uint32_t>::max() / kPageSize ||
      args->input_bf16 == nullptr || args->locations == nullptr ||
      args->cache == nullptr || !IsAligned(args->input_bf16, 16) ||
      !IsAligned(args->cache, 16) || args->page_stride_bytes < kMinPageBytes ||
      args->page_stride_bytes % 16 != 0) {
    return Invalid("GLM sparse MLA KV-pack pointers or page stride are invalid");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_sparse_mla_pack_kv_launch(
    const SparkServeGlmSparseMlaPackKvArgs* args) {
  SparkServeStatus status = sparkserve_glm_sparse_mla_pack_kv_validate(args);
  if (status.code != SPARKSERVE_STATUS_OK) return status;
  PackKvKernel<<<dim3(args->tokens, kScales), dim3(kQuantGroup), 0,
                 static_cast<cudaStream_t>(args->cuda_stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(args->input_bf16), args->locations,
      args->cache, args->tokens, args->num_pages, args->page_stride_bytes);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("GLM sparse MLA KV-pack launch failed: ", error);
}

extern "C" SparkServeStatus sparkserve_glm_sparse_mla_pad_query_validate(
    const SparkServeGlmSparseMlaPadQueryArgs* args) {
  if (args == nullptr) return Invalid("GLM sparse MLA query-pad args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_SPARSE_MLA_ABI_VERSION) {
    return Invalid("GLM sparse MLA query-pad ABI mismatch");
  }
  if (args->batch_size == 0 || args->batch_size > kMaxBatch ||
      args->num_heads != kHeads || args->input_dim != kLatentDim ||
      args->padded_dim != kPaddedQueryDim || args->reserved0 != 0 ||
      args->reserved1 != 0 || args->input_bf16 == nullptr ||
      args->output_bf16 == nullptr || !IsAligned(args->input_bf16, 16) ||
      !IsAligned(args->output_bf16, 16)) {
    return Invalid("GLM sparse MLA query-pad geometry or pointers are invalid");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_sparse_mla_pad_query_launch(
    const SparkServeGlmSparseMlaPadQueryArgs* args) {
  SparkServeStatus status = sparkserve_glm_sparse_mla_pad_query_validate(args);
  if (status.code != SPARKSERVE_STATUS_OK) return status;
  PadQueryKernel<<<args->batch_size * kHeads, 256, 0,
                   static_cast<cudaStream_t>(args->cuda_stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(args->input_bf16),
      reinterpret_cast<__nv_bfloat16*>(args->output_bf16),
      args->batch_size * kHeads);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("GLM sparse MLA query-pad launch failed: ", error);
}

extern "C" SparkServeStatus sparkserve_glm_sparse_mla_decode_validate(
    const SparkServeGlmSparseMlaDecodeArgs* args) {
  if (args == nullptr) return Invalid("GLM sparse MLA decode args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_SPARSE_MLA_ABI_VERSION) {
    return Invalid("GLM sparse MLA decode ABI mismatch");
  }
  if (args->batch_size == 0 || args->batch_size > kMaxBatch ||
      args->num_heads != kHeads || args->query_dim != kPaddedQueryDim ||
      args->value_dim != kLatentDim || args->page_size != kPageSize ||
      args->history_topk != kHistoryTopk || args->tail_topk != kTailTopk ||
      args->history_splits != kHistorySplits || args->tail_splits != kTailSplits ||
      args->num_pages == 0 || args->num_sms != kNumSms ||
      args->selected_stride != kSelectionWidth || args->reserved0 != 0 ||
      args->reserved1 != 0 || args->reserved2 != 0 ||
      !std::isfinite(args->softmax_scale) || args->softmax_scale <= 0.0F) {
    return Invalid("GLM sparse MLA decode geometry must match the fixed GB10 GLM-5.3 plan");
  }
  if (args->num_pages > std::numeric_limits<uint32_t>::max() / kPageSize ||
      args->page_stride_bytes < kMinPageBytes || args->page_stride_bytes % 16 != 0) {
    return Invalid("GLM sparse MLA page count or stride is invalid");
  }
  const void* pointers[] = {
      args->query_bf16,          args->cache,
      args->selected_indices,    args->query_positions,
      args->sequence_lengths,    args->history_indices,
      args->tail_indices,        args->history_lengths,
      args->tail_lengths,        args->history_mid_out_bf16,
      args->history_mid_lse,     args->output_bf16,
      args->output_lse,          args->tail_mid_out_bf16,
      args->tail_mid_lse,        args->tail_output_bf16,
      args->tail_output_lse,
  };
  for (const void* pointer : pointers) {
    if (pointer == nullptr) return Invalid("all GLM sparse MLA device pointers are required");
  }
  for (const void* pointer : {static_cast<const void*>(args->query_bf16),
                              static_cast<const void*>(args->cache),
                              static_cast<const void*>(args->history_mid_out_bf16),
                              static_cast<const void*>(args->output_bf16),
                              static_cast<const void*>(args->tail_mid_out_bf16),
                              static_cast<const void*>(args->tail_output_bf16)}) {
    if (!IsAligned(pointer, 16)) return Invalid("GLM sparse MLA tensor pointers must be 16-byte aligned");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_sparse_mla_decode_launch(
    const SparkServeGlmSparseMlaDecodeArgs* args) {
  SparkServeStatus status = sparkserve_glm_sparse_mla_decode_validate(args);
  if (status.code != SPARKSERVE_STATUS_OK) return status;
  status = ValidateGb10();
  if (status.code != SPARKSERVE_STATUS_OK) return status;
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  SegmentSelectionKernel<<<args->batch_size, 256, 0, stream>>>(
      args->selected_indices, args->query_positions, args->sequence_lengths,
      args->history_indices, args->tail_indices, args->history_lengths,
      args->tail_lengths, args->batch_size, args->selected_stride);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM sparse MLA selection segmentation failed: ", error);
  }

  error = LaunchDonor<kHistoryTopk, kHistorySplits>(
      reinterpret_cast<const __nv_bfloat16*>(args->query_bf16), args->cache,
      args->history_indices,
      reinterpret_cast<__nv_bfloat16*>(args->history_mid_out_bf16),
      args->history_mid_lse, args->history_lengths,
      reinterpret_cast<__nv_bfloat16*>(args->output_bf16), args->output_lse,
      args->batch_size, args->softmax_scale, args->page_stride_bytes, stream);
  if (error != cudaSuccess) {
    return CudaError("FlashInfer GLM_NSA history decode failed: ", error);
  }
  error = LaunchDonor<kTailTopk, kTailSplits>(
      reinterpret_cast<const __nv_bfloat16*>(args->query_bf16), args->cache,
      args->tail_indices,
      reinterpret_cast<__nv_bfloat16*>(args->tail_mid_out_bf16),
      args->tail_mid_lse, args->tail_lengths,
      reinterpret_cast<__nv_bfloat16*>(args->tail_output_bf16),
      args->tail_output_lse, args->batch_size, args->softmax_scale,
      args->page_stride_bytes, stream);
  if (error != cudaSuccess) {
    return CudaError("FlashInfer GLM_NSA tail decode failed: ", error);
  }

  MergeSegmentsKernel<<<dim3(args->batch_size, kHeads), 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(args->output_bf16), args->output_lse,
      reinterpret_cast<const __nv_bfloat16*>(args->tail_output_bf16),
      args->tail_output_lse, args->batch_size);
  error = cudaGetLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("GLM sparse MLA segment merge failed: ", error);
}
